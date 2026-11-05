"""What the validation samples from: prompts, generations and judgements. Implemented in P04.

Codes against lift.schemas and the CLAUDE.md file layout, not against the harness code:

- Prompt files are read through lift.data.build.load_prompts, the reader of every stage, so a pid's
  split is the committed split file's (data/splits): a file stamped from another assignment, or a
  missing split file, is refused (``Prompts``). A cell short of its frozen split is still read.
- A generation is *current* when its ``prompt_sha`` equals ``sha256_text`` of the prompt text now
  in data/prompts. A translation fixed after a run (the native spot checks) leaves a stale record
  behind until the harness regenerates the file, which it does for the whole file: the other
  prompts come back under the same keys, possibly with new text. A pid whose only records are stale
  is left out of the pool and counted. If no record of a file matches its prompts at all, the
  hashing convention is wrong, which is an error.
- Records in a precision other than ``generation.dtype`` (meta "dtype", e.g. a T4 fp16 check) are
  left out and counted, unless that precision is asked for. Naming one licenses nothing: records
  read in it must be covered by Gate 0's licence (decisions.md 27), checked by the harness's own
  rule (``check_licence``: gate0-{dtype}.json and its bf16 gate0.json, and the harness, decoding
  settings and GPU class the cell was made and judged with), or the read is a LicenceError. Two
  current records for one pid (two runs mixed in one file) are an error.
- A pid of the split with no record at all means the run is unfinished. The harness batches by
  length, so a partial run holds only the shortest prompts, and sampling from it would bias the
  sheet. Callers refuse such files.
- Judgements are found by generation key in every directory under outputs/judgements, and one counts
  only if its ``output_sha`` (P02 records the sha256 of the output it judged; keys name inputs only)
  is that of the output the caller holds for the key. A judgement of another output (the harness
  regenerated the file after judging it) or one that does not say is skipped and counted (``stale``),
  as P02's pipeline and sync refuse to pair it. The judge
  is told apart by the ``judge`` id in each record, not by the directory name or its config role:
  Qwen3Guard is the id (before any "@revision") of whichever judges block names Qwen3Guard, and
  any other judge is the frontier judge. This keeps working after the Nov 6 decision swaps the
  primary and second blocks. Either can be pinned by id when more than one is found.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pydantic

from lift import config, paths
from lift.io import sha256_text
from lift.schemas import GenerationRecord, JudgementRecord, PromptRecord
from lift.validation.store import ValidateError

SAFETY_SETS = ("harmful", "easy_benign", "hard_benign")
REPAIR_METHODS = ("wiseft", "resta", "cast", "cast_always")
QWEN3GUARD = "Qwen/Qwen3Guard-Gen-4B"

_ADAPTER = r"adapter-[A-Za-z0-9]+-(?P<recipe>[A-Za-z0-9]+)-s(?P<seed>\d+)"
_NAME_SCHEME = (   # configs/variants.yaml documents these names
    (re.compile(r"^stock$"), "stock"),
    (re.compile(rf"^{_ADAPTER}$"), "adapter"),
    (re.compile(rf"^wiseft-l[0-9.]+-{_ADAPTER}$"), "wiseft"),
    (re.compile(rf"^resta-a[0-9.]+-{_ADAPTER}$"), "resta"),
    (re.compile(rf"^cast-m[0-9.]+-{_ADAPTER}$"), "cast"),
    (re.compile(rf"^castalways-m[0-9.]+-{_ADAPTER}$"), "cast_always"),
)
_NL, _CR = chr(10), chr(13)
_DTYPES = {"bfloat16": "bf16", "float16": "fp16", "half": "fp16", "float32": "fp32", "float": "fp32"}


@dataclass(frozen=True)
class VariantInfo:
    name: str
    method: str
    seed: int | None
    recipe: str | None = None        # the adapter's training recipe (std, stress), from its name

    @property
    def group(self) -> str:
        return self.method if self.method in ("stock", "adapter") else "repaired"


def variant_info(name: str) -> VariantInfo | None:
    """Method and seed of a variant: from configs/variants.yaml when it is registered there,
    otherwise from the naming scheme that file documents. None if neither knows the name."""
    parsed = None
    for pattern, method in _NAME_SCHEME:
        m = pattern.match(name)
        if m:
            seed = m.groupdict().get("seed")
            parsed = VariantInfo(name, method, int(seed) if seed else None, m.groupdict().get("recipe"))
            break
    for entry in config.load("variants").get("variants") or []:
        if entry.get("name") == name:
            seed = entry.get("seed")
            if seed is None and parsed is not None:
                seed = parsed.seed
            return VariantInfo(name, str(entry["method"]), None if seed is None else int(seed),
                               parsed.recipe if parsed else None)
    return parsed


def variants_with_outputs() -> list[str]:
    root = paths.generations_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def read_records(path: Path) -> Iterator[tuple[int, dict]]:
    """(line number, row) for each record of a JSONL file.

    Lines end at a newline only, as lift.io reads them (stable_json leaves U+2028 and friends
    unescaped inside strings, so str.splitlines would cut records apart). A last line without its
    newline that does not parse was cut short by a killed append and is skipped, as
    lift.io.existing_keys and the harness's repair do (its pid then counts as missing). Any other
    unreadable line is a ValidateError naming the file and line, never a traceback.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValidateError(f"{path}: not a UTF-8 JSONL file") from None
    complete = text.endswith(_NL)
    lines = text.split(_NL)
    if complete:
        lines.pop()                                      # the empty piece after the final newline
    for lineno, line in enumerate(lines, 1):
        line = line.removesuffix(_CR)
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if lineno == len(lines) and not complete:
                return                                   # cut short by a killed append
            raise ValidateError(f"{path}:{lineno}: not a JSON record") from None
        if not isinstance(row, dict):
            raise ValidateError(f"{path}:{lineno}: not a JSON object")
        yield lineno, row


def fields_of(err: pydantic.ValidationError) -> str:
    """Field names only: pydantic's own message quotes the input, which may be a harmful output."""
    return ", ".join(".".join(str(x) for x in e["loc"]) or "?" for e in err.errors())


class Prompts:
    """PromptRecords by (set, lang) and pid, read on first use through lift.data.build.load_prompts, the
    reader of every stage: a record whose split stamp is not the one the committed split file
    (data/splits) gives its pid, or a missing split file, is a ValidateError naming the remedy, so
    dev/final membership here is the committed split's. A cell short of its frozen split (a translation
    failure awaiting a reader's verdict, which every run refuses) is still read (``allow_short``): a sheet
    may show its prompts meanwhile."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], dict[str, PromptRecord]] = {}

    @staticmethod
    def path(set_name: str, lang: str) -> Path:
        return paths.prompts_dir() / f"{set_name}_{lang}.jsonl"

    def exists(self, set_name: str, lang: str) -> bool:
        return self.path(set_name, lang).is_file()

    def get(self, set_name: str, lang: str) -> dict[str, PromptRecord]:
        key = (set_name, lang)
        if key not in self._cache:
            from lift.data import build  # noqa: PLC0415 (the one reader of the prompt files, not a copy)

            path = self.path(set_name, lang)
            if not path.is_file():
                raise ValidateError(f"{path} is missing: build the prompt sets (lift data build) "
                                    "or fetch them (lift sync pull)")
            try:
                found = build.load_prompts(set_name, lang, allow_short=True)
            except pydantic.ValidationError as e:     # its own message would quote the prompt text
                raise ValidateError(f"{path}: a record is not a PromptRecord ({fields_of(e)})") from None
            except (OSError, ValueError) as e:        # stamps, the split file, an unreadable line
                raise ValidateError(str(e)) from None
            records: dict[str, PromptRecord] = {}
            for rec in found:
                if rec.pid in records:
                    raise ValidateError(f"{path}: pid {rec.pid} appears twice")
                records[rec.pid] = rec
            self._cache[key] = records
        return self._cache[key]


def included(prompt: PromptRecord) -> bool:
    """False for an item a native reader marked unfixable (P01 drops these before generation)."""
    return not prompt.extra.get("excluded")


def dtype_of(meta: dict) -> str | None:
    value = meta.get("dtype")
    if value is None:
        return None
    v = str(value).strip().lower().removeprefix("torch.")
    return _DTYPES.get(v, v)


class LicenceError(ValidateError):
    """Outputs read in a precision other than ``generation.dtype`` that Gate 0's licence does not
    cover (``check_licence``). Callers that take an unreadable generations file for "unknown" never
    take this for one."""


def qwen3guard_role() -> str:
    """The judges role ("primary" or "second") whose block names Qwen3Guard now."""
    for role in ("primary", "second"):
        block = config.get(f"judges.safety_{role}", {})
        if isinstance(block, dict) and "qwen3guard" in str(block.get("id", "")).lower():
            return role
    return "primary"


def licence_problems(dtype: str, gens: Sequence[GenerationRecord], judgements: Sequence[JudgementRecord],
                     cell: tuple[str, str, str, str]) -> list[str]:
    """Why Gate 0's licence does not cover reading ``cell`` in ``dtype``: none for ``generation.dtype``,
    which needs no licence. Otherwise the harness's own rule, as ``lift.judge.pipeline.load_records``
    applies it to every reported number (``_licence_problems``): gate0-{dtype}.json passes on at least 200
    prompts against the bf16 Gate 0 that gate0.json still records, both for the revision of Global that
    made ``gens``; ``gens`` were made under the comparison's harness, decoding settings and GPU class; and
    the ``judgements`` with a stack (Qwen3Guard's) were made on that GPU class. Empty ``gens`` (a file
    that cannot be read here) still need the two reports to pass."""
    if dtype == str(config.get("generation.dtype")):
        return []
    from lift.judge.pipeline import _licence_problems  # noqa: PLC0415 (the harness's rule, not a copy)

    return _licence_problems(dtype, list(gens), list(judgements), cell, qwen3guard_role())


def check_licence(dtype: str, gens: Sequence[GenerationRecord], judgements: Sequence[JudgementRecord],
                  cell: tuple[str, str, str, str]) -> None:
    """Raise LicenceError, naming gate0-{dtype}.json and the remedy, unless Gate 0's licence covers
    reading ``cell`` in ``dtype`` (``licence_problems``)."""
    problems = licence_problems(dtype, gens, judgements, cell)
    if problems:
        variant, set_name, lang, split = cell
        raise LicenceError(
            f"{variant}/{set_name}_{lang}_{split}: outputs in {dtype} are read only while Gate 0's licence "
            f"(gate0-{dtype}.json, decisions.md 27) covers them, and it does not: " + "; ".join(problems)
            + ". Get the licence first (the problems above say how; `lift sync pull` fetches Gate 0's "
            f"reports), or use the {config.get('generation.dtype')} outputs (no --dtype)")


@dataclass
class Pool:
    """The current generations of one variant for one set, language and split."""

    variant: str
    set: str
    lang: str
    split: str
    path: Path
    records: list[GenerationRecord]          # one current record per pid, sorted by pid
    expected: int                            # pids of this split in the prompt file
    missing: list[str] = field(default_factory=list)      # pids with no record at all
    stale_pids: list[str] = field(default_factory=list)   # pids whose records all predate a fix
    stale_records: int = 0
    other_dtype: int = 0
    duplicates: int = 0                      # repeated keys (a resumed run), counted once

    @property
    def name(self) -> str:
        return f"{self.variant}/{self.set}_{self.lang}_{self.split}"

    def problem(self) -> str | None:
        """Why the pool cannot be sampled from, or None."""
        if not self.records and self.other_dtype:
            return (f"{self.name}: its {self.other_dtype} records are in another dtype than the one "
                    "asked for (generation.dtype, or --dtype once Gate 0 has cleared a T4 run)")
        if not self.records and self.stale_records:
            return (f"{self.name}: none of its {self.stale_records} records matches the current prompt "
                    f"text. {STALE_HINT}")
        if self.missing:
            return (f"{self.name}: {len(self.missing)} of {self.expected} prompts have no generation "
                    "yet (an unfinished run)")
        return None

    def counts(self) -> dict[str, int]:
        return {"current": len(self.records), "expected": self.expected,
                "stale_pids": len(self.stale_pids), "missing": len(self.missing),
                "other_dtype": self.other_dtype, "duplicates": self.duplicates}


STALE_HINT = ("A record is stale when its prompt_sha differs from lift.io.sha256_text of the prompt's "
              "text in data/prompts: the translation was fixed after the run, or the harness hashed "
              "something else.")


def generations_path(variant: str, set_name: str, lang: str, split: str) -> Path:
    return paths.generations_dir() / variant / f"{set_name}_{lang}_{split}.jsonl"


def load_pool(variant: str, set_name: str, lang: str, split: str, prompts: Prompts, *,
              prompt_check: bool = True, dtype: str | None = None) -> Pool | None:
    """None when the variant has no generations file for this set, language and split.

    ``prompt_check=False`` is the escape hatch for a harness that hashes something other than the
    prompt text: every record then counts as current, and stale outputs can no longer be told
    apart (callers record and announce it). ``dtype`` is the precision the outputs must have
    (default ``generation.dtype``). Another is read only while Gate 0's licence covers the records
    kept in it: otherwise a LicenceError naming gate0-{dtype}.json (``check_licence``).
    """
    path = generations_path(variant, set_name, lang, split)
    if not path.is_file():
        return None
    texts = prompts.get(set_name, lang)
    expected = {pid for pid, p in texts.items() if p.split == split and included(p)}
    want_dtype = dtype or str(config.get("generation.dtype"))
    by_key: dict[str, GenerationRecord] = {}
    duplicates = other_dtype = 0
    for i, row in read_records(path):
        try:
            rec = GenerationRecord.model_validate(row)
        except pydantic.ValidationError as e:
            raise ValidateError(f"{path}:{i}: not a GenerationRecord ({fields_of(e)})") from None
        if (rec.variant, rec.set, rec.lang, rec.split) != (variant, set_name, lang, split):
            raise ValidateError(f"{path}:{i}: the record belongs to "
                                f"{rec.variant}/{rec.set}_{rec.lang}_{rec.split}")
        if rec.key in by_key:
            duplicates += 1
            continue
        dtype = dtype_of(rec.meta)
        if dtype is not None and dtype != want_dtype:
            other_dtype += 1
            continue
        by_key[rec.key] = rec
    if by_key:
        check_licence(want_dtype, list(by_key.values()), [], (variant, set_name, lang, split))
    current: dict[str, GenerationRecord] = {}
    stale_seen: set[str] = set()
    stale_records = 0
    for rec in by_key.values():
        prompt = texts.get(rec.pid)
        if prompt is None or not included(prompt) or (
                prompt_check and sha256_text(prompt.text) != rec.prompt_sha):
            stale_records += 1
            stale_seen.add(rec.pid)
            continue
        if prompt.split != split:
            raise ValidateError(f"{path}: {rec.pid} is in split {prompt.split}, not {split}")
        if rec.pid in current:
            other = current[rec.pid]
            differ = sorted(k for k in set(other.meta) | set(rec.meta)
                            if other.meta.get(k) != rec.meta.get(k))
            raise ValidateError(f"{path}: {rec.pid} has two current generations (keys {other.key[:12]}... "
                                f"and {rec.key[:12]}...; meta fields that differ: "
                                f"{', '.join(differ) or 'none'}): the file mixes two runs of the same "
                                "prompts. Keep one run's records in the file")
        current[rec.pid] = rec
    stale_pids = sorted((stale_seen & expected) - set(current))
    missing = sorted(expected - set(current) - stale_seen)
    return Pool(variant=variant, set=set_name, lang=lang, split=split, path=path,
                records=sorted(current.values(), key=lambda r: r.pid), expected=len(expected),
                missing=missing, stale_pids=stale_pids, stale_records=stale_records,
                other_dtype=other_dtype, duplicates=duplicates)


@dataclass(frozen=True)
class Target:
    """A generation the validation needs a judgement for. ``output_sha`` is sha256_text of the output
    the judgement must be of: the generation now under ``key`` (or, when its file cannot be read, the
    output the reader was shown). None when neither is known: no judgement can then be tied to it."""

    key: str
    variant: str
    set: str
    lang: str
    split: str
    pid: str
    output_sha: str | None = None


@dataclass
class Judged:
    """Judgements of the targets by generation key, for Qwen3Guard and the frontier judge."""

    qwen3guard_id: str | None
    frontier_id: str | None
    qwen3guard: dict[str, JudgementRecord]
    frontier: dict[str, JudgementRecord]
    # Judgements skipped because they are not of the target's output (``Target.output_sha``), by kind,
    # counted for the chosen judge of each kind (every judge of the kind when none was found usable).
    stale: dict[str, int] = field(default_factory=lambda: {"qwen3guard": 0, "frontier": 0})


def qwen3guard_id() -> str:
    """The Qwen3Guard model id, from whichever judges block names it."""
    for role in ("safety_primary", "safety_second"):
        block = config.get(f"judges.{role}", {})
        if isinstance(block, dict) and "qwen3guard" in str(block.get("id", "")).lower():
            return str(block["id"])
    return QWEN3GUARD


def _matches(judge_id: str, wanted: str) -> bool:
    """A pin names a judge by its whole id, by id@revision without the "#dtype" the harness may
    append, or by the model id alone."""
    return wanted in (judge_id, judge_id.split("#", 1)[0], judge_id.split("@", 1)[0])


def judge_kind(judge_id: str, *, qwen3guard_judge: str | None = None,
               frontier_judge: str | None = None) -> str | None:
    """"qwen3guard", "frontier", or None for a judge left out by the pins."""
    is_qwen = judge_id.split("@", 1)[0].lower() == qwen3guard_id().lower()
    if qwen3guard_judge and _matches(judge_id, qwen3guard_judge):
        return "qwen3guard"
    if is_qwen:
        # Another Qwen3Guard revision: out when one is pinned, never a frontier judge.
        return None if qwen3guard_judge else "qwen3guard"
    if frontier_judge:
        return "frontier" if _matches(judge_id, frontier_judge) else None
    return "frontier"


def load_judgements(targets: Iterable[Target], *, qwen3guard_judge: str | None = None,
                    frontier_judge: str | None = None) -> Judged:
    wanted = {t.key: t for t in targets}
    files = sorted({(t.variant, t.set, t.lang, t.split) for t in wanted.values()})
    found: dict[str, dict[str, dict[str, JudgementRecord]]] = {"qwen3guard": {}, "frontier": {}}
    stale: dict[str, dict[str, set[str]]] = {"qwen3guard": {}, "frontier": {}}
    root = paths.judgements_dir()
    judge_dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    for judge_dir in judge_dirs:
        for variant, set_name, lang, split in files:
            path = judge_dir / variant / f"{set_name}_{lang}_{split}.jsonl"
            if not path.is_file():
                continue
            for i, row in read_records(path):
                if row.get("key") not in wanted:
                    continue
                try:
                    rec = JudgementRecord.model_validate(row)
                except pydantic.ValidationError as e:
                    raise ValidateError(f"{path}:{i}: not a JudgementRecord "
                                        f"({fields_of(e)})") from None
                t = wanted[rec.key]
                if (rec.variant, rec.set, rec.lang, rec.split, rec.pid) != (
                        t.variant, t.set, t.lang, t.split, t.pid):
                    raise ValidateError(f"{path}:{i}: key {rec.key[:12]}... does not "
                                        "match that generation's variant, set, language, split or pid")
                kind = judge_kind(rec.judge, qwen3guard_judge=qwen3guard_judge,
                                  frontier_judge=frontier_judge)
                if kind is None:
                    continue
                if t.output_sha is None or rec.output_sha != t.output_sha:
                    # Of another output than the one under the key now, or unsaid: never paired.
                    stale[kind].setdefault(rec.judge, set()).add(rec.key)
                    continue
                by_key = found[kind].setdefault(rec.judge, {})
                prev = by_key.get(rec.key)
                if prev is not None and (prev.label, prev.judge_label) != (rec.label, rec.judge_label):
                    raise ValidateError(f"{rec.judge} gives two different labels to generation "
                                        f"{rec.key[:12]}... ({path})")
                by_key.setdefault(rec.key, rec)
    chosen: dict[str, tuple[str | None, dict[str, JudgementRecord]]] = {}
    for pin, kind in ((qwen3guard_judge, "qwen3guard"), (frontier_judge, "frontier")):
        if pin and wanted and not found[kind] and not stale[kind]:
            raise ValidateError(f"no {kind} judgement matches {pin!r} for these items")
    for kind, flag in (("qwen3guard", "--qwen3guard-judge"), ("frontier", "--frontier-judge")):
        ids = sorted(found[kind])
        if len(ids) > 1:
            raise ValidateError(f"found several {kind} judges ({', '.join(ids)}); pick one with {flag}")
        chosen[kind] = (ids[0], found[kind][ids[0]]) if ids else (None, {})
    counts = {}
    for kind in ("qwen3guard", "frontier"):
        judge_id = chosen[kind][0]
        skipped = [stale[kind].get(judge_id, set())] if judge_id else list(stale[kind].values())
        counts[kind] = len(set().union(*skipped)) if skipped else 0
    return Judged(qwen3guard_id=chosen["qwen3guard"][0], frontier_id=chosen["frontier"][0],
                  qwen3guard=chosen["qwen3guard"][1], frontier=chosen["frontier"][1], stale=counts)


def final_judgements_exist() -> bool:
    root = paths.judgements_dir()
    return root.is_dir() and any(root.glob("*/*/*_final.jsonl"))
