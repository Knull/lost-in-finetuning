"""What the validation samples from: prompts, generations and judgements. Implemented in P04.

Codes against lift.schemas and the CLAUDE.md file layout, not against the harness code:

- Prompt files are read through lift.data.build.load_prompts, the reader of every stage, so a pid's
  split is the committed split file's (data/splits): a file stamped from another assignment, or a
  missing split file, is refused (``Prompts``: ``PromptFileError``, as is a file that does not parse,
  which the scores never take for a prompt file that is missing here). A cell short of its frozen
  split is still read.
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
  as P02's pipeline and sync refuse to pair it. The judge is told apart by the ``judge`` id in each
  record, not by the directory name or its config role, and only the judges the config names count
  (``judge_rules``): the id must fullmatch the rule lift.judge.pipeline.load_records reports by, for the
  judges block that makes a Qwen3Guard judge (its model, the revision when the config pins a sha, the
  tag of the labelling rules now in force, the precision the outputs were drawn in and the
  Controversial mode) or for the one that makes the frontier judge (its provider, model, rubric version
  and sampling settings, and the rules tag; any API version). Any other judgement (a loose-mode run,
  another revision, labels made under other labelling rules, another rubric or model) is skipped and
  counted by judge id (``other``), so its item waits for its judge. This keeps working after the Nov 6
  decision swaps the primary and second blocks. When several ids fit a kind's rule (two revisions
  while none is pinned, two API versions), one is pinned by id.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pydantic

from lift import config, paths
from lift.io import sha256_text
from lift.schemas import GenerationRecord, JudgementRecord, PromptRecord
from lift.validation.store import ValidateError

SAFETY_SETS = ("harmful", "easy_benign", "hard_benign")
REPAIR_METHODS = ("wiseft", "resta", "cast", "cast_always")

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


class PromptFileError(ValidateError):
    """A prompt file that is here but refused (``Prompts``): a record whose split stamp is not the one the
    committed split gives its pid, or a missing committed split file (lift.data.build.SplitStampError: a
    stale or foreign copy), or a file that does not parse. Callers that go on without a prompt file that is
    missing here (the scores) never take this for one: it stops them."""


class Prompts:
    """PromptRecords by (set, lang) and pid, read on first use through lift.data.build.load_prompts, the
    reader of every stage: a record whose split stamp is not the one the committed split file
    (data/splits) gives its pid, or a missing split file, is a PromptFileError naming the remedy, so
    dev/final membership here is the committed split's; so is a file that does not parse. A prompt file
    that is missing is a plain ValidateError. A cell short of its frozen split (a translation failure
    awaiting a reader's verdict, which every run refuses) is still read (``allow_short``): a sheet may
    show its prompts meanwhile."""

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
                raise PromptFileError(f"{path}: a record is not a PromptRecord ({fields_of(e)})") from None
            except (OSError, ValueError) as e:        # stamps, the split file (SplitStampError), a bad line
                raise PromptFileError(str(e)) from None
            records: dict[str, PromptRecord] = {}
            for rec in found:
                if rec.pid in records:
                    raise PromptFileError(f"{path}: pid {rec.pid} appears twice")
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
    output the reader was shown). None when neither is known: no judgement can then be tied to it.
    ``dtype`` is the precision its outputs were drawn in (the round's draw; None for
    ``generation.dtype``), which a Qwen3Guard judge id names (``judge_rules``)."""

    key: str
    variant: str
    set: str
    lang: str
    split: str
    pid: str
    output_sha: str | None = None
    dtype: str | None = None


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
    # Judgements skipped because no judge the config names made them (``judge_rules``), by judge id: the
    # number of targets each judged.
    other: dict[str, int] = field(default_factory=dict)


Rules = Mapping[str, tuple[re.Pattern[str], str]]


def judge_rules(dtype: str | None = None) -> dict[str, tuple[re.Pattern[str], str]]:
    """{"qwen3guard" | "frontier": (rule, how to name it)}: the ids the judgements of each kind may carry
    to be read for outputs in ``dtype`` (default ``generation.dtype``). Each is the rule
    lift.judge.pipeline.load_records applies to every reported number (``_expected_judge``), for the
    judges role whose config block makes that kind of judge (``judge_dir``, so a swap of the blocks
    changes nothing): the configured model, the revision when the config pins a sha, and the tag of the
    labelling rules now in force (lift.judge.labels.rules_tag); for Qwen3Guard also the precision and the
    Controversial mode, for the frontier judge its rubric version and sampling settings (any API
    version). A config that does not make one judge of each kind is a ValidateError."""
    from lift.judge.pipeline import ROLES, _expected_judge, judge_dir  # noqa: PLC0415 (the harness's rule)

    want = dtype or str(config.get("generation.dtype"))
    rules: dict[str, tuple[re.Pattern[str], str]] = {}
    try:
        for role in ROLES:
            kind = judge_dir(role)
            if kind in rules:
                raise ValueError(f"both judges roles make a {kind} judge")
            rules[kind] = _expected_judge(role, want)
    except (KeyError, OSError, TypeError, ValueError) as e:
        raise ValidateError("configs/project.yaml judges: the validation reads the judgements of the "
                            "configured Qwen3Guard and frontier judges, and cannot tell their ids "
                            f"({e})") from None
    return rules


def _kind(judge_id: str, rules: Rules) -> str | None:
    """The kind whose configured judge made ``judge_id`` (``judge_rules``), None for any other judge."""
    return next((kind for kind, (rule, _) in rules.items() if rule.fullmatch(judge_id)), None)


def _matches(judge_id: str, wanted: str) -> bool:
    """A pin names a judge by its whole id, by the id up to its first "#" (Qwen3Guard: model@revision,
    without the rules tag, precision and mode; a frontier judge: FrontierJudge.judge_id, which ends in its
    API version, without the rules tag), or by the model alone (before the "@": provider/model for a
    frontier judge), which names every revision or API version of it."""
    return wanted in (judge_id, judge_id.split("#", 1)[0], judge_id.split("@", 1)[0])


def judge_kind(judge_id: str, *, dtype: str | None = None, qwen3guard_judge: str | None = None,
               frontier_judge: str | None = None) -> str | None:
    """"qwen3guard" or "frontier" for an id of that kind's configured judge (``judge_rules``, for outputs
    in ``dtype``); None for any other judge, and for one its kind's pin leaves out (never relabelled as
    the other kind)."""
    kind = _kind(judge_id, judge_rules(dtype))
    pin = {"qwen3guard": qwen3guard_judge, "frontier": frontier_judge}.get(kind or "")
    return None if kind is None or (pin and not _matches(judge_id, pin)) else kind


def load_judgements(targets: Iterable[Target], *, qwen3guard_judge: str | None = None,
                    frontier_judge: str | None = None) -> Judged:
    """The judgements of ``targets`` by the Qwen3Guard and frontier judges the config names
    (``judge_rules``, for each target's ``dtype``), of the output each target holds. A judgement of any
    other judge is skipped and counted by its id (``Judged.other``), one of another output by kind
    (``Judged.stale``). A pin chooses among the ids that fit its kind's rule (``_matches``); several ids
    of one kind without a pin, or a pin that matches none, is a ValidateError."""
    wanted = {t.key: t for t in targets}
    files = sorted({(t.variant, t.set, t.lang, t.split) for t in wanted.values()})
    gen_dtype = str(config.get("generation.dtype"))
    rules = {d: judge_rules(d) for d in sorted({t.dtype or gen_dtype for t in wanted.values()})}
    pins = {"qwen3guard": qwen3guard_judge, "frontier": frontier_judge}
    found: dict[str, dict[str, dict[str, JudgementRecord]]] = {"qwen3guard": {}, "frontier": {}}
    stale: dict[str, dict[str, set[str]]] = {"qwen3guard": {}, "frontier": {}}
    other: dict[str, set[str]] = {}
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
                kind = _kind(rec.judge, rules[t.dtype or gen_dtype])
                if kind is None:
                    # No configured judge made it (another mode, revision, labelling rules, rubric or
                    # model): never scored, so its item waits for its judge.
                    other.setdefault(rec.judge, set()).add(rec.key)
                    continue
                pin = pins[kind]
                if pin and not _matches(rec.judge, pin):
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
    for kind, pin in pins.items():
        if pin and wanted and not found[kind] and not stale[kind]:
            shown = ", ".join(sorted({r[kind][1] for r in rules.values()}))
            raise ValidateError(f"no {kind} judgement matches {pin!r} for these items: a pin chooses among "
                                f"the judgements of the configured {kind} judge ({shown})")
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
                  qwen3guard=chosen["qwen3guard"][1], frontier=chosen["frontier"][1], stale=counts,
                  other={judge_id: len(keys) for judge_id, keys in sorted(other.items())})


def final_judgements_exist() -> bool:
    root = paths.judgements_dir()
    return root.is_dir() and any(root.glob("*/*/*_final.jsonl"))
