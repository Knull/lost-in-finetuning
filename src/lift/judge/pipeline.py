"""Judge a generations file into a judgements file, and read judgements back for reporting.
Implemented in P02.

For each selected GenerationRecord: GlotLID top-3 on the output (an empty or whitespace-only output
gets ``[("und", 0.0)]`` without a LID call) -> unclear flags -> judge, in chunks -> final label
(lift.judge.labels) -> one JudgementRecord appended per item to
outputs/judgements/{judge_dir}/{variant}/{set}_{lang}_{split}.jsonl after each chunk.

- Exactly one judgement per selected generation, naming the output it judged (``output_sha``: the
  generation key identifies the inputs, and a regenerated output under the same key can differ). A
  rerun skips the keys already judged (a truncated last line is repaired first). A judgements file
  holding another judge's records, duplicate keys, judgements of generations no longer in the
  generations file, or of another output than the one now under their key, raises: judges and
  stale outputs are never mixed.
- Every prompt stays in the denominator: a judge failure is written as label "unclear" with
  ``native.judge_failure`` set, and counted in the run manifest.
- Selection: every generation (the primary judge); ``keys=`` exactly the listed keys present in the
  file (``lift judge --keys``); ``sample=f`` P04's draw by pid (``second_judge_pids``) plus every
  key of the validation rounds. P04's validation/keys/second_judge/{variant}_{split}.txt lists are
  the canonical second-judge selection (``lift judge --judge second --keys``); ``--sample``
  reproduces their random part on a complete generations file: the same prompts in every variant
  and language, so the second judge's view stays paired by pid.
- The judgements directory comes from the kind of the role's config entry: an entry with a
  ``provider`` is the frontier LLM judge ("frontier"), a Qwen3Guard id is "qwen3guard". A Qwen3Guard
  judge id ends in ``#{dtype}#{strict|loose}``, so two precisions or two Controversial mappings
  never share a file.

Reading:
- ``load_records`` (whole JudgementRecords) and ``load_labels`` ({pid: label}) are the door to every
  reported number, the loose label, flags and judge failures included. Each returned judgement must
  be of a generation now in the generations file, of its current output, made from the current
  prompt text; every generation of the cell in ``dtype`` (default ``generation.dtype``: one
  precision for every reported number) under one run configuration; and the judge must be the
  role's configured judge (model, revision when pinned to a sha, and for Qwen3Guard the precision
  and Controversial mode). ``complete=True`` (every reported number) also needs a generation for
  every prompt of the split and a judgement of every generation; ``complete=False`` is for the
  second judge's sample and runs in progress.
- ``provenance`` describes a cell (judge id; generation dtype, run configuration, model revision,
  harness, library versions and GPU). ``paired_labels`` reads two variants' cells and refuses a
  pair whose judge, precision, model revision or harness differ. Every paired comparison downstream
  (Gate 2, P07) must use ``paired_labels``, never two ``load_labels`` calls.
- ``load_judgements`` reads one file with only its own checks: never report from it.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from lift import config, prompts
from lift.io import append_jsonl, read_jsonl, sha256_text
from lift.judge import labels
from lift.paths import REPO_ROOT, judgements_dir, validation_dir
from lift.schemas import GenerationRecord, JudgementRecord, UnclearFlags

log = logging.getLogger(__name__)

ROLES: tuple[str, ...] = ("primary", "second")
_ROLE_ENTRIES = {"primary": "safety_primary", "second": "safety_second"}     # under judges: in project.yaml
CONTROVERSIAL_MODES = ("strict", "loose")
EMPTY_OUTPUT_LID: tuple[tuple[str, float], ...] = (("und", 0.0),)
QWEN3GUARD_SAFETY = ("Safe", "Unsafe", "Controversial")
PROVENANCE_META = ("dtype", "run_config", "model_rev", "harness", "torch", "transformers", "gpu")
PAIR_MUST_MATCH = ("judge_id", "dtype", "model_rev", "harness")
PAIR_MAY_DIFFER = ("gpu", "torch", "transformers")
_KEY = re.compile(r"[0-9a-f]{64}")
_SHA = re.compile(r"[0-9a-f]{40}")
_DIR_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class JudgeItem:
    key: str             # the GenerationRecord key
    prompt: str          # the native prompt text (not the chat-formatted text)
    response: str
    prompt_lang: str


@dataclass(frozen=True)
class JudgeOutcome:
    """The judge's call mapped to our four classes. A failure is ``judge_label="unclear"`` with
    ``native["judge_failure"]`` set: still written and counted, never dropped."""

    judge_label: str
    native: dict[str, Any]
    raw: str


class Judge(Protocol):
    """What ``judge_file`` needs. ``judge_id`` (revision included) goes into every record, so two
    judges never share a file; ``short`` names the judgements directory."""

    judge_id: str
    short: str

    def judge(self, items: Sequence[JudgeItem]) -> list[JudgeOutcome]: ...


# ----------------------------------------------------------------------------- roles and judges


def _where(role: str) -> str:
    return f"judges.{_ROLE_ENTRIES[role]}"


def _judge_entry(role: str) -> dict[str, Any]:
    if role not in _ROLE_ENTRIES:
        raise ValueError(f"judge role must be one of {ROLES}, got {role!r}")
    entry = config.get(_where(role))
    if not isinstance(entry, dict):
        raise ValueError(f"configs/project.yaml {_where(role)} must be a mapping")
    return entry


def _entry_kind(role: str, entry: Mapping[str, Any]) -> str:
    frontier = "provider" in entry
    qwen = "Qwen3Guard" in str(entry.get("id", ""))
    if frontier == qwen:
        raise ValueError(f"configs/project.yaml {_where(role)}: cannot tell the judge kind "
                         "(expected a frontier entry with a provider, or a Qwen3Guard id)")
    return "frontier" if frontier else "qwen3guard"


def _controversial_mode(role: str, entry: Mapping[str, Any]) -> str:
    mode = entry.get("controversial")
    if mode not in CONTROVERSIAL_MODES:
        raise ValueError(f"{_where(role)}.controversial must be 'strict' or 'loose', got {mode!r}")
    return str(mode)


def judge_dir(role: str) -> str:
    """The judgements directory of ``role`` ("primary" | "second"), from its config entry's kind."""
    return _entry_kind(role, _judge_entry(role))


class _JudgeDirs(Mapping[str, str]):
    """``JUDGE_DIRS[role]``, read from the config on each access so it never goes stale."""

    def __getitem__(self, role: str) -> str:
        if role not in _ROLE_ENTRIES:
            raise KeyError(role)
        return judge_dir(role)

    def __iter__(self) -> Iterator[str]:
        return iter(ROLES)

    def __len__(self) -> int:
        return len(ROLES)


JUDGE_DIRS: Mapping[str, str] = _JudgeDirs()


def _check_count(results: Sequence[Any], items: Sequence[JudgeItem], who: str) -> None:
    if len(results) != len(items):
        raise RuntimeError(f"{who} returned {len(results)} results for {len(items)} items")


class Qwen3GuardAdapter:
    """Qwen3Guard-Gen response moderation as a Judge.

    ``judge_label`` follows the configured Controversial mode (strict: harmful). ``native`` keeps
    Safety, Categories and Refusal, the guard's ``consistent`` (False for Unsafe with no category or
    Safe with one) and ``judge_label_loose``, so the loose sensitivity analysis needs no re-judging.
    The judge id ends in ``#{dtype}#{mode}``: labels from two precisions or two Controversial modes
    never share a judgements file, and resume and load refuse a mix.
    """

    short = "qwen3guard"

    def __init__(self, guard: Any, *, strict: bool, device: str | None, dtype: str) -> None:
        self.guard = guard
        self.strict = strict
        self.mode = "strict" if strict else "loose"
        self.device = device
        self.dtype = dtype
        self.judge_id = f"{guard.judge_id}#{dtype}#{self.mode}"

    @property
    def stats(self) -> dict[str, Any] | None:
        stats = getattr(self.guard, "stats", None)
        return dict(stats) if stats is not None else None

    @property
    def info(self) -> dict[str, Any] | None:
        info = getattr(self.guard, "info", None)
        return dict(info) if info is not None else None

    def judge(self, items: Sequence[JudgeItem]) -> list[JudgeOutcome]:
        results = self.guard.judge([i.prompt for i in items], [i.response for i in items])
        _check_count(results, items, "Qwen3Guard")
        return [self._outcome(item, r) for item, r in zip(items, results, strict=True)]

    def _outcome(self, item: JudgeItem, r: Mapping[str, Any]) -> JudgeOutcome:
        raw = r.get("raw") or ""
        if r.get("error"):
            failure = {"judge_failure": str(r["error"]), "safety": None, "categories": [], "refusal": None,
                       "consistent": None, "judge_label_loose": "unclear"}
            return JudgeOutcome("unclear", failure, raw)
        safety, refusal, categories = r.get("safety"), r.get("refusal"), r.get("categories")
        if (safety not in QWEN3GUARD_SAFETY or refusal not in ("Yes", "No")
                or not isinstance(categories, (list, tuple))
                or not all(isinstance(c, str) for c in categories)):
            raise ValueError(f"Qwen3Guard result for key {item.key[:12]} has no error but is not "
                             "{safety: Safe|Unsafe|Controversial, categories: [str], refusal: Yes|No} "
                             "(the lift.judge.qwen3guard contract)")
        refused = refusal == "Yes"
        consistent = r.get("consistent")
        if not isinstance(consistent, bool):          # the parser's rule, for a guard that omits it
            named = len(categories) > 0
            consistent = not ((safety == "Unsafe" and not named) or (safety == "Safe" and named))
        native = {
            "safety": safety,
            "categories": list(categories),
            "refusal": refusal,
            "consistent": consistent,
            "judge_label_loose": labels.map_qwen3guard(safety, refused, strict=False),
        }
        return JudgeOutcome(labels.map_qwen3guard(safety, refused, strict=self.strict), native, raw)


class FrontierAdapter:
    """The frontier LLM judge (``lift.judge.frontier.FrontierJudge``) as a Judge. ``native`` keeps the
    judge's label and reason, and ``model_echo`` (the model id that answered) when the result has
    one. ``stats`` passes through its call, token and cost counters for the run manifest."""

    short = "frontier"
    device = None
    dtype = None

    def __init__(self, frontier: Any) -> None:
        self.frontier = frontier
        self.judge_id = frontier.judge_id

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self.frontier.stats)

    def judge(self, items: Sequence[JudgeItem]) -> list[JudgeOutcome]:
        results = self.frontier.safety([i.prompt for i in items], [i.response for i in items],
                                       [i.prompt_lang for i in items])
        _check_count(results, items, "the frontier judge")
        out = []
        for item, r in zip(items, results, strict=True):
            raw = r.get("raw") or ""
            echo = {"model_echo": r["model_echo"]} if r.get("model_echo") else {}
            if r.get("error"):
                out.append(JudgeOutcome("unclear", {"judge_failure": str(r["error"]), "label": None,
                                                    "reason": r.get("reason"), **echo}, raw))
                continue
            label = r.get("label")
            if not isinstance(label, str):
                raise ValueError(f"frontier result for key {item.key[:12]} has neither a label nor an error")
            out.append(JudgeOutcome(labels.map_frontier(label),
                                    {"label": label, "reason": r.get("reason"), **echo}, raw))
        return out


def default_judge_device() -> str:
    """cuda:1 when there are two GPUs (Kaggle T4 x2: the generator keeps cuda:0), else cuda:0."""
    import torch

    return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"


def make_judge(role: str, *, device: str | None = None, dtype: str | None = None) -> Judge:
    """The judge of ``role`` built from configs/project.yaml, by the kind of its entry.

    Qwen3Guard: revision resolved to a commit sha, ``device`` defaults to ``default_judge_device()``
    and ``dtype`` to ``generation.dtype``. Frontier: provider, model and the rubric file from the
    entry, and every other setting (max_tokens, effort, price, workers, cache path) from the same
    entry, so the role's own judge runs also after the decision rule swaps the primary and second
    entries (``device``/``dtype`` do not apply). Judge modules are imported here, not at import time.
    """
    entry = _judge_entry(role)
    where = _where(role)
    if _entry_kind(role, entry) == "qwen3guard":
        from lift.judge.qwen3guard import Qwen3GuardJudge
        from lift.models.load import resolved_revision

        mode = _controversial_mode(role, entry)
        dtype = dtype or config.get("generation.dtype")
        device = device or default_judge_device()
        revision = resolved_revision(entry["id"], entry["revision"])
        guard = Qwen3GuardJudge(entry["id"], revision, device=device, dtype=dtype)
        return Qwen3GuardAdapter(guard, strict=mode == "strict", device=device, dtype=dtype)

    provider, model = entry.get("provider"), entry.get("model")
    if not provider or not model or "TBD" in (provider, model):
        raise ValueError(f"{where}.provider and .model must be set in configs/project.yaml "
                         f"(got {provider!r}, {model!r})")
    rubric = Path(entry["rubric"])
    rubric = rubric if rubric.is_absolute() else REPO_ROOT / rubric
    if not rubric.is_file():
        raise FileNotFoundError(f"{where}.rubric: {rubric} not found")
    from lift.judge.frontier import FrontierJudge

    return FrontierAdapter(FrontierJudge(provider, model, str(rubric), config_entry=_ROLE_ENTRIES[role]))


# ----------------------------------------------------------------------------- paths and keys


def judgements_path(judge_dir: str, variant: str, set_name: str, lang: str, split: str) -> Path:
    return judgements_dir() / judge_dir / variant / f"{set_name}_{lang}_{split}.jsonl"


def read_keys_file(path: str | Path) -> list[str]:
    """Generation keys from a text file: one 64-hex key per line, blank lines and ``#`` comments
    skipped (a UTF-8 BOM, as PowerShell writes it, is tolerated). A malformed line raises."""
    keys = []
    with open(path, encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if not _KEY.fullmatch(s):
                raise ValueError(f"{path}:{lineno}: not a 64-hex generation key")
            keys.append(s)
    return keys


def _round_key(path: Path, item: str, rec: Any) -> str:
    if not isinstance(rec, dict):
        raise ValueError(f"{path}: item {item} is not a record")
    found = [rec[f] for f in ("key", "generation_key") if f in rec]
    if not found:
        raise ValueError(f"{path}: item {item} has neither 'key' nor 'generation_key'")
    if len(found) == 2 and found[0] != found[1]:
        raise ValueError(f"{path}: item {item} has a 'key' and a 'generation_key' that differ")
    key = found[0]
    if not isinstance(key, str) or not _KEY.fullmatch(key):
        raise ValueError(f"{path}: item {item} has a malformed generation key")
    return key


def validation_keys() -> set[str]:
    """Every generation key named in validation/keys/round*.json (written by P04): a mapping
    item_id -> record or a list of records, each with a ``key`` or ``generation_key``."""
    keys: set[str] = set()
    for path in sorted((validation_dir() / "keys").glob("round*.json")):
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(doc, dict):
            items = [(str(k), v) for k, v in doc.items()]
        elif isinstance(doc, list):
            items = [(str(i), v) for i, v in enumerate(doc)]
        else:
            raise ValueError(f"{path}: expected a mapping item_id -> record or a list of records")
        keys.update(_round_key(path, item, rec) for item, rec in items)
    return keys


def _rank(seed: int, *parts: object) -> str:
    """P04's seeded sort key (lift.validation.sample.rank): the same on every platform."""
    return sha256_text(":".join(str(p) for p in (seed, *parts)))


def second_judge_pids(set_name: str, split: str, fraction: float, seed: int) -> set[str]:
    """P04's random second-judge sample of one set and split (lift.validation.second_judge.chosen_pids,
    exactly): of the pids of that split in every language with a prompt file (a record marked
    ``extra.excluded`` left out), the first ceil(fraction x n) by ``_rank(seed, "second", set,
    split, pid)``. Drawn by pid, never by generation key (which differs per variant and language), so
    the same prompts are drawn in every variant and language, whatever the file order."""
    population: set[str] = set()
    for lang in config.get("languages.all"):
        try:
            records = prompts.load(set_name, lang, split)
        except FileNotFoundError:
            continue
        population.update(p.pid for p in records if not p.extra.get("excluded"))
    k = math.ceil(fraction * len(population))
    return set(sorted(population, key=lambda pid: _rank(seed, "second", set_name, split, pid))[:k])


def _select(gens: Sequence[GenerationRecord], *, sample: float | None, keys: Collection[str] | None,
            seed: int | None, set_name: str, split: str) -> tuple[list[GenerationRecord], dict[str, Any]]:
    if sample is not None and keys is not None:
        raise ValueError("pass sample or keys, not both")
    if seed is not None and sample is None:
        raise ValueError("seed applies only with sample")
    if keys is not None:
        if isinstance(keys, str):
            raise TypeError("keys must be a collection of generation keys, not one string")
        wanted = set(keys)
        present = {g.key for g in gens}
        return ([g for g in gens if g.key in wanted],
                {"mode": "keys", "n_keys_given": len(wanted), "n_keys_not_in_file": len(wanted - present)})
    if sample is not None:
        if isinstance(sample, bool) or not isinstance(sample, (int, float)) or not 0 < sample <= 1:
            raise ValueError(f"sample must be a fraction in (0, 1], got {sample!r}")
        seed = int(config.get("split_seed")) if seed is None else int(seed)
        drawn = second_judge_pids(set_name, split, float(sample), seed)
        extra = validation_keys()
        chosen = [g for g in gens if g.pid in drawn or g.key in extra]
        return chosen, {"mode": "sample", "fraction": float(sample), "seed": seed, "n_pids_drawn": len(drawn),
                        "n_validation_keys_in_file": sum(g.key in extra for g in gens)}
    return list(gens), {"mode": "all"}


# ----------------------------------------------------------------------------- reading records


def _read_judgements(path: Path) -> list[JudgementRecord]:
    """All records of one judgements file. Two records for one key or one pid raise."""
    records = [JudgementRecord.model_validate(row) for row in read_jsonl(path)]
    seen_keys: set[str] = set()
    seen_pids: set[str] = set()
    dupes = []
    for r in records:
        if r.key in seen_keys or r.pid in seen_pids:
            dupes.append(r.pid)
        seen_keys.add(r.key)
        seen_pids.add(r.pid)
    if dupes:
        raise ValueError(f"{path}: {len(dupes)} duplicate judgements (same key or pid), "
                         f"e.g. pids {dupes[:5]}")
    return records


def _one_cell(gens: Sequence[GenerationRecord], path: Path) -> tuple[str, str, str, str]:
    cells = sorted({(g.variant, g.set, g.lang, g.split) for g in gens})
    if len(cells) != 1:
        raise ValueError(f"{path}: expected the generations of one variant/set/lang/split, "
                         f"found {len(cells)} {cells[:4]}")
    return cells[0]


def _check_prompt_texts(gens: Sequence[GenerationRecord], texts: Mapping[str, str], path: Path) -> None:
    missing = sorted(g.pid for g in gens if g.pid not in texts)
    if missing:
        raise ValueError(f"{path}: no prompt text for {len(missing)} pids, e.g. {missing[:5]}")
    changed = sorted(g.pid for g in gens if g.prompt_sha != sha256_text(texts[g.pid]))
    if changed:
        raise ValueError(f"{path}: {len(changed)} generations were made from another prompt text (the prompt "
                         f"changed after generation), e.g. {changed[:5]}; regenerate them")


def _resume_state(out_path: Path, judge_id: str, gens: Sequence[GenerationRecord],
                  cell: tuple[str, str, str, str], *, resume: bool) -> dict[str, JudgementRecord]:
    """{key: record} already in ``out_path``, after checking it holds this judge's judgements of
    these generations' current outputs only (a record without ``output_sha`` is not trusted)."""
    from lift.models.generate import repair_tail

    if not out_path.exists():
        return {}
    if not resume:
        raise FileExistsError(f"{out_path} exists; pass resume=True or move it aside")
    repair_tail(out_path)
    records = _read_judgements(out_path)
    others = sorted({r.judge for r in records} - {judge_id})
    if others:
        raise ValueError(f"{out_path} holds judgements by {others}, not {judge_id}; judges are never mixed "
                         "in one file")
    gen_of = {g.key: g for g in gens}
    stale = sorted(r.pid for r in records if (r.variant, r.set, r.lang, r.split) != cell
                   or r.key not in gen_of or gen_of[r.key].pid != r.pid)
    if stale:
        raise ValueError(f"{out_path}: {len(stale)} judgements are of generations that are not in the "
                         f"generations file (regenerated?), e.g. pids {stale[:5]}; move the file aside")
    changed = sorted(r.pid for r in records if r.output_sha != sha256_text(gen_of[r.key].output))
    if changed:
        raise ValueError(f"{out_path}: {len(changed)} judgements are of another output than the generation "
                         f"now under their key, e.g. pids {changed[:5]}: generations changed since judging; "
                         "move the judgements file aside")
    return {r.key: r for r in records}


# ----------------------------------------------------------------------------- judging


def _flags(g: GenerationRecord, lid: Any, accept: Mapping[str, Sequence[str]],
           unclear: Mapping[str, Any]) -> UnclearFlags:
    lid_top = lid.top(g.output, k=3) if g.output.strip() else list(EMPTY_OUTPUT_LID)
    return labels.unclear_flags(
        text=g.output, n_new_tokens=g.n_new_tokens, lid_top=lid_top, prompt_lang=g.lang,
        accept_labels=accept, min_new_tokens=int(unclear["min_new_tokens"]),
        max_repetition_ratio=float(unclear["max_repetition_ratio"]),
        min_lid_prob=float(unclear["min_lid_prob"]),
    )


def _judgement(g: GenerationRecord, flags: UnclearFlags, outcome: JudgeOutcome,
               judge_id: str) -> JudgementRecord:
    native = dict(outcome.native)
    if native.get("judge_failure") and outcome.judge_label != "unclear":
        raise ValueError(f"key {g.key[:12]}: a judge failure must carry judge_label 'unclear'")
    label = labels.final_label(outcome.judge_label, flags)
    if "judge_label_loose" in native:
        native["label_loose"] = labels.final_label(native["judge_label_loose"], flags)
    return JudgementRecord(key=g.key, variant=g.variant, pid=g.pid, lang=g.lang, set=g.set, split=g.split,
                           judge=judge_id, judge_label=outcome.judge_label, label=label, flags=flags,
                           native=native, raw=outcome.raw, output_sha=sha256_text(g.output))


def _failed(r: JudgementRecord) -> bool:
    return bool(r.native.get("judge_failure"))


def _stats_delta(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """What this run added to a judge's cumulative counters (one judge object may judge many files)."""
    if after is None:
        return None
    out: dict[str, Any] = {}
    for k, v in after.items():
        b = (before or {}).get(k, 0)
        numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (v, b))
        out[k] = (round(v - b, 6) if isinstance(v, float) else v - b) if numeric else v
    return out


def judge_file(generations_path: str | Path, *, judge: str | Judge = "primary", resume: bool = True,
               lid: Any = None, sample: float | None = None, keys: Collection[str] | None = None,
               out_path: str | Path | None = None, device: str | None = None, dtype: str | None = None,
               prompts_by_pid: Mapping[str, str] | None = None, seed: int | None = None,
               batch_size: int = 64) -> Path:
    """Judge one generations file (one variant/set/lang/split) and return the judgements path.

    ``judge`` is a role ("primary" | "second", built with ``make_judge(role, device=, dtype=)``) or a
    Judge object. ``lid`` has ``top(text, k)`` (default ``lift.lang.LangID()``, built only if a
    non-empty output needs it). Prompt texts come from ``prompts_by_pid`` or the prompt files, and
    every generation's ``prompt_sha`` must match them. ``out_path`` defaults to
    outputs/judgements/{judge.short}/{variant}/{set}_{lang}_{split}.jsonl. The manifest records the
    counts (``selection.n_keys_not_in_file`` for ``keys=``, ``selection.n_pids_drawn`` for
    ``sample=``) and the judge's cost counters, also when the run fails. The judgements file exists
    only once something has been judged.
    """
    from lift.manifest import finish_run, start_run
    from lift.models.generate import read_generations

    generations_path = Path(generations_path)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    gens = read_generations(generations_path)
    cell = _one_cell(gens, generations_path)
    variant, set_name, lang, split = cell
    texts = dict(prompts_by_pid) if prompts_by_pid is not None else prompts.texts_by_pid(set_name, lang)
    _check_prompt_texts(gens, texts, generations_path)
    selected, selection = _select(gens, sample=sample, keys=keys, seed=seed, set_name=set_name, split=split)

    if isinstance(judge, str):
        judge = make_judge(judge, device=device, dtype=dtype)
    elif device is not None or dtype is not None:
        raise ValueError("device and dtype apply only when judge_file builds the judge from a role")
    if not _DIR_NAME.fullmatch(str(judge.short)):
        raise ValueError(f"judge.short must be a plain directory name, got {judge.short!r}")
    out_path = (Path(out_path) if out_path is not None
                else judgements_path(judge.short, variant, set_name, lang, split))
    existing = _resume_state(out_path, judge.judge_id, gens, cell, resume=resume)
    todo = sorted((g for g in selected if g.key not in existing), key=lambda g: g.pid)
    n_resumed = len(selected) - len(todo)

    judge_device, judge_dtype = getattr(judge, "device", None), getattr(judge, "dtype", None)
    stats_before = getattr(judge, "stats", None)
    stats_before = dict(stats_before) if stats_before is not None else None
    run = start_run("judge", variant=variant, set=set_name, lang=lang, split=split, judge=judge.judge_id,
                    judge_dir=judge.short, device=judge_device, dtype=judge_dtype,
                    generations_path=str(generations_path), out_path=str(out_path), selection=selection,
                    n_generations=len(gens), n_selected=len(selected), n_resumed=n_resumed,
                    batch_size=batch_size, devices=0 if judge_device is None else 1,
                    judge_info=getattr(judge, "info", None))
    log.info("%s/%s: %d to judge (%d of %d selected already judged; %d generations) with %s",
             variant, out_path.name, len(todo), n_resumed, len(selected), len(gens), judge.judge_id)
    if selection.get("n_keys_not_in_file"):
        log.info("%s/%s: %d of the given keys are not in this file", variant, out_path.name,
                 selection["n_keys_not_in_file"])

    new: list[JudgementRecord] = []
    timing = {"seconds": 0.0, "lid_seconds": 0.0}

    def summary(status: str, **extra: Any) -> dict[str, Any]:
        chosen = {g.key for g in selected}
        final = [r for k, r in existing.items() if k in chosen] + new
        loose = Counter(r.native["label_loose"] for r in final if "label_loose" in r.native)
        return {"status": status, "n_judged": len(new), "n_failures": sum(map(_failed, new)),
                "n_failures_selected": sum(map(_failed, final)),
                "label_counts": dict(sorted(Counter(r.label for r in final).items())),
                **({"label_loose_counts": dict(sorted(loose.items()))} if loose else {}),
                "seconds": round(timing["seconds"], 3), "lid_seconds": round(timing["lid_seconds"], 3),
                "judge_stats": _stats_delta(stats_before, getattr(judge, "stats", None)), **extra}

    accept = config.get("languages.accept_labels")
    unclear = config.get("unclear")
    try:
        for start in range(0, len(todo), batch_size):
            chunk = todo[start: start + batch_size]
            if lid is None and any(g.output.strip() for g in chunk):
                from lift.lang import LangID

                lid = LangID()
            t0 = time.perf_counter()
            flags = [_flags(g, lid, accept, unclear) for g in chunk]
            t1 = time.perf_counter()
            outcomes = judge.judge([JudgeItem(g.key, texts[g.pid], g.output, g.lang) for g in chunk])
            t2 = time.perf_counter()
            _check_count(outcomes, chunk, f"judge {judge.judge_id}")
            records = [_judgement(g, f, o, judge.judge_id)
                       for g, f, o in zip(chunk, flags, outcomes, strict=True)]
            append_jsonl(out_path, [r.model_dump() for r in records])
            new.extend(records)
            timing["lid_seconds"] += t1 - t0
            timing["seconds"] += t2 - t1
            log.info("%s/%s: %d/%d judged", variant, out_path.name, len(new), len(todo))
    except BaseException as exc:
        finish_run(run, **summary("failed", error=type(exc).__name__))
        raise
    done = summary("ok")
    if done["n_failures"]:
        log.warning("%s/%s: %d judge failures recorded as unclear", variant, out_path.name,
                    done["n_failures"])
    log.info("%s/%s: wrote %d judgements; labels %s", variant, out_path.name, len(new), done["label_counts"])
    finish_run(run, **done)
    return out_path


# ----------------------------------------------------------------------------- reading for reports


def load_judgements(variant: str, set_name: str, lang: str, split: str, *,
                    judge: str = "primary") -> list[JudgementRecord]:
    """The judgements file of one variant/set/lang/split by the ``judge`` role's judge, sorted by pid,
    with only the file's own checks (duplicate keys or pids, records of another cell, a file mixing
    judges). Nothing reported comes from here: ``load_records`` checks it against the generations."""
    path = judgements_path(judge_dir(judge), variant, set_name, lang, split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `lift judge` (or `lift sync pull`)")
    records = _read_judgements(path)
    ids = sorted({r.judge for r in records})
    if len(ids) > 1:
        raise ValueError(f"{path} mixes judges {ids}")
    cell = (variant, set_name, lang, split)
    wrong = sorted(r.pid for r in records if (r.variant, r.set, r.lang, r.split) != cell)
    if wrong:
        raise ValueError(f"{path}: {len(wrong)} records belong to another variant/set/lang/split, "
                         f"e.g. {wrong[:5]}")
    return sorted(records, key=lambda r: r.pid)


def _expected_judge(role: str, dtype: str) -> tuple[re.Pattern[str], str]:
    """The judge ids a reported file of ``role`` may hold, and how to name them: the configured model
    (and revision, when the config pins a sha); for Qwen3Guard also ``#{dtype}#{mode}``."""
    entry = _judge_entry(role)
    if _entry_kind(role, entry) == "frontier":
        prefix = f"{entry.get('provider')}/{entry.get('model')}@"
        return re.compile(re.escape(prefix) + ".+"), prefix + "..."
    revision = str(entry.get("revision", ""))
    pinned = bool(_SHA.fullmatch(revision))
    tail = f"#{dtype}#{_controversial_mode(role, entry)}"
    pattern = re.escape(f"{entry['id']}@") + (re.escape(revision) if pinned else "[^@#]+") + re.escape(tail)
    return re.compile(pattern), f"{entry['id']}@{revision if pinned else '<sha>'}{tail}"


def _check_cell(records: Sequence[JudgementRecord], gens: Sequence[GenerationRecord], gpath: Path,
                cell: tuple[str, str, str, str], *, role: str, complete: bool, dtype: str) -> None:
    """Raise unless ``records`` can be reported (the rules are in the module docstring)."""
    variant, set_name, lang, split = cell
    texts = {p.pid: p.text for p in prompts.load(set_name, lang, split)}
    gen_of = {g.key: g for g in gens}
    current = [r for r in records if r.key in gen_of and gen_of[r.key].pid == r.pid]
    current_keys = {r.key for r in current}
    prompted = gens if complete else [gen_of[r.key] for r in current]   # whose prompt must be current
    checks = [
        ([g.pid for g in gens if (g.variant, g.set, g.lang, g.split) != cell],
         "generations belong to another variant/set/lang/split"),
        ([r.pid for r in records if r.key not in current_keys],
         f"judgements are of generations no longer in {gpath.name}"),
        ([r.pid for r in current if r.output_sha != sha256_text(gen_of[r.key].output)],
         "judgements are of another output than the generation now under their key (generations changed "
         "since judging; move the judgements file aside)"),
        ([g.pid for g in prompted if g.pid not in texts],
         f"generations are of pids that are not {split} prompts"),
        ([g.pid for g in prompted if g.pid in texts and g.prompt_sha != sha256_text(texts[g.pid])],
         "generations were made from another prompt text"),
    ]
    if complete:
        judged = {r.key for r in records}
        checks += [
            ([g.pid for g in gens if g.key not in judged], f"of {len(gens)} generations have no judgement"),
            (list(set(texts) - {g.pid for g in gens}), f"of {len(texts)} prompts have no generation"),
        ]
    problems = [f"{len(pids)} {what}, e.g. {sorted(pids)[:5]}" for pids, what in checks if pids]
    off = sorted(g.pid for g in gens if g.meta.get("dtype") != dtype)
    if off:
        dtypes = sorted({str(g.meta.get("dtype")) for g in gens})
        problems.append(f"{len(off)} generations are not in {dtype} (the cell has {dtypes}), e.g. {off[:5]}; "
                        "every reported number uses one precision, another only when named with dtype= "
                        "after Gate 0 licensed it")
    run_configs = sorted({str(g.meta.get("run_config")) for g in gens})
    if len(run_configs) > 1:
        problems.append(f"generations mix {len(run_configs)} run configurations {run_configs[:4]}")
    rule, shown = _expected_judge(role, dtype)
    others = sorted({r.judge for r in records if not rule.fullmatch(r.judge)})
    if others:
        problems.append(f"judged by {others}, not the configured {role} judge {shown}")
    if problems:
        raise ValueError(f"{variant}/{set_name}_{lang}_{split} cannot be reported "
                         f"({'complete' if complete else 'partial'} read): " + "; ".join(problems))


def _read_cell(variant: str, set_name: str, lang: str, split: str, *, judge: str, complete: bool,
               dtype: str | None) -> tuple[list[JudgementRecord], list[GenerationRecord]]:
    """One cell's judgements (sorted by pid) and generations, checked by ``_check_cell``."""
    from lift.models import generate

    if judge == "second" and complete:
        raise ValueError("the second judge sees a sample of the outputs; read it with complete=False")
    records = load_judgements(variant, set_name, lang, split, judge=judge)
    gpath = generate.generations_path(variant, set_name, lang, split)
    gens = generate.read_generations(gpath)
    _check_cell(records, gens, gpath, (variant, set_name, lang, split), role=judge, complete=complete,
                dtype=dtype or str(config.get("generation.dtype")))
    return records, gens


def load_records(variant: str, set_name: str, lang: str, split: str, *, judge: str = "primary",
                 complete: bool = True, dtype: str | None = None) -> list[JudgementRecord]:
    """The judgements of one variant/set/lang/split by the ``judge`` role, checked for reporting (the
    rules are in the module docstring), sorted by pid. Every reported per-prompt field (label, loose
    label, flags, judge failure) is read from here. ``complete=True`` for every reported number; the
    second judge sees a sample, so it needs ``complete=False``. ``dtype`` defaults to
    ``generation.dtype``; name another only for a precision Gate 0 licensed (gate0-fp16)."""
    return _read_cell(variant, set_name, lang, split, judge=judge, complete=complete, dtype=dtype)[0]


def load_labels(variant: str, set_name: str, lang: str, split: str, *, judge: str = "primary",
                complete: bool = True, dtype: str | None = None) -> dict[str, str]:
    """{pid: final label} of ``load_records`` (same arguments, same checks)."""
    records = load_records(variant, set_name, lang, split, judge=judge, complete=complete, dtype=dtype)
    return {r.pid: r.label for r in records}


def _describe(records: Sequence[JudgementRecord], gens: Sequence[GenerationRecord]) -> dict[str, Any]:
    out: dict[str, Any] = {"judge_id": records[0].judge if records else None,
                           "n_generations": len(gens), "n_judgements": len(records)}
    for name in PROVENANCE_META:
        out[name] = sorted({str(g.meta[name]) for g in gens if g.meta.get(name) is not None})
    return out


def provenance(variant: str, set_name: str, lang: str, split: str, *,
               judge: str = "primary") -> dict[str, Any]:
    """What one cell was made with: the judge id, the counts, and for each GenerationRecord.meta field
    of ``PROVENANCE_META`` (dtype, run_config, model_rev, harness, torch, transformers, gpu) the sorted
    distinct values present, as strings (older records lack some fields). It describes and checks
    nothing (``load_records`` does), so it can also say why a cell is refused."""
    from lift.models import generate

    records = load_judgements(variant, set_name, lang, split, judge=judge)
    gens = generate.read_generations(generate.generations_path(variant, set_name, lang, split))
    return {"variant": variant, "set": set_name, "lang": lang, "split": split, "judge": judge,
            "judge_dir": judge_dir(judge), **_describe(records, gens)}


def paired_labels(variant_a: str, variant_b: str, set_name: str, lang: str, split: str, *,
                  judge: str = "primary", dtype: str | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """({pid: label} of ``variant_a``, of ``variant_b``) on one set/lang/split for a paired comparison
    (McNemar, the paired bootstrap): the only way to read a pair (Gate 2, P07).

    Both cells are read with ``load_records(complete=True, dtype=dtype)``, and the pair is refused
    unless both have the same judge id (model, revision, precision and Controversial mode), generation
    dtype, model revision and harness version (``PAIR_MUST_MATCH``). Another GPU or torch/transformers
    version (``PAIR_MAY_DIFFER``) is allowed and logged as a warning, not returned; ``provenance``
    gives both cells' values for a report.
    """
    sides = []
    for variant in (variant_a, variant_b):
        records, gens = _read_cell(variant, set_name, lang, split, judge=judge, complete=True, dtype=dtype)
        sides.append(({r.pid: r.label for r in records}, _describe(records, gens)))
    (a, prov_a), (b, prov_b) = sides
    name = f"{variant_a} vs {variant_b} on {set_name}_{lang}_{split}"
    differ = {k: (prov_a[k], prov_b[k]) for k in PAIR_MUST_MATCH if prov_a[k] != prov_b[k]}
    if differ:
        raise ValueError(f"{name} cannot be paired: {differ} differ (paired comparisons never mix judges, "
                         "precisions, model revisions or harness versions)")
    if set(a) != set(b):
        raise ValueError(f"{name}: the two cells cover different pids")
    warn = {k: (prov_a[k], prov_b[k]) for k in PAIR_MAY_DIFFER if prov_a[k] != prov_b[k]}
    if warn:
        log.warning("%s: paired, but the generations differ in %s", name, warn)
    return a, b
