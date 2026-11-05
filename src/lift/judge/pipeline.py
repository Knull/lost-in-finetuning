"""Judge a generations file into a judgements file, and read judgements back for reporting.
Implemented in P02.

For each selected GenerationRecord: GlotLID's top 5 on the output (``lift.lang.LID_TOP_K``, the labels
``LangID.is_lang`` sums: decisions item 43; an empty or whitespace-only output gets ``[("und", 0.0)]``
without a LID call) -> unclear flags -> judge, in chunks -> final label (lift.judge.labels) -> one
JudgementRecord appended per item to
outputs/judgements/{judge_dir}/{variant}/{set}_{lang}_{split}.jsonl after each chunk.

- Exactly one judgement per selected generation, naming the output it judged (``output_sha``: the
  generation key identifies the inputs, and a regenerated output under the same key can differ). A
  rerun skips the keys already judged (a truncated last line is repaired first); an empty file (what
  ``lift restart`` leaves) is judged from the start. Duplicate keys raise.
- A judge on local hardware judges a cell only in the precision of its generations (``meta.dtype``):
  judgements in another could never be reported. Another raises ``OtherPrecisionError`` (from
  lift.models.generate) before any other check of the cell; its remedy is the file's ``--dtype`` or
  leaving the cell out, never a restart, also for a cell that is stale as well (a restart line run where
  the other precision runs would make that cell again in its precision).
- Judges are never mixed in one file: a file another judge began (another model, revision, labelling
  rules, precision or Controversial mode) raises ``MixedJudgesError``, whose remedies are that judge or
  the role's ``lift restart --judge ROLE`` line (``command``); only the restart line for judgements made
  under other labelling rules, which no current judge applies.
- A stale cell raises ``StaleJudgementsError``. It and both refusals above are
  ``lift.models.generate.StaleOutputError``s, so ``lift run-dev`` and ``lift judge`` set the cell aside
  and go on with the others. A message (and ``command``) names the ``lift restart`` line to run right
  after the job's ``lift sync pull``: a file moved aside by hand comes back with that pull. Judgements of
  generations no longer in the generations file or of another output than the one now under their key,
  and generations of changed prompts, name the restart of the whole cell (generations and every judge's
  judgements); judgements on another stack name ``--judge ROLE`` (only that role's judgements). Each
  line names the cell's precision when it is not ``generation.dtype`` (``lift restart`` refuses
  another).
- Every prompt stays in the denominator: a judge failure is written as label "unclear" with
  ``native.judge_failure`` set, and counted in the run manifest.
- Selection: every generation (the primary judge); ``keys=`` exactly the listed keys present in the
  file (``lift judge --keys``); ``sample=f`` P04's draw by pid (``second_judge_pids``) plus every
  key of the validation rounds. P04's validation/keys/second_judge/{variant}_{split}.txt lists are
  the canonical second-judge selection (``lift judge --judge second --keys``); ``--sample``
  reproduces their random part on a complete generations file: the same prompts in every variant
  and language, so the second judge's view stays paired by pid.
- The judgements directory comes from the kind of the role's config entry: an entry with a
  ``provider`` is the frontier LLM judge ("frontier"), a Qwen3Guard id is "qwen3guard".
- Every judge id carries the tag of the labelling rules its labels were made under
  (``lift.judge.labels.rules_tag``: the labels code's ``LABELS_VERSION``, configs/project.yaml
  ``unclear`` and ``languages.accept_labels``, the GlotLID entry ``langid`` and ``LID_TOP_K``): a
  frontier judge's id is ``{FrontierJudge.judge_id}#rules-<8 hex>``, a Qwen3Guard id
  ``{model}@{sha}#rules-<8 hex>#{dtype}#{strict|loose}``. So two sets of labelling rules, two precisions
  or two Controversial mappings never share a file, a read or a pair: after a change to any rule, each
  judged cell is judged again (its role's ``lift restart --judge ROLE`` line, which keeps the
  generations).
- A judge on local hardware (Qwen3Guard) exposes ``stack`` (``lift.models.generate.stack`` of its
  device: torch and transformers versions, GPU class) and writes it into every judgement's
  ``native["stack"]``: greedy guard outputs can flip borderline calls between stacks, so a
  judgements file is never extended on another stack (a complete one is re-entered without error).
  Frontier judgements carry no stack.

Reading:
- ``load_records`` (whole JudgementRecords) and ``load_labels`` ({pid: label}) are the door to every
  reported number, the loose label, flags and judge failures included. Each returned judgement must
  be of a generation now in the generations file, of its current output, made from the current
  prompt text; every generation of the cell in ``dtype`` under one run configuration and one stack;
  every judgement on one judge stack; and the judge must be the role's configured judge (model,
  revision when pinned to a sha, the labelling rules now in force; for Qwen3Guard the precision and
  Controversial mode; for a frontier judge everything its id names but the API version, so the rubric's
  version and the sampling settings of the role's entry too: an edited rubric is another judge).
  ``dtype`` defaults to ``generation.dtype`` (one precision for every reported number); any other needs Gate
  0's licence (``check_dtype_licence``), because every ``--dtype`` run names its dtype: a comparison
  that paired with the bf16 Gate 0 run gate0.json records now, and a cell made under the harness and
  decoding settings of that comparison, generated and judged (Qwen3Guard) on its GPU class. A refusal
  names the remedy for the side that is out of date: after a harness fix, the bf16 Gate 0 and then the
  comparison; for a cell of another harness or GPU class, its restart line (judged elsewhere: its
  judge's).
  ``complete=True`` (every reported number) also needs a generation for every prompt of the split
  and a judgement of every generation; ``complete=False`` is for the second judge's sample and runs
  in progress.
- ``provenance`` describes a cell (judge id and stack; generation dtype, run configuration, model
  revision, harness, chat template, decoding settings, library versions and GPU). ``paired_labels``
  reads two variants' cells and refuses a pair whose judge (labelling rules included), precision, model
  revision, harness, chat template or decoding settings differ. Every paired comparison downstream
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
from lift.io import append_jsonl, read_jsonl, sha256_obj, sha256_text, stable_json
from lift.judge import labels
from lift.lang import LID_TOP_K
from lift.models.generate import OtherPrecisionError, StaleOutputError, restart_command
from lift.paths import REPO_ROOT, judgements_dir, metrics_dir, validation_dir
from lift.schemas import GenerationRecord, JudgementRecord, UnclearFlags

log = logging.getLogger(__name__)

LICENCE_MIN_PROMPTS = 200       # decisions.md 27: fp16 agrees with bf16 on 95% of the same 200 prompts
# What a licensed cell's generations must have been made with (meta field -> where gate0-{dtype}.json records
# it): Gate 0's comparison says nothing about another harness, other decoding settings or another GPU class.
LICENCE_COVERS = {"harness": ("checks", "fp16", "harness"), "decoding": ("checks", "fp16", "decoding"),
                  "gpu_capability": ("gpu_capability",)}
ROLES: tuple[str, ...] = ("primary", "second")
_ROLE_ENTRIES = {"primary": "safety_primary", "second": "safety_second"}     # under judges: in project.yaml
CONTROVERSIAL_MODES = ("strict", "loose")
EMPTY_OUTPUT_LID: tuple[tuple[str, float], ...] = (("und", 0.0),)
QWEN3GUARD_SAFETY = ("Safe", "Unsafe", "Controversial")
PROVENANCE_META = ("dtype", "run_config", "model_rev", "harness", "template", "decoding", "torch",
                   "transformers", "gpu")
PAIR_MUST_MATCH = ("judge_id", "dtype", "model_rev", "harness", "template", "decoding")
PAIR_MAY_DIFFER = ("gpu", "torch", "transformers", "judge_stack")
_STACK_FIELDS = ("torch", "transformers", "gpu_capability")     # a generation without meta["stack"]
_KEY = re.compile(r"[0-9a-f]{64}")
_SHA = re.compile(r"[0-9a-f]{40}")
_DIR_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_JUDGE_PRECISION = re.compile(r"#([^#@]+)#(?:strict|loose)$")       # a Qwen3Guard judge id's #{dtype}#{mode}
_RULES_TAG = re.compile(r"#rules-[0-9a-f]{8}(?=#|$)")              # a judge id's labels.rules_tag segment


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


class StaleJudgementsError(StaleOutputError):
    """``judge_file`` cannot go on with a cell as it stands: its judgements were made on another stack,
    or of generations or prompts that changed since. A StaleOutputError, so a run sets the cell aside
    and goes on. ``command`` is the ``lift restart`` line that sets the cell's files aside (None for a
    file no role owns, which no restart line covers); the message names it."""

    def __init__(self, message: str, *, command: str | None = None) -> None:
        super().__init__(message)
        self.command = command


class MixedJudgesError(StaleOutputError):
    """``judge_file`` refuses a judgements file another judge began (another model, revision, labelling
    rules, precision or Controversial mode): judges are never mixed in one file. A StaleOutputError, so a
    run lists the cell and goes on. ``command`` is the role's ``lift restart ... --judge ROLE`` line (None
    for a file no role owns); the message also names the other remedy, finishing the file with its own
    judge, unless that judge can never be reported (another precision than the generations', or only
    other labelling rules)."""


class Judge(Protocol):
    """What ``judge_file`` needs. ``judge_id`` (revision included) goes into every record, so two
    judges never share a file; ``short`` names the judgements directory. A judge on local hardware
    also exposes ``stack`` (read with getattr; None or absent for an API judge) and writes it into
    every judgement's ``native["stack"]``."""

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
    Safe with one) and ``judge_label_loose``, so the loose sensitivity analysis needs no re-judging,
    and on every judgement, failures included, the judge's ``stack``. The judge id ends in
    ``#{rules tag}#{dtype}#{mode}`` (``labels.rules_tag``): labels from two sets of labelling rules, two
    precisions or two Controversial modes never share a judgements file, and resume and load refuse a mix;
    ``judge_file`` and ``load_records`` do the same for stacks.
    """

    short = "qwen3guard"

    def __init__(self, guard: Any, *, strict: bool, device: str | None, dtype: str) -> None:
        self.guard = guard
        self.strict = strict
        self.mode = "strict" if strict else "loose"
        self.device = device
        self.dtype = dtype
        self.judge_id = f"{guard.judge_id}#{labels.rules_tag()}#{dtype}#{self.mode}"
        self._stack: Any = None

    @property
    def stack(self) -> Any:
        """``lift.models.generate.stack`` of the judge's device (torch and transformers versions, GPU
        class), read on first use: it cannot change within a process."""
        if self._stack is None:
            from lift.models import generate

            self._stack = generate.stack(self.device)
        return self._stack

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
        return [self._outcome(item, r, self.stack) for item, r in zip(items, results, strict=True)]

    def _outcome(self, item: JudgeItem, r: Mapping[str, Any], stack: Any) -> JudgeOutcome:
        raw = r.get("raw") or ""
        if r.get("error"):
            failure = {"judge_failure": str(r["error"]), "safety": None, "categories": [], "refusal": None,
                       "consistent": None, "judge_label_loose": "unclear", "stack": stack}
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
            "stack": stack,
        }
        return JudgeOutcome(labels.map_qwen3guard(safety, refused, strict=self.strict), native, raw)


class FrontierAdapter:
    """The frontier LLM judge (``lift.judge.frontier.FrontierJudge``) as a Judge. ``native`` keeps the
    judge's label and reason, and ``model_echo`` (the model id that answered) when the result has
    one. ``stats`` passes through its call, token and cost counters for the run manifest. An API
    judge runs on no hardware of ours, so it has no ``stack``. The judge id is the frontier judge's own
    id and ``#{rules tag}`` (``labels.rules_tag``); the frontier judge's id is left as it is, so its call
    cache is reused when a cell is judged again under new labelling rules."""

    short = "frontier"
    device = None
    dtype = None
    stack = None

    def __init__(self, frontier: Any) -> None:
        self.frontier = frontier
        self.judge_id = f"{frontier.judge_id}#{labels.rules_tag()}"

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


def _same_file(a: Path, b: Path) -> bool:
    return Path(a).resolve() == Path(b).resolve()


def _role_of(out_path: Path, cell: tuple[str, str, str, str]) -> str | None:
    """The role whose judgements file of ``cell`` is ``out_path`` (``lift restart --judge ROLE`` sets
    aside exactly that file), or None."""
    for role in ROLES:
        try:
            if _same_file(judgements_path(judge_dir(role), *cell), out_path):
                return role
        except (KeyError, ValueError):          # an entry that names no judge kind
            continue
    return None


def _cell_restart(cell: tuple[str, str, str, str], gens_path: Path, out_path: Path | None = None, *,
                  dtype: str | None = None) -> str | None:
    """The ``lift restart`` line of the whole cell (its generations and every judge's judgements; ``dtype``
    the precision they were made in), when ``gens_path`` and ``out_path`` are the cell's own files; else
    None (a test's or Gate 0's files)."""
    from lift.models.generate import generations_path

    own = _same_file(gens_path, generations_path(*cell)) and (
        out_path is None or _same_file(out_path, judgements_path(Path(out_path).parent.parent.name, *cell)))
    return restart_command(*cell, dtype=dtype) if own else None


def _file_dtype(gens: Sequence[GenerationRecord]) -> str | None:
    """The one precision ``gens`` were made in (None: none recorded, or a mix), which every restart line of
    the cell names: ``lift restart`` refuses a cell made in another precision than its --dtype."""
    made = {g.meta.get("dtype") for g in gens}
    only = made.pop() if len(made) == 1 else None
    return str(only) if only else None


def _rejudge(restart: str | None, path: Path, *, judge_only: bool) -> str:
    """The remedy for judgements that no longer fit the cell's current generations: judge it again. The
    role's ``lift restart --judge`` sets aside only its judgements and keeps the generations."""
    if not judge_only:
        return _remake(restart, path)
    return (f"set these judgements aside and judge the cell again: run `{restart}` as a job line right after "
            "`lift sync pull`, followed by the line that failed (the generations are kept)")


def _remake(restart: str | None, path: Path) -> str:
    """The remedy for a cell whose generations no longer fit: make it again, generations included."""
    if restart is None:
        return f"set {path} aside by hand and make the cell again (no restart line covers that file)"
    return (f"set the cell aside and make it again: run `{restart}` as a job line right after "
            "`lift sync pull`, followed by the line that failed (a `lift judge` line needs `lift generate` "
            "for the cell first: the restart sets its generations aside too)")


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


def _check_prompt_texts(gens: Sequence[GenerationRecord], texts: Mapping[str, str], path: Path,
                        restart: str | None) -> None:
    """Every generation is of a prompt still in the prompt files, made from its current text (a fixed
    translation changes it); otherwise the cell is stale and is made again (``restart``)."""
    missing = sorted(g.pid for g in gens if g.pid not in texts)
    changed = sorted(g.pid for g in gens if g.pid in texts and g.prompt_sha != sha256_text(texts[g.pid]))
    problems = [f"no prompt text for {len(missing)} pids, e.g. {missing[:5]}"] if missing else []
    if changed:
        problems.append(f"{len(changed)} generations were made from another prompt text (the prompt changed "
                        f"after generation), e.g. {changed[:5]}")
    if problems:
        raise StaleJudgementsError(f"{path}: " + "; ".join(problems) + "; " + _remake(restart, path),
                                   command=restart)


def _resume_state(out_path: Path, judge_id: str, gens: Sequence[GenerationRecord],
                  cell: tuple[str, str, str, str], *, resume: bool,
                  restart: str | None, judge_only: bool = False) -> dict[str, JudgementRecord]:
    """{key: record} already in ``out_path``, after checking it holds this judge's judgements of
    these generations' current outputs only (a record without ``output_sha`` is not trusted). Another
    judge's records raise ``MixedJudgesError``. Judgements of generations that changed since are stale:
    ``restart`` judges the cell again (``judge_only``: the role's ``--judge`` line, which keeps the current
    generations; else the whole cell's)."""
    from lift.models.generate import repair_tail

    if not out_path.exists():
        return {}
    if not resume:
        raise FileExistsError(f"{out_path} exists; pass resume=True or move it aside")
    repair_tail(out_path)
    records = _read_judgements(out_path)
    others = sorted({r.judge for r in records} - {judge_id})
    if others:
        raise _mixed_judges(out_path, others, judge_id, _file_dtype(gens), restart if judge_only else None)
    gen_of = {g.key: g for g in gens}
    stale = sorted(r.pid for r in records if (r.variant, r.set, r.lang, r.split) != cell
                   or r.key not in gen_of or gen_of[r.key].pid != r.pid)
    if stale:
        raise StaleJudgementsError(
            f"{out_path}: {len(stale)} judgements are of generations that are not in the generations file "
            f"(regenerated?), e.g. pids {stale[:5]}; " + _rejudge(restart, out_path, judge_only=judge_only),
            command=restart)
    changed = sorted(r.pid for r in records if r.output_sha != sha256_text(gen_of[r.key].output))
    if changed:
        raise StaleJudgementsError(
            f"{out_path}: {len(changed)} judgements are of another output than the generation now under "
            f"their key, e.g. pids {changed[:5]}: generations changed since judging; "
            + _rejudge(restart, out_path, judge_only=judge_only), command=restart)
    return {r.key: r for r in records}


def _mixed_judges(out_path: Path, others: Sequence[str], judge_id: str, made: str | None,
                  restart: str | None) -> MixedJudgesError:
    """The refusal of a file another judge began. Its remedies: finish it with that judge, or ``restart``
    (the role's ``--judge`` line) to judge the cell again with this one. Only the second when the file's
    judge worked in another precision than the generations (``made``: a file from before judge_file refused
    that), or differs from this one only in its labelling rules (its rules tag, or none: a file from before
    judge ids carried one), as those judgements can never be reported."""
    if restart:
        again = (f"set its judgements aside and judge the cell again with this judge: run `{restart}` as a "
                 "job line right after `lift sync pull`, followed by the line that failed (the generations "
                 "are kept)")
    else:
        again = f"set {out_path} aside by hand and judge the cell again (no restart line covers that file)"
    head = (f"{out_path} holds judgements by {list(others)}, not {judge_id}; judges are never mixed in one "
            "file. ")
    precisions = sorted({m.group(1) for judge in others if (m := _JUDGE_PRECISION.search(judge))})
    if made and set(precisions) - {made}:
        why = f"They were judged in {precisions} of generations made in {made}, so they can never be reported"
        return MixedJudgesError(f"{head}{why}: {again}.", command=restart)
    if all(_RULES_TAG.sub("", judge) == _RULES_TAG.sub("", judge_id) for judge in others):
        why = ("They differ from this judge only in their labelling rules (the rules tag: lift.judge.labels, "
               "configs/project.yaml unclear.* and languages.accept_labels, the GlotLID entry langid), which "
               "changed since they were made, so they can never be reported")
        return MixedJudgesError(f"{head}{why}: {again}.", command=restart)
    own = ("Either finish the file with its own judge (its id names its revision, and for Qwen3Guard its "
           "precision and Controversial mode), or")
    return MixedJudgesError(f"{head}{own} {again}.", command=restart)


def _check_precision(judge: Any, gens: Sequence[GenerationRecord], path: Path) -> None:
    """A judge on local hardware (one with a ``dtype``: Qwen3Guard) judges a cell only in the precision its
    generations were made in: ``load_records`` reports a cell only with the judge of its generations' dtype,
    so other judgements would never count. The remedy is the file's ``--dtype`` or leaving the cell out,
    never a restart (it would discard the other precision's outputs)."""
    mine = getattr(judge, "dtype", None)
    made = sorted({str(g.meta["dtype"]) for g in gens if g.meta.get("dtype")})
    if mine is None or not made or made == [str(mine)]:
        return
    if len(made) > 1:
        raise ValueError(f"{path}: generations mix precisions {made}; one cell is one precision (sync never "
                         "merges two), so the file was edited by hand: set it aside by hand")
    raise OtherPrecisionError(
        f"{path} was generated in {made[0]}, and this judge ({judge.judge_id}) runs in {mine}: a cell is "
        "judged in the precision it was generated in, as no other judgements can be reported. Judge it with "
        f"--dtype {made[0]} where that precision runs (bf16 on an A100 or L4, fp16 only on the T4), or leave "
        "the cell out with --sets/--langs.", dtype=made[0])


def _stack_id(stack: Any) -> str:
    """A judge stack (``native["stack"]``; None for a judge without one) as one comparable string."""
    return stable_json(stack)


def _stack_moves(stacks: Collection[Any], stack: Any) -> dict[str, str]:
    """{field: "was -> now"} for each of torch, transformers and the GPU class in which the judgements'
    ``stacks`` differ from the judge's ``stack``."""
    moves = {}
    for name in _STACK_FIELDS:
        was = sorted({str(_field(s, name)) for s in stacks})
        if was != [str(_field(stack, name))]:
            moves[name] = f"{'/'.join(was)} -> {_field(stack, name)}"
    return moves


def _finish_on(moved: Collection[str]) -> str:
    """Where a file judged on another stack can still be finished. constraints.txt pins transformers;
    torch is the runtime's own CUDA build, which no install of ours changes."""
    where = []
    if "transformers" in moved:
        where.append("constraints.txt pins transformers: a runtime installed with it (pip install -c "
                     "constraints.txt) can finish the file when that is the version that judged it.")
    if {"torch", "gpu_capability"} & set(moved):
        where.append("torch is the runtime's own CUDA build: only the same runtime and GPU class can finish "
                     "the file.")
    return " ".join(where) or "Only the stack that started the file can finish it."


def _check_judge_stack(out_path: Path, existing: Collection[JudgementRecord], stack: Any, n_todo: int, *,
                       restart: str | None) -> None:
    """One judgements file is one judge stack: judging the ``n_todo`` remaining generations on another
    stack than the judgements already in ``out_path`` makes the cell stale, and ``restart`` (the role's
    ``lift restart --judge``) judges it again. A file with nothing left to judge is re-entered (logged),
    as ``generate`` reuses a finished generations file."""
    mine = _stack_id(stack)
    others = [r.native.get("stack") for r in existing if _stack_id(r.native.get("stack")) != mine]
    if not others:
        return
    if n_todo:
        moves = _stack_moves(others, stack)
        moved = ", ".join(f"{name} {move}" for name, move in moves.items()) or "torch, transformers or GPU"
        if restart:
            rejudge = (f"Or judge the cell again on this stack: run `{restart}` as a job line right after "
                       "`lift sync pull`, followed by the line that failed.")
        else:
            rejudge = f"Or set {out_path} aside by hand and judge the whole cell again."
        raise StaleJudgementsError(
            f"{out_path}: {len(others)} judgements were made on another stack ({moved}); one judgements file "
            f"is one stack, so the {n_todo} remaining are not judged here. {_finish_on(moves)} {rejudge}",
            command=restart)
    log.info("%s: everything selected is already judged (%d on another stack); nothing appended",
             out_path.name, len(others))


# ----------------------------------------------------------------------------- judging


def _flags(g: GenerationRecord, lid: Any, accept: Mapping[str, Sequence[str]],
           unclear: Mapping[str, Any]) -> UnclearFlags:
    lid_top = lid.top(g.output, k=LID_TOP_K) if g.output.strip() else list(EMPTY_OUTPUT_LID)
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
    ``sample=``) and the judge's cost counters, also when the run fails. The judgements file is
    written only once something has been judged (an empty one is judged from the start), and is
    extended only by the same judge on the same stack. Before anything is judged, a judge on local hardware
    in another precision than the generations raises ``OtherPrecisionError``, checked before anything else
    of the cell, so a cell of another precision is never offered a restart line, also when it is stale
    too. A file another judge began raises ``MixedJudgesError``, and a stale cell ``StaleJudgementsError``.
    """
    from lift.manifest import finish_run, start_run
    from lift.models.generate import read_generations

    generations_path = Path(generations_path)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    gens = read_generations(generations_path)
    if not gens:
        raise ValueError(f"{generations_path} holds no generations: generate the cell before judging it "
                         "(`lift generate` or `lift run-dev`; `lift restart` leaves an empty file)")
    cell = _one_cell(gens, generations_path)
    variant, set_name, lang, split = cell
    if isinstance(judge, str):
        judge = make_judge(judge, device=device, dtype=dtype)
    elif device is not None or dtype is not None:
        raise ValueError("device and dtype apply only when judge_file builds the judge from a role")
    if not _DIR_NAME.fullmatch(str(judge.short)):
        raise ValueError(f"judge.short must be a plain directory name, got {judge.short!r}")
    _check_precision(judge, gens, generations_path)         # first: such a cell is never offered a restart
    made = _file_dtype(gens)
    texts = dict(prompts_by_pid) if prompts_by_pid is not None else prompts.texts_by_pid(set_name, lang)
    _check_prompt_texts(gens, texts, generations_path, _cell_restart(cell, generations_path, dtype=made))
    selected, selection = _select(gens, sample=sample, keys=keys, seed=seed, set_name=set_name, split=split)

    out_path = (Path(out_path) if out_path is not None
                else judgements_path(judge.short, variant, set_name, lang, split))
    role = _role_of(out_path, cell)
    rejudge = restart_command(*cell, judge=role, dtype=made) if role else None
    existing = _resume_state(out_path, judge.judge_id, gens, cell, resume=resume, judge_only=role is not None,
                             restart=rejudge or _cell_restart(cell, generations_path, out_path, dtype=made))
    todo = sorted((g for g in selected if g.key not in existing), key=lambda g: g.pid)
    n_resumed = len(selected) - len(todo)
    if existing:
        _check_judge_stack(out_path, list(existing.values()), getattr(judge, "stack", None), len(todo),
                           restart=rejudge)

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
    """The judge ids a reported file of ``role`` may hold, and how to name them. A frontier judge's id is
    the configured judge's exactly but for the API version its SDK speaks: the provider, model, rubric
    version (file stem and sha) and sampling settings of the role's entry, as ``FrontierJudge.judge_id``
    names them, then the tag of the labelling rules now in force (``#{labels.rules_tag()}``). A Qwen3Guard
    id names the configured model (and revision, when the config pins a sha), then the rules tag and
    ``#{dtype}#{mode}``."""
    entry = _judge_entry(role)
    rules = f"#{labels.rules_tag()}"
    if _entry_kind(role, entry) == "frontier":
        from lift.judge.frontier import _check_choice, _settings, _settings_tag, load_rubric

        provider, model, where = entry.get("provider"), entry.get("model"), _ROLE_ENTRIES[role]
        _check_choice(provider, model, where)
        settings = _settings(provider, where, {})
        tag = _settings_tag(settings["temperature"], settings["effort"], settings["thinking"],
                            settings["max_tokens"])
        head = f"{provider}/{model}@{load_rubric(entry['rubric']).version}+{tag}+api"
        return re.compile(re.escape(head) + "[^#]+" + re.escape(rules)), f"{head}<version>{rules}"
    revision = str(entry.get("revision", ""))
    pinned = bool(_SHA.fullmatch(revision))
    tail = f"{rules}#{dtype}#{_controversial_mode(role, entry)}"
    pattern = re.escape(f"{entry['id']}@") + (re.escape(revision) if pinned else "[^@#]+") + re.escape(tail)
    return re.compile(pattern), f"{entry['id']}@{revision if pinned else '<sha>'}{tail}"


def check_dtype_licence(dtype: str, model_rev: str | None) -> None:
    """Raise ValueError unless Gate 0 licensed reporting ``dtype`` (decisions.md 27): outputs/metrics/
    gate0-{dtype}.json (written by Gate 0's comparison run, ``lift smoke --compare-dtype {dtype}`` on the
    T4) with "pass" true, checks.fp16.pass true and checks.fp16.n at least ``LICENCE_MIN_PROMPTS`` (the
    95% rule is on 200 prompts, so a quick ``--n-precision`` run licenses nothing), and the reference
    run's own Gate 0 (gate0.json, bf16 on the A100) with "pass" true, both with revisions.global equal to
    ``model_rev`` when it is given. gate0.json must still be the run the comparison paired with: its
    checks.precision.digest (that run's outputs and labels) equal to the licence's
    checks.fp16.reference_digest, so a bf16 Gate 0 re-run since (after a harness fix, say) needs a new
    comparison. A T4 fp16 run counts only if Gate 0's fp16-vs-bf16 check passed (CLAUDE.md), and nothing
    counts before Gate 0 itself passed; a comparison run with --skip could otherwise pass against a failed
    reference. The error lists every problem of both reports, a missing or unreadable one included, so a
    failed fp16 verdict shows although gate0.json is missing too. Every ``--dtype`` run names its dtype, so
    naming one licenses nothing. ``load_records`` asks whenever the requested dtype is not
    ``generation.dtype``, and also checks that the cell was made as the comparison was (``LICENCE_COVERS``)
    and judged on its GPU class (``_coverage_problems``).
    """
    licence, reference = metrics_dir() / f"gate0-{dtype}.json", metrics_dir() / "gate0.json"
    rerun = f"`lift smoke --compare-dtype {dtype}` on the T4"
    compared, problems = _gate0_report(licence, rerun, model_rev, fp16=True)
    current, more = _gate0_report(reference, "`lift smoke` on the A100", model_rev, fp16=False)
    problems += more
    if compared is not None and current is not None:
        problems += _reference_problems(compared, current, licence.name, rerun)
    if problems:
        raise ValueError(f"Gate 0's licence {licence} (with its bf16 reference {reference.name}) does not "
                         f"license {dtype}: " + "; ".join(problems))


def _gate0_report(path: Path, writer: str, model_rev: str | None, *, fp16: bool) -> tuple[Any, list[str]]:
    """One Gate 0 report (None when missing or unreadable) and why it does not count toward the licence:
    missing or unreadable, "pass" not true, revisions.global not ``model_rev``, and with ``fp16`` an fp16
    check that did not pass on at least ``LICENCE_MIN_PROMPTS`` prompts."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, [f"{path.name} does not exist ({writer} writes it, `lift sync pull` fetches it)"]
    except (OSError, ValueError) as exc:          # unreadable, or not JSON
        return None, [f"{path.name} cannot be read ({type(exc).__name__})"]
    flags = [('"pass"', ("pass",)), *([("checks.fp16.pass", ("checks", "fp16", "pass"))] if fp16 else [])]
    problems = [f"{path.name} {label} is {_field(doc, *keys)!r}, not true" for label, keys in flags
                if _field(doc, *keys) is not True]
    n = _field(doc, "checks", "fp16", "n")
    if fp16 and (isinstance(n, bool) or not isinstance(n, int) or n < LICENCE_MIN_PROMPTS):
        problems.append(f"{path.name} checks.fp16.n is {n!r}: the comparison must cover at least "
                        f"{LICENCE_MIN_PROMPTS} prompts (decisions.md 27)")
    licensed = _field(doc, "revisions", "global")
    if model_rev is not None and licensed != model_rev:
        problems.append(f"{path.name} revisions.global is {licensed!r}, not {model_rev}")
    return doc, problems


def _reference_problems(licence: Any, reference: Any, name: str, rerun: str) -> list[str]:
    """Why gate0.json is not the bf16 Gate 0 run the licence's comparison paired with: its
    checks.precision.digest must be the licence's checks.fp16.reference_digest. Two stored digests that
    differ mean that this machine holds an old copy of one report (a pull settles it), or that the bf16
    Gate 0 was re-run since (after a harness fix, say): that replaces both its verdict and its run, which
    no comparison has seen yet. A comparison against a gate0.json that does not pass would not run, so
    then the remedy is the bf16 Gate 0, not a comparison."""
    compared = _field(licence, "checks", "fp16", "reference_digest")
    current = _field(reference, "checks", "precision", "digest")
    if _field(reference, "pass") is True:
        new = f"re-run {rerun} against the current bf16 Gate 0"
    else:
        new = ("the bf16 Gate 0 comes first, as gate0.json does not pass (`lift smoke` on an A100; a "
               "comparison against a failed one would not run)")
    if not isinstance(compared, str) or not compared:
        return [f"{name} checks.fp16.reference_digest is {compared!r}: it does not say which bf16 Gate 0 run "
                f"it compared with, so {new}"]
    if current != compared:
        shown = current[:12] if isinstance(current, str) else repr(current)
        return [f"gate0.json checks.precision.digest is {shown}, not the {compared[:12]} {name} compared "
                f"with: gate0.json and {name} name different bf16 Gate 0 runs. Pull both (`lift sync pull`), "
                "as this machine may hold an old copy of one; if the runs repo's copies differ too, the bf16 "
                f"Gate 0 was re-run since the comparison, so {new}"]
    return []


def _field(node: Any, *keys: str) -> Any:
    """``node[k1][k2]...``, or None where a level is missing or not a mapping."""
    for key in keys:
        node = node.get(key) if isinstance(node, dict) else None
    return node


def _licence_problems(dtype: str, gens: Sequence[GenerationRecord], records: Sequence[JudgementRecord],
                      cell: tuple[str, str, str, str], role: str) -> list[str]:
    """Why Gate 0's licence for ``dtype`` does not cover the cell of ``role``'s ``records``: its
    generations' revision of Global (``check_dtype_licence``), or how they and the judgements were made
    (``_coverage_problems``)."""
    problems = []
    unnamed = sum(not g.meta.get("model_rev") for g in gens)
    if unnamed:
        problems.append(f"{unnamed} generations name no model revision for Gate 0's {dtype} licence to cover")
    for rev in sorted({str(g.meta["model_rev"]) for g in gens if g.meta.get("model_rev")}) or [None]:
        try:
            check_dtype_licence(dtype, rev)
        except ValueError as exc:
            problems.append(str(exc))
    return list(dict.fromkeys(problems + _coverage_problems(dtype, gens, records, cell, role)))


def _coverage_problems(dtype: str, gens: Sequence[GenerationRecord], records: Sequence[JudgementRecord],
                       cell: tuple[str, str, str, str], role: str) -> list[str]:
    """Why Gate 0's comparison for ``dtype`` says nothing about the cell, and the remedy
    (``_coverage_remedies``). Each ``LICENCE_COVERS`` field of the generations (harness, decoding settings,
    GPU class) must equal what the licence records of its paired runs: a cell made on another GPU class (an
    A100 job given --dtype fp16) or under another harness never had its precision compared. So must the GPU
    class of every judgement with a ``stack`` (Qwen3Guard; an API judge runs on no GPU of ours): the
    comparison ran generator and judge both on the T4 (decisions.md 27), while an fp16 judge may also run on
    a CPU or on another GPU below compute 8. Empty when the licence cannot be read (``check_dtype_licence``
    says so)."""
    path = metrics_dir() / f"gate0-{dtype}.json"
    try:
        licence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    problems, moved = [], set()
    for name, keys in LICENCE_COVERS.items():
        where, covered = f"{path.name} {'.'.join(keys)}", _field(licence, *keys)
        if covered is None:
            problems.append(f"{where} is missing: the licence does not say which {name} its comparison "
                            "ran with")
            continue
        off = sorted(g.pid for g in gens if g.meta.get(name) != covered)
        if off:
            moved.add(name)
            made = sorted({str(g.meta.get(name)) for g in gens})
            problems.append(f"{len(off)} generations were made with {name} {made}, not {where} "
                            f"{covered!r}, e.g. {off[:5]}")
    gpu = _field(licence, *LICENCE_COVERS["gpu_capability"])
    judged_on = {r.pid: _field(r.native["stack"], "gpu_capability") for r in records
                 if r.native.get("stack") is not None}
    off = sorted(pid for pid, on in judged_on.items() if on != gpu)
    if gpu is not None and off:
        moved.add("judge")
        classes = sorted({str(on) for on in judged_on.values()})
        problems.append(f"{len(off)} judgements were made on GPU class {classes}, not {path.name} "
                        f"gpu_capability {gpu!r}, e.g. {off[:5]}")
    if problems:
        problems.append(f"a cell in {dtype} counts only when generated under the harness and decoding "
                        "settings of Gate 0's comparison, and generated and judged on its GPU class: "
                        + "; ".join(_coverage_remedies(dtype, licence, gens, moved, cell, role)))
    return problems


def _coverage_remedies(dtype: str, licence: Any, gens: Sequence[GenerationRecord], moved: Collection[str],
                       cell: tuple[str, str, str, str], role: str) -> list[str]:
    """What makes Gate 0's comparison cover the cell, by the side of each mismatch in ``moved`` that is out
    of date. The code knows its harness (``HARNESS_VERSION``): a licence of another one predates a harness
    fix, and a new comparison pairs only with a bf16 Gate 0 of this harness, so that is re-run first (a T4
    comparison against the old one would not run); a cell of another one is made again. Decoding settings
    can also move without a harness bump (a stop set or cap), so for them both remedies are named. A cell
    made on another GPU class is made again on the T4, and one only judged elsewhere is judged again
    there."""
    from lift.models.generate import HARNESS_VERSION

    made, version = _file_dtype(gens), HARNESS_VERSION
    gate0 = ("re-run the full bf16 Gate 0 on an A100 (`lift smoke`), then "
             f"`lift smoke --compare-dtype {dtype}` on the T4 (a comparison pairs only with a bf16 Gate 0 of "
             "the same harness and decoding settings)")
    remake = (f"make the cell again on the T4: run `{restart_command(*cell, dtype=made)}` as a job line "
              "right after `lift sync pull`, then generate and judge it there")
    harness = "harness" in moved
    old_cell = harness and any(g.meta.get("harness") != version for g in gens)
    out = []
    if harness and _field(licence, *LICENCE_COVERS["harness"]) != version:
        out.append(f"the bf16 Gate 0 and its T4 comparison predate this harness (version {version}): {gate0}")
    if old_cell:
        out.append(f"the cell was made under another harness than this code's (version {version}): {remake}")
    if "decoding" in moved and not harness:
        out.append(f"if the decoding settings changed since the comparison (a stop set or cap), {gate0}; "
                   f"if the cell is the older one, {remake}")
    if "gpu_capability" in moved:
        out.append(f"the cell was made on another GPU class than the comparison's: {remake}")
    if "judge" in moved and not old_cell and "gpu_capability" not in moved:
        rejudge = restart_command(*cell, judge=role, dtype=made)
        out.append(f"judge the cell again on the T4: run `{rejudge}` as a job line right after "
                   f"`lift sync pull`, then judge it there with --dtype {dtype} (the generations are kept)")
    return out or [f"re-run `lift smoke --compare-dtype {dtype}` on the T4 against the current bf16 Gate 0"]


def _generation_stack(meta: Mapping[str, Any]) -> str:
    """The stack a generation ran on: ``meta["stack"]``, else (a record from before it) the same digest
    lift.models.generate takes of its torch and transformers versions and GPU class."""
    return str(meta.get("stack") or sha256_obj({k: meta.get(k) for k in _STACK_FIELDS})[:16])


def _one_configuration(gens: Sequence[GenerationRecord], records: Sequence[JudgementRecord],
                       dtype: str) -> list[str]:
    """Why a cell is not one precision (``dtype``; Gate 0's licence for another than ``generation.dtype`` is
    ``_licence_problems``), one run configuration and one stack for its generations, and one judge stack
    for its judgements."""
    problems = []
    off = sorted(g.pid for g in gens if g.meta.get("dtype") != dtype)
    if off:
        dtypes = sorted({str(g.meta.get("dtype")) for g in gens})
        problems.append(f"{len(off)} generations are not in {dtype} (the cell has {dtypes}), e.g. {off[:5]}; "
                        "every reported number uses one precision, another only when named with dtype= "
                        "and licensed by Gate 0")
    run_configs = sorted({str(g.meta.get("run_config")) for g in gens})
    if len(run_configs) > 1:
        problems.append(f"generations mix {len(run_configs)} run configurations {run_configs[:4]}")
    stacks = sorted({_generation_stack(g.meta) for g in gens})
    if len(stacks) > 1:
        moved = [f"{k} {'/'.join(sorted({str(g.meta.get(k)) for g in gens}))}" for k in _STACK_FIELDS
                 if len({str(g.meta.get(k)) for g in gens}) > 1]
        problems.append(f"generations mix {len(stacks)} stacks {stacks[:4]} "
                        f"({', '.join(moved) or 'torch/transformers version or GPU class'}): one cell is one "
                        "stack (a file merged from two machines, or extended after an upgrade?)")
    judge_stacks = sorted({_stack_id(r.native.get("stack")) for r in records})
    if len(judge_stacks) > 1:
        problems.append(f"judgements mix {len(judge_stacks)} judge stacks {judge_stacks[:4]}: one cell is "
                        "judged on one stack")
    return problems


def _check_cell(records: Sequence[JudgementRecord], gens: Sequence[GenerationRecord], gpath: Path,
                cell: tuple[str, str, str, str], *, role: str, complete: bool, dtype: str) -> None:
    """Raise unless ``records`` can be reported (the rules are in the module docstring)."""
    variant, set_name, lang, split = cell
    texts = {p.pid: p.text for p in prompts.load(set_name, lang, split)}
    gen_of = {g.key: g for g in gens}
    current = [r for r in records if r.key in gen_of and gen_of[r.key].pid == r.pid]
    current_keys = {r.key for r in current}
    prompted = gens if complete else [gen_of[r.key] for r in current]   # whose prompt must be current
    stale = [       # the cell's files no longer belong together: it is made again
        ([r.pid for r in records if r.key not in current_keys],
         f"judgements are of generations no longer in {gpath.name}"),
        ([r.pid for r in current if r.output_sha != sha256_text(gen_of[r.key].output)],
         "judgements are of another output than the generation now under their key (generations changed "
         "since judging)"),
        ([g.pid for g in prompted if g.pid not in texts],
         f"generations are of pids that are not {split} prompts"),
        ([g.pid for g in prompted if g.pid in texts and g.prompt_sha != sha256_text(texts[g.pid])],
         "generations were made from another prompt text"),
    ]
    checks = [([g.pid for g in gens if (g.variant, g.set, g.lang, g.split) != cell],
               "generations belong to another variant/set/lang/split"), *stale]
    if complete:
        judged = {r.key for r in records}
        checks += [
            ([g.pid for g in gens if g.key not in judged], f"of {len(gens)} generations have no judgement"),
            (list(set(texts) - {g.pid for g in gens}), f"of {len(texts)} prompts have no generation"),
        ]
    problems = [f"{len(pids)} {what}, e.g. {sorted(pids)[:5]}" for pids, what in checks if pids]
    problems += _one_configuration(gens, records, dtype)
    if dtype != str(config.get("generation.dtype")):
        problems += _licence_problems(dtype, gens, records, cell, role)
    rule, shown = _expected_judge(role, dtype)
    others = sorted({r.judge for r in records if not rule.fullmatch(r.judge)})
    if others:
        problems.append(f"judged by {others}, not the configured {role} judge {shown}")
    if any(pids for pids, _ in stale):
        restart = restart_command(*cell, dtype=_file_dtype(gens))
        problems.append(f"to make the cell again, run `{restart}` as a job line right after "
                        "`lift sync pull`, then generate and judge it")
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
    ``generation.dtype``; another is refused unless Gate 0 licensed it for the cell's revision of
    Global (``check_dtype_licence``)."""
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
    out["judge_stack"] = sorted({_stack_id(r.native["stack"]) for r in records
                                 if r.native.get("stack") is not None})
    return out


def provenance(variant: str, set_name: str, lang: str, split: str, *,
               judge: str = "primary") -> dict[str, Any]:
    """What one cell was made with: the judge id, the counts, for each GenerationRecord.meta field of
    ``PROVENANCE_META`` (dtype, run_config, model_rev, harness, template, decoding, torch, transformers,
    gpu) the sorted distinct values present, as strings (older records lack some fields), and the
    judge's ``judge_stack`` (canonical JSON; none for an API judge). It describes and checks nothing
    (``load_records`` does), so it can also say why a cell is refused."""
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
    unless both have the same judge id (model, revision, labelling rules, precision and Controversial
    mode), generation dtype, model revision, harness version, chat template and decoding settings
    (``PAIR_MUST_MATCH``): two arms fed other token ids would confound drift with formatting, and two arms
    labelled under other rules would confound it with the rules. Another GPU, torch/transformers
    version or judge stack (``PAIR_MAY_DIFFER``) is allowed and logged as a warning, not returned;
    ``provenance`` gives both cells' values for a report.
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
                         "precisions, model revisions, harness versions, chat templates or decoding)")
    if set(a) != set(b):
        raise ValueError(f"{name}: the two cells cover different pids")
    warn = {k: (prov_a[k], prov_b[k]) for k in PAIR_MAY_DIFFER if prov_a[k] != prov_b[k]}
    if warn:
        log.warning("%s: paired, but the generations differ in %s", name, warn)
    return a, b
