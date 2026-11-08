"""Hindi utility of a variant against stock: both sides' answers, pairwise verdicts in both orders, the
win rate, Hindi fidelity and the AuditKIT record (``lift utility``). Implemented in P03.

What is compared. The held-out Hindi utility instructions of one split (``lift.prompts.load("utility",
"hi", split)``: 100 dev, 200 final, never trained on), answered by stock and by the candidate through the
frozen harness (``lift.models.generate``, set "utility"): outputs/generations/{variant}/utility_hi_{split}
.jsonl. Any non-stock variant can be the candidate (an adapter for Gate 1, a repaired model for the
selection rule); stock is always the reference.

Rules, and why:
- Pairing (``pair_generations``). Both answer files must answer every prompt of the split (rule 3: a pair
  is never judged on the rest when an answer is missing), from the current prompt text (``prompt_sha``),
  as one file each of one run configuration and one stack, every record naming the provenance it is paired
  on; and the two sides must agree on precision, model revision, harness, chat template and decoding
  settings (``pipeline.PAIR_MUST_MATCH``, as ``pipeline.paired_labels`` refuses a safety pair): answers fed
  other token ids would confound the adapter's gain with formatting. The precision is the run's
  (``generation.dtype`` unless ``--dtype``). Another one counts only under Gate 0's licence for both cells
  (``pipeline._licence_problems``, the rule every safety cell is read under), and its files carry the dtype
  in their names, so they never replace the reported precision's. A refusal is a ``UtilityRefused`` that
  names its remedy.
- Checked answers (``record_checked``, ``check_answers``). A pair is judged only from answer files a GPU line
  checked against their variant as it is registered now. Every ``lift utility`` run but ``--judge-only``
  loads both sides and passes each file through ``lift.models.generate.generate``, also when the file
  answers every prompt: it appends nothing to a current file and refuses one made under another
  configuration (another adapter or registry entry, model revision, chat template, decoding settings or
  harness: a StaleOutputError naming the cell's ``lift restart`` line), so the answers of a replaced adapter
  are never reported as the new one's (rule 4). The command then records the check beside the answers,
  outputs/utility/checked/{variant}/utility_hi_{split}.json: the file's sha256 and run configuration, and
  the adapter hash, intervention digest, model revision and harness version the variant was built with.
  ``evaluate`` reads it without loading a model (``--judge-only``, on the laptop) and refuses a side whose
  check is missing or names another file, or whose registry entry (its intervention digest, a pinned
  adapter hash), harness version or pinned model revision has moved since: the GPU line checks it again
  (``--generate-only``), or the cell's ``lift restart`` line starts it over. A side generate refuses as
  stale has its check withdrawn, and the pair's verdicts, metrics and AuditKIT records retired
  (``withdraw``): generate compares a file with more than a check names (decoding settings, the chat
  template, the cache-key version), so a check it contradicts must go, or the laptop would still judge, and
  Gate 1 count, the answers the GPU line refused. The job's push carries the emptied files. A file made in
  another precision is not stale, and withdraws nothing.
- Both orders. Every pair is judged twice in one ``pairwise`` call: order AB shows the candidate as A and
  stock as B, order BA the reverse. In each order the candidate scores 1 for a win, 0.5 for a tie and 0 for
  a loss (so "B" in order BA is a win), and the pair's score is the mean of its two orders: 1, 0.75, 0.5,
  0.25 or 0. Position bias cancels, and the win rate is the mean score (``lift.stats.win_rate``, a bootstrap
  interval clustered by pid; parity is 0.5).
- Failures stay in the denominator. A failed order (a refusal, a reply unparseable after the reminder turn,
  an item-level API error) is stored as "TIE" with its cause in ``error_ab``/``error_ba`` and counts 0.5.
  Every report gives the failed orders, the pairs with one, and the win rate with every failed order
  counted as a loss and as a win: the most failures can move it. A judge that stops the run
  (FrontierConfigError, SpendCapExceeded, FrontierUnavailable) writes no verdict: its finished calls are
  cached, so a rerun resumes for free.
- Verdicts. One UtilityVerdict per pid in outputs/utility/{judge_dir}/{variant}/utility_hi_{split}.jsonl,
  judge_dir "frontier" for judges.utility and "fallback" for the local Qwen3-8B; P04's blind check reads
  them (``lift.validation.utility.load_verdicts``). Not under outputs/judgements/: lift.sync pairs every
  file there with a generations file as JudgementRecords. Each verdict names the two answers it judged
  (generation keys and text sha256), so a file is current only while both answer files are. One that is
  not (an answer made again, a pid missing) is judged again and rewritten whole, atomically, never
  extended; a current one is reused as it is, with no judge call. One file is one judge: a file another
  judge wrote is refused, and ``--rejudge`` copies it to outputs/.aside/ (never synced; named
  ``*.jsonl.aside``, because P04's scan reads every .jsonl under outputs/ as a possible verdict file),
  leaves it empty and judges every pair again.
- Judges (``make_judge``). "frontier" is ``FrontierJudge.from_config("utility")`` with its own call cache,
  outputs/judge_cache/frontier/utility__{model}.jsonl, so it never loads the safety second judge's file.
  Its whole 2N-request set first goes to ``prefill_pairwise``, which sends the uncached requests as Message
  Batches when judges.utility.bulk is batch (a no-op otherwise) and puts the replies in the call cache: the
  synchronous ``pairwise`` call that follows then decides every verdict exactly as without it. Prefilling
  is skipped on request (``lift utility --sync``), and with ``wait=False`` it returns at once with the
  batches pending (FrontierBatchPending: rerun the same line to collect). A synchronous pass never pays
  twice for a request an earlier batch holds (one left by ``wait=False`` or a stopped wait): ``pairwise``
  collects that batch first once it has ended, and while it still runs stops with FrontierBatchPending,
  judging nothing, so ``--sync`` exits 3 then too. "fallback" is
  lift.judge.local_pairwise, used only when asked for: no fallback model ever answers in the frontier
  judge's place.
- Reported beside the win rate (``summarize``): outcome and score counts; position consistency and bias;
  Hindi fidelity of both sides' answers, the share whose top GlotLID label is a Hindi variety
  (languages.accept_labels.hi) and the share passing ``is_lang(text, "hi")`` (an empty answer is not Hindi
  and stays in the denominator); both answer files' provenance and cap-hit rates; and all of it again by
  source (pid utl-aya: Aya Dataset, human-written Hindi; utl-ind: IndicAlign Dolly_T, machine-translated).
  Metrics go to outputs/metrics/utility/{variant}_{split}.json and .md (UTF-8, LF, atomic), then
  ``lift.audit.record_utility`` writes the AuditKIT record from the same files. Version 2 of the metrics
  also names what the numbers were made from, so a reader can tell when they no longer describe the files
  or the registry: besides each answer file's sha256 (``generation.{side}.file_sha256``) and the verdict
  file's (``verdicts_sha256``), the candidate's adapter hash and intervention digest
  (``candidate_adapter_hash``, ``candidate_intervention``) and each side's recorded check
  (``generation.{side}.checked``: its path and sha256, and the run configuration, adapter hash,
  intervention, model revision and harness it names).
- Retired, never left behind (``retire``). What lift utility derived from a pair (both judge dirs' verdict
  files, the metrics, the AuditKIT utility records) must not outlive the answers it describes: P04 reads
  verdicts by pid and Gate 1 reads the metrics. When ``evaluate`` refuses a pair, when a GPU line refuses
  a side as stale (``withdraw``), and when ``lift restart`` starts a utility cell over (for stock's cell,
  every candidate's: each was paired with stock's answers), they are copied to outputs/.aside/ (never
  synced; verdict copies are named ``*.jsonl.aside``) and left empty, never removed: sync never deletes, so
  the next push replaces the runs repo's copies, and every reader takes an empty file for none. A refused
  pair keeps its answer checks. ``check_answers`` reads each one again on every run, against its file, the
  registry, the harness and the pin, so a kept check never lets stale answers through, while only a GPU
  line can make one again; and a refusal may be this checkout's own (a registry entry, harness or pin it
  is behind the GPU job's ref on), which a git pull settles: ``--judge-only`` then judges the pair again
  (the frontier judge from its call cache), with no GPU line. A check goes only with the answers it names
  (``lift restart`` of the cell) or when a GPU line refuses them (``withdraw``).
- No instruction, answer or judge reason is ever logged: counts, ids, paths and hashes only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pydantic

from lift import config
from lift import prompts as prompt_files
from lift.cachekey import intervention_digest
from lift.io import read_jsonl, sha256_file, sha256_text, write_jsonl
from lift.judge import pipeline
from lift.models import generate as harness
from lift.models.generate import (
    SetAside,
    aside_root,
    generations_path,
    read_generations,
    record_stack,
    restart_command,
)
from lift.models.load import DTYPES
from lift.paths import metrics_dir, output_dir, utility_dir
from lift.schemas import GenerationRecord, UtilityVerdict
from lift.stats.paired import win_rate

log = logging.getLogger(__name__)

SET, LANG, REFERENCE = "utility", "hi", "stock"
JUDGE_DIRS = ("frontier", "fallback")          # judges.utility, and its local fallback (local_pairwise)
CHECKED_DIR = "checked"                        # outputs/utility/checked/: GPU lines' checks (record_checked)
WINNERS = ("A", "B", "TIE")
SCORE_AB = {"A": 1.0, "TIE": 0.5, "B": 0.0}    # the candidate's score in order AB, where it is shown as A
SCORE_BA = {"B": 1.0, "TIE": 0.5, "A": 0.0}    # ... in order BA, where it is shown as B
OUTCOME_AB = {"A": "win", "TIE": "tie", "B": "loss"}   # the candidate's outcome of one order, likewise
OUTCOME_BA = {"B": "win", "TIE": "tie", "A": "loss"}
SCORES = (1.0, 0.75, 0.5, 0.25, 0.0)           # a pair's possible scores, the mean of its two orders
SOURCES = {"aya": "Aya Dataset (human-written Hindi)",
           "ind": "IndicAlign Dolly_T (machine-translated Hindi)"}
N_BOOT = 10_000
METRICS_VERSION = 2              # 2: the registry and check fields (candidate_adapter_hash, ..., checked)
CHECK_VERSION = 1
# What a recorded check must name for check_answers to compare it with the file, the registry and the pin.
CHECK_FIELDS = ("variant", "split", "dtype", "sha256", "run_config", "adapter_hash", "intervention",
                "model_rev", "harness")
ASIDE_SUFFIX = ".aside"
_SHA = re.compile(r"[0-9a-f]{40}")
# What a pair's two answer files must agree on (the safety pairs' rule, less the judge: none judged them yet),
# and what every record must name for that check to mean anything.
MATCH = tuple(k for k in pipeline.PAIR_MUST_MATCH if k != "judge_id")
REQUIRED_META = (*MATCH, "run_config")
MAY_DIFFER = tuple(k for k in pipeline.PAIR_MAY_DIFFER if k != "judge_stack")
_PID = re.compile(r"utl-(aya|ind)-\d{5}")


class UtilityRefused(ValueError):
    """The pair of answer files, or a verdict file, cannot be judged or reported as it stands. The message
    says why and names the remedy; ``lift utility`` lists it and exits 3."""


# ------------------------------------------------------------------------------------------ paths --


def _tag(dtype: str | None) -> str:
    """"-{dtype}" for a precision other than ``generation.dtype``: its files never replace the
    reported ones."""
    return f"-{dtype}" if dtype and dtype != config.get("generation.dtype") else ""


def _judge_dir(name: str) -> str:
    if name not in JUDGE_DIRS:
        raise ValueError(f"the utility judge is one of {JUDGE_DIRS}, got {name!r}")
    return name


def verdicts_path(judge_dir: str, variant: str, split: str, *, dtype: str | None = None) -> Path:
    """outputs/utility/{judge_dir}/{variant}/utility_hi_{split}.jsonl (``-{dtype}`` before the suffix for
    another precision than ``generation.dtype``)."""
    return utility_dir() / _judge_dir(judge_dir) / variant / f"{SET}_{LANG}_{split}{_tag(dtype)}.jsonl"


def metrics_paths(variant: str, split: str, *, dtype: str | None = None) -> tuple[Path, Path]:
    """outputs/metrics/utility/{variant}_{split}.json and .md (variant names hold dots, so no with_suffix)."""
    stem = f"{variant}_{split}{_tag(dtype)}"
    folder = metrics_dir() / "utility"
    return folder / f"{stem}.json", folder / f"{stem}.md"


def checked_path(variant: str, split: str, *, dtype: str | None = None) -> Path:
    """outputs/utility/checked/{variant}/utility_hi_{split}.json (``-{dtype}`` for another precision than
    ``generation.dtype``): the check a GPU line recorded of the variant's answer file (``record_checked``).
    A .json, so P04's scan for verdict files (every .jsonl under outputs/) passes it over."""
    return utility_dir() / CHECKED_DIR / variant / f"{SET}_{LANG}_{split}{_tag(dtype)}.json"


def frontier_cache_path(model: str) -> Path:
    """The utility judge's own call cache, outputs/judge_cache/frontier/utility__{model}.jsonl (the model
    named as lift.judge.frontier names its default cache), so it never loads the safety second judge's."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return output_dir() / "judge_cache" / "frontier" / f"utility__{safe}.jsonl"


def logical(path: Path) -> str:
    """``path`` as lift.sync names it, "outputs/...", when it is under the outputs dir (which LIFT_OUTPUT_DIR
    can move), so a metrics file names the same file on the laptop and on a VM."""
    try:
        return "outputs/" + Path(path).resolve().relative_to(output_dir()).as_posix()
    except ValueError:
        return str(path)


def source_of(pid: str) -> str:
    """"aya" or "ind", from a utility pid (utl-{src}-{i:05d})."""
    m = _PID.fullmatch(pid)
    if m is None:
        raise ValueError(f"{pid!r} is not a utility pid (utl-aya-NNNNN or utl-ind-NNNNN)")
    return m.group(1)


def outcome(score: float) -> str:
    """The candidate's outcome of one pair: above 0.5 a win, below a loss, 0.5 a tie (as P04 reads it)."""
    return "win" if score > 0.5 else "loss" if score < 0.5 else "tie"


def failure_kind(error: str | None) -> str | None:
    """What kind of failure a failed order's cause is ("refused", "unparseable judge output",
    "BadRequestError 400"): the cause up to its first colon, without the details after it. None for an order
    that was judged."""
    return None if error is None else error.split(":", 1)[0]


# ---------------------------------------------------------------------------------------- pairing --


@dataclass(frozen=True)
class Pair:
    pid: str
    instruction: str                 # the prompt text both sides answered
    candidate: GenerationRecord
    stock: GenerationRecord


@dataclass(frozen=True)
class Pairing:
    """Both sides' answers to every utility prompt of a split, checked for pairing (``pair_generations``)."""

    variant: str
    split: str
    dtype: str
    pairs: tuple[Pair, ...]          # one per prompt, by pid
    candidate_path: Path
    stock_path: Path


def _fields(exc: pydantic.ValidationError) -> str:
    """The fields a record failed on, never their values (which may hold an answer's text)."""
    return ", ".join(sorted({".".join(str(p) for p in e["loc"]) or "record" for e in exc.errors()}))


def _answers(name: str, candidate: str, split: str, texts: Mapping[str, str],
             dtype: str) -> list[GenerationRecord]:
    """One side's answers: one complete, current file in ``dtype``, every record naming its provenance."""
    path = generations_path(name, SET, LANG, split)
    generate_line = f"`lift utility --variant {candidate} --split {split}` (it generates what is missing)"
    if not path.is_file() or not path.stat().st_size:
        raise UtilityRefused(f"{logical(path)} is missing or empty: {name}'s answers come first. Run "
                             f"{generate_line} on a GPU runtime, or `lift sync pull` to fetch them")
    try:
        records = read_generations(path)
    except pydantic.ValidationError as exc:
        raise UtilityRefused(f"{logical(path)} holds a record that is not a GenerationRecord "
                             f"({_fields(exc)})") from None
    except ValueError as exc:
        raise UtilityRefused(f"{exc}: a last line cut by a kill is repaired when the cell is generated "
                             f"({generate_line}); a duplicate record means the file is corrupt") from None
    made = sorted({str(r.meta["dtype"]) for r in records if r.meta.get("dtype") is not None})
    restart = restart_command(name, SET, LANG, split, dtype=made[0] if len(made) == 1 else None)
    over = (f"start the cell over: run `{restart}` as a job line right after `lift sync pull`, then "
            f"{generate_line}")
    have = {r.pid for r in records}
    problems = []
    wrong = sorted(r.pid for r in records if (r.variant, r.set, r.lang, r.split) != (name, SET, LANG, split))
    if wrong:
        problems.append(f"{len(wrong)} records belong to another variant, set, language or split, e.g. "
                        f"{wrong[:5]}; {over}")
    missing = sorted(set(texts) - have)
    if missing:
        problems.append(f"{len(missing)} of the {len(texts)} {split} prompts have no answer, e.g. "
                        f"{missing[:5]}; every pair stays in the denominator, so finish the cell first: "
                        f"{generate_line}")
    extra = sorted(have - set(texts))
    if extra:
        problems.append(f"{len(extra)} answers are to pids that are not {split} utility prompts, e.g. "
                        f"{extra[:5]}; {over}")
    stale = sorted(r.pid for r in records if r.pid in texts and r.prompt_sha != sha256_text(texts[r.pid]))
    if stale:
        problems.append(f"{len(stale)} answers were made from another prompt text, e.g. {stale[:5]}; {over}")
    unnamed = {k: n for k in REQUIRED_META if (n := sum(r.meta.get(k) is None for r in records))}
    if unnamed:
        problems.append(f"records name no {sorted(unnamed)} ({unnamed}), so the pair cannot be checked: they "
                        f"were not made by lift.models.generate; {over}")
    configs = sorted({str(r.meta.get("run_config")) for r in records})
    stacks = sorted({record_stack(r.meta) for r in records})
    if len(configs) > 1 or len(stacks) > 1:
        problems.append(f"the file mixes {len(configs)} run configurations and {len(stacks)} stacks; one "
                        f"file is one configuration and one stack (merged from two machines?); {over}")
    if len(made) > 1:
        problems.append(f"its answers mix the precisions {made}; {over}")
    elif made and made != [dtype]:
        problems.append(f"its answers are in {made[0]}, and this run is {dtype}: every number of a run is "
                        f"one precision, so run with --dtype {made[0]} where {made[0]} runs")
    if problems:
        raise UtilityRefused(f"{logical(path)}: " + "; ".join(problems))
    return records


def _values(records: Sequence[GenerationRecord], key: str) -> list[str]:
    return sorted({str(r.meta.get(key)) for r in records})


def pair_generations(variant: str, split: str, *, dtype: str | None = None) -> Pairing:
    """Stock's and ``variant``'s answers to every utility prompt of ``split`` in ``dtype`` (default
    ``generation.dtype``), checked for pairing (module docstring). Raises UtilityRefused, naming the
    remedy, when the pair cannot be judged or reported (both sides' problems at once); ValueError when
    ``variant`` is stock."""
    if variant == REFERENCE:
        raise ValueError("the candidate cannot be stock: stock is the reference every pair is judged against")
    dtype = dtype or str(config.get("generation.dtype"))
    texts = {p.pid: p.text for p in prompt_files.load(SET, LANG, split)}
    if not texts:
        raise UtilityRefused(f"{SET}_{LANG} has no {split} prompts")
    for pid in texts:
        source_of(pid)
    sides: dict[str, list[GenerationRecord]] = {}
    refused = []
    for name in (variant, REFERENCE):
        try:
            sides[name] = _answers(name, variant, split, texts, dtype)
        except UtilityRefused as exc:
            refused.append(str(exc))
    if refused:
        raise UtilityRefused("; ".join(refused))
    what = f"{variant} vs {REFERENCE} on {SET}_{LANG}_{split}"
    differ = {k: (_values(sides[variant], k), _values(sides[REFERENCE], k)) for k in MATCH
              if _values(sides[variant], k) != _values(sides[REFERENCE], k)}
    if differ:
        lines = " or ".join(f"`{restart_command(name, SET, LANG, split, dtype=dtype)}`"
                            for name in (variant, REFERENCE))
        raise UtilityRefused(
            f"{what} cannot be paired: {differ} differ (a pair is judged only between answers made in one "
            "precision, from one model revision, under one harness, chat template and decoding settings). "
            f"Make the side that is out of date again: {lines} as a job line right after `lift sync pull`, "
            f"then `lift utility --variant {variant} --split {split}`")
    reported = str(config.get("generation.dtype"))
    if dtype != reported:
        problems = [p for name in (variant, REFERENCE)
                    for p in pipeline._licence_problems(dtype, sides[name], [], (name, SET, LANG, split),
                                                        "primary")]
        if problems:
            raise UtilityRefused(f"{what} in {dtype}: every reported number is {reported}, and another "
                                 "precision counts only under Gate 0's licence: "
                                 + "; ".join(dict.fromkeys(problems)))
    warn = {k: (_values(sides[variant], k), _values(sides[REFERENCE], k)) for k in MAY_DIFFER
            if _values(sides[variant], k) != _values(sides[REFERENCE], k)}
    if warn:
        log.warning("%s: paired, but the answers differ in %s", what, warn)
    cand, stock = ({r.pid: r for r in sides[name]} for name in (variant, REFERENCE))
    pairs = tuple(Pair(pid, texts[pid], cand[pid], stock[pid]) for pid in sorted(texts))
    return Pairing(variant=variant, split=split, dtype=dtype, pairs=pairs,
                   candidate_path=generations_path(variant, SET, LANG, split),
                   stock_path=generations_path(REFERENCE, SET, LANG, split))


# -------------------------------------------------------------------------------- checked answers --


@dataclass(frozen=True)
class Checked:
    """A GPU line's recorded check of one side's answer file (``record_checked``), as ``check_answers`` read
    it: the check's own file, that file's sha256, and its fields."""

    path: Path
    sha256: str
    doc: dict[str, Any]


def _short(digest: str | None) -> str:
    return digest[:12] if digest else "none"


def record_checked(variant: Any, split: str) -> Path:
    """Record that ``variant`` (a ``LoadedVariant``, built as the registry and the pin say now) checked its
    utility_hi_{split} answer file. Call it right after ``lift.models.generate.generate`` returned for that
    file: generate appends what is missing and refuses a file made under another configuration than the
    variant's (StaleOutputError), even one that answers every prompt. The record, ``checked_path`` in the
    variant's precision (written whole and atomically), names the file's sha256 and run configuration and
    what the variant was built from: its adapter hash, intervention digest (``lift.cachekey``: method,
    adapter, seed, params and artifacts), model revision and the harness version. ``check_answers`` compares
    them with the file, the registry, the pin and the harness on a machine that loads no model. A record
    that already says all of it is left as it is (its "created" time too), so a GPU line that changed
    nothing gives sync nothing to send. Raises ValueError for a file that is not this variant's answers in
    one configuration as built now (generate never leaves one)."""
    name, dtype = variant.spec.name, variant.dtype
    path = generations_path(name, SET, LANG, split)
    records = read_generations(path) if path.is_file() else []
    if not records:
        raise ValueError(f"{logical(path)} holds no answers to check")
    configs = _values(records, "run_config")
    found = {"variant": sorted({r.variant for r in records}), "dtype": _values(records, "dtype"),
             "model_rev": _values(records, "model_rev"), "harness": _values(records, "harness")}
    built = {"variant": [name], "dtype": [dtype], "model_rev": [str(variant.model_rev)],
             "harness": [str(harness.HARNESS_VERSION)]}
    wrong = {k: v for k, v in found.items() if v != built[k]}
    if len(configs) != 1 or wrong:
        raise ValueError(f"{logical(path)} is not {name}'s answers in one configuration as built now "
                         f"({len(configs)} run configurations; differs in {sorted(wrong)})")
    doc = {"kind": "utility_checked", "version": CHECK_VERSION, "variant": name, "set": SET, "lang": LANG,
           "split": split, "dtype": dtype, "path": logical(path), "sha256": sha256_file(path),
           "n": len(records), "run_config": configs[0], "adapter_hash": variant.adapter_hash,
           "intervention": intervention_digest(variant.spec), "model_rev": variant.model_rev,
           "harness": harness.HARNESS_VERSION, "created": datetime.now(UTC).isoformat(timespec="seconds")}
    out = checked_path(name, split, dtype=dtype)
    try:
        old = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = None

    def said(d: Any) -> Any:
        return {k: v for k, v in d.items() if k != "created"} if isinstance(d, dict) else None

    if said(old) != said(doc):
        write_json(out, doc)
    return out


def _pinned_revision() -> str | None:
    """The revision models.global's pin loads now, resolved as ``build_variant`` resolves it (a commit sha is
    itself; a model in a local directory, as in the tests, is its content hash), or None while the pin is not
    a commit sha: nothing then says which revision is current."""
    pin = str(config.get("models.global.revision"))
    if not _SHA.fullmatch(pin):
        return None
    from lift.models.load import resolved_revision

    return resolved_revision(str(config.get("models.global.id")), pin)


def _check_side(name: str, candidate: str, split: str, dtype: str, records: Sequence[GenerationRecord],
                path: Path, revision: str | None) -> Checked:
    """One side's answer file against its recorded check, its registry entry, the harness and the pin
    (``check_answers``). A check that is missing, unreadable or about another file is the GPU line's to
    make again. Answers the registry, the harness or the pin here has moved past are another
    configuration's, which only a restart of the cell makes again, unless this checkout is the one behind
    (the GPU job ran a newer ref): then updating it settles them."""
    from lift.models.variants import get_spec

    where, check = logical(path), checked_path(name, split, dtype=dtype)
    flag = "" if dtype == str(config.get("generation.dtype")) else f" --dtype {dtype}"
    gpu = (f"`lift utility --variant {candidate} --split {split} --generate-only{flag}` on a GPU runtime (it "
           "loads both sides and checks their answers against the registry; nothing is generated again), "
           "then `lift sync pull` here")
    restart = restart_command(name, SET, LANG, split, dtype=dtype)
    over = (f"start the cell over: run `{restart}` as a job line right after `lift sync pull`, then "
            f"`lift utility --variant {candidate} --split {split}{flag}`. If instead this checkout is behind "
            "the ref the GPU job ran, update it (git pull)")
    problems = []
    made = _values(records, "harness")
    if made != [str(harness.HARNESS_VERSION)]:
        problems.append(f"its answers were made under harness version {made}, and the harness here is "
                        f"version {harness.HARNESS_VERSION}: they are another configuration's, so {over}")
    revs = _values(records, "model_rev")
    if revision is not None and revs != [revision]:
        problems.append(f"its answers were made from model revision {revs}, and models.global is pinned to "
                        f"{revision} here: they are another configuration's, so {over}")
    data = check.read_bytes() if check.is_file() else b""
    doc: Any = None
    if data.strip():
        try:
            doc = json.loads(data.decode("utf-8"))
        except ValueError:
            doc = None
    absent = [k for k in CHECK_FIELDS if not isinstance(doc, dict) or k not in doc]
    if not data.strip():
        problems.append(f"no GPU line has checked its answers against {name} as registered now "
                        f"({logical(check)} is missing or empty): run {gpu}. If a GPU line refused them as "
                        f"stale (it withdraws their check), start the cell over first: `{restart}` as a job "
                        "line right after `lift sync pull`")
    elif absent:
        problems.append(f"{logical(check)} is not a check lift utility wrote (it names no {absent}): "
                        f"run {gpu}")
    elif (doc["variant"], doc["split"], doc["dtype"]) != (name, split, dtype):
        problems.append(f"{logical(check)} is a check of {doc['variant']}'s {doc['split']} answers in "
                        f"{doc['dtype']}, not of these: run {gpu}")
    elif doc["sha256"] != sha256_file(path) or [str(doc["run_config"])] != _values(records, "run_config"):
        problems.append(f"it changed since a GPU line checked it ({logical(check)} names another sha256 or "
                        f"run configuration): run {gpu}")
    else:
        try:
            spec = get_spec(name)
        except KeyError:
            spec = None
            problems.append(f"{name} is not in configs/variants.yaml here, so nothing says what its answers "
                            "must be made from: update this checkout (git pull) if the ref the GPU job ran "
                            "has its entry, or register it (an adapter with `lift register-adapter`) and "
                            "commit the entry. Its check is kept, and read again against that entry")
        moved = []
        if spec is not None and intervention_digest(spec) != doc["intervention"]:
            moved.append("intervention digest (method, adapter, seed, params, artifacts)")
        pinned = spec.artifacts.get("adapter") if spec is not None else None
        if pinned and pinned != doc["adapter_hash"]:
            moved.append(f"adapter hash ({_short(doc['adapter_hash'])} made, {_short(pinned)} pinned)")
        if moved:
            problems.append(f"{name}'s entry in configs/variants.yaml here is not the one its answers were "
                            f"made under: its {' and its '.join(moved)} differ (an adapter registered again "
                            f"with --replace, a refit artifact), so they are another variant's answers; "
                            f"{over}")
    if problems:
        raise UtilityRefused(f"{where}: " + "; ".join(problems))
    return Checked(path=check, sha256=sha256_file(check), doc=doc)


def check_answers(pairing: Pairing) -> dict[str, Checked]:
    """Both sides of ``pairing`` checked with no model loaded (module docstring): each answer file must have
    the check a GPU line recorded of it (``record_checked``) naming this very file (its sha256 and run
    configuration), and neither the variant's registry entry (its intervention digest, a pinned adapter
    hash), the harness version nor a pinned models.global revision may have moved since its answers were
    made. Returns each side's check by variant name; raises UtilityRefused naming each side's remedy (both
    sides' problems at once)."""
    revision = _pinned_revision()
    sides = ((pairing.variant, pairing.candidate_path, [p.candidate for p in pairing.pairs]),
             (REFERENCE, pairing.stock_path, [p.stock for p in pairing.pairs]))
    out: dict[str, Checked] = {}
    refused = []
    for name, path, records in sides:
        try:
            out[name] = _check_side(name, pairing.variant, pairing.split, pairing.dtype, records, path,
                                    revision)
        except UtilityRefused as exc:
            refused.append(str(exc))
    if refused:
        raise UtilityRefused("; ".join(refused))
    return out


# --------------------------------------------------------------------------------------- verdicts --


def pair_score(order_ab: str, order_ba: str) -> float:
    """The candidate's score of one pair: the mean of its score in order AB (shown as A) and in order BA
    (shown as B)."""
    return (SCORE_AB[order_ab] + SCORE_BA[order_ba]) / 2


def _order(result: Mapping[str, Any], where: str) -> tuple[str, str | None, str | None]:
    """(the winner stored, the reason, the error) of one order's judge result. A failure (winner None, the
    cause in error) is stored as "TIE"; a result that is neither a winner nor a failure is a broken judge."""
    winner, reason, error = result.get("winner"), result.get("reason"), result.get("error")
    if reason is not None and not isinstance(reason, str):
        raise ValueError(f"{where}: the judge's reason is a {type(reason).__name__}, not text")
    if error is not None:
        if winner is not None or not isinstance(error, str) or not error:
            raise ValueError(f"{where}: a failed judge result has winner None and its cause as text, got "
                             f"winner {winner!r} and a {type(error).__name__} error")
        return "TIE", None, error
    if winner not in WINNERS:
        raise ValueError(f"{where}: the judge gave neither a winner (A, B or TIE) nor an error "
                         f"(winner {winner!r})")
    return winner, reason, None


def make_verdict(pair: Pair, ab: Mapping[str, Any], ba: Mapping[str, Any], *, variant: str, split: str,
                 judge_id: str) -> UtilityVerdict:
    """One pair's verdict from its two judge results: ``ab`` with the candidate shown as A, ``ba`` as B."""
    order_ab, reason_ab, error_ab = _order(ab, f"{pair.pid} order AB")
    order_ba, reason_ba, error_ba = _order(ba, f"{pair.pid} order BA")
    return UtilityVerdict(
        pid=pair.pid, variant=variant, judge=judge_id, order_ab=order_ab, order_ba=order_ba,
        score=pair_score(order_ab, order_ba), split=split, lang=LANG, candidate_key=pair.candidate.key,
        stock_key=pair.stock.key, candidate_sha=sha256_text(pair.candidate.output),
        stock_sha=sha256_text(pair.stock.output), reason_ab=reason_ab, reason_ba=reason_ba, error_ab=error_ab,
        error_ba=error_ba)


def judge_pairs(pairing: Pairing, judge: Any, *, prefill: bool = False,
                wait: bool = True) -> tuple[list[UtilityVerdict], dict[str, Any] | None]:
    """Every pair judged in both orders by ``judge`` (``pairwise(instructions, a, b)``, ``judge_id``) in one
    call of 2N pairs: the first N in order AB, the next N in order BA. With ``prefill`` the same 2N go to
    ``judge.prefill_pairwise(..., wait=wait)`` first (the frontier judge's Message Batches pre-fill), whose
    summary is returned beside the verdicts (None without)."""
    pairs = pairing.pairs
    n = len(pairs)
    instructions = [p.instruction for p in pairs] * 2
    a = [p.candidate.output for p in pairs] + [p.stock.output for p in pairs]
    b = [p.stock.output for p in pairs] + [p.candidate.output for p in pairs]
    batch = judge.prefill_pairwise(instructions, a, b, wait=wait) if prefill else None
    results = judge.pairwise(instructions, a, b)
    if len(results) != 2 * n:
        raise RuntimeError(f"{judge.judge_id} returned {len(results)} results for {2 * n} judged orders")
    verdicts = [make_verdict(p, results[i], results[n + i], variant=pairing.variant, split=pairing.split,
                             judge_id=judge.judge_id) for i, p in enumerate(pairs)]
    return verdicts, batch


def write_verdicts(path: Path, verdicts: Sequence[UtilityVerdict]) -> None:
    """The whole verdict file, by pid, written atomically (lift.io.write_jsonl): one file is one judge."""
    judges = sorted({v.judge for v in verdicts})
    if len(judges) > 1:
        raise ValueError(f"verdicts from {len(judges)} judges for one file; one verdict file is one judge")
    write_jsonl(path, [v.model_dump() for v in sorted(verdicts, key=lambda v: v.pid)])


def read_verdicts(path: Path) -> list[UtilityVerdict]:
    """Every verdict of a verdict file, by pid; [] for an empty file (what ``--rejudge`` leaves). Raises
    FileNotFoundError for a missing file and ValueError for one that is not one judge's verdicts: a record
    that is not a UtilityVerdict, two judge ids, a pid twice, a score that is not the mean of its two
    orders, or a failed order not stored as "TIE"."""
    where = logical(path)
    verdicts = []
    for i, row in enumerate(read_jsonl(path), 1):
        try:
            verdicts.append(UtilityVerdict.model_validate(row))
        except pydantic.ValidationError as exc:
            raise ValueError(f"{where}: record {i} is not a UtilityVerdict ({_fields(exc)})") from None
    judges = sorted({v.judge for v in verdicts})
    if len(judges) > 1:
        raise ValueError(f"{where} mixes {len(judges)} judges ({', '.join(judges)}); one verdict file is "
                         "one judge")
    twice = sorted(p for p, n in Counter(v.pid for v in verdicts).items() if n > 1)
    if twice:
        raise ValueError(f"{where} holds {len(twice)} pids twice, e.g. {twice[:5]}")
    scored = sorted(v.pid for v in verdicts if v.score != pair_score(v.order_ab, v.order_ba))
    if scored:
        raise ValueError(f"{where}: {len(scored)} scores are not the mean of their two orders, e.g. "
                         f"{scored[:5]}")
    untied = sorted(v.pid for v in verdicts
                    if (v.error_ab is not None and v.order_ab != "TIE")
                    or (v.error_ba is not None and v.order_ba != "TIE"))
    if untied:
        raise ValueError(f"{where}: {len(untied)} failed orders are not stored as TIE, e.g. {untied[:5]}")
    return sorted(verdicts, key=lambda v: v.pid)


def _describes(v: UtilityVerdict, pair: Pair, pairing: Pairing) -> bool:
    """Whether ``v`` is a verdict on exactly the pair as it now stands (the same two answers)."""
    return (v.variant == pairing.variant and v.split == pairing.split and v.lang == LANG
            and v.candidate_key == pair.candidate.key and v.stock_key == pair.stock.key
            and v.candidate_sha == sha256_text(pair.candidate.output)
            and v.stock_sha == sha256_text(pair.stock.output))


def stale_pids(verdicts: Sequence[UtilityVerdict], pairing: Pairing) -> list[str]:
    """The pids whose verdict is missing or about other answers than the pair's now, then the pids with a
    verdict but no pair."""
    by_pid = {v.pid: v for v in verdicts}
    out = [p.pid for p in pairing.pairs if p.pid not in by_pid or not _describes(by_pid[p.pid], p, pairing)]
    return out + sorted(set(by_pid) - {p.pid for p in pairing.pairs})


def current_verdicts(path: Path, pairing: Pairing) -> list[UtilityVerdict]:
    """The verdicts of ``path`` in pairing order, every one describing its pair as it stands. Raises
    UtilityRefused when some pair has no such verdict (judge them with lift utility first), ValueError for a
    file that cannot be read and FileNotFoundError for a missing one."""
    verdicts = read_verdicts(path)
    stale = stale_pids(verdicts, pairing)
    if stale:
        line = f"lift utility --variant {pairing.variant} --split {pairing.split} --judge-only"
        raise UtilityRefused(f"{logical(path)}: {len(stale)} of {len(pairing.pairs)} pairs have no verdict "
                             f"on their answers as they now stand, e.g. {stale[:5]}; judge them first "
                             f"(`{line}`)")
    by_pid = {v.pid: v for v in verdicts}
    return [by_pid[p.pid] for p in pairing.pairs]


def reusable(path: Path, pairing: Pairing, judge_id: str) -> list[UtilityVerdict] | None:
    """The verdicts of ``path`` in pairing order when ``judge_id`` gave them and every one describes its pair
    as it stands; None when the file is missing, empty or out of date (judge again). Raises UtilityRefused for
    a file another judge wrote or one that cannot be read: ``--rejudge`` is the remedy for both."""
    if not path.is_file() or not path.read_bytes().strip():
        return None
    rejudge = ("add --rejudge once (a job line right after `lift sync pull`) to set the file aside and judge "
               "every pair again")
    try:
        verdicts = read_verdicts(path)
    except ValueError as exc:
        raise UtilityRefused(f"{exc}; {rejudge}") from None
    others = sorted({v.judge for v in verdicts} - {judge_id})
    if others:
        raise UtilityRefused(f"{logical(path)} holds verdicts by {others[0]}, and this run's judge is "
                             f"{judge_id}: one verdict file is one judge. Judge with the file's judge, or "
                             f"{rejudge} with this one")
    stale = stale_pids(verdicts, pairing)
    if stale:
        log.info("%s: %d of %d pairs have no current verdict (an answer made again, or a pid missing); "
                 "judging every pair again", logical(path), len(stale), len(pairing.pairs))
        return None
    by_pid = {v.pid: v for v in verdicts}
    return [by_pid[p.pid] for p in pairing.pairs]


def _set_aside(path: Path, aside: Path, *, suffix: str = "") -> SetAside:
    """Copy ``path`` to its place under ``aside`` (its name followed by ``suffix``) and leave it empty, as
    ``lift.models.generate.restart_cell`` does: the copy is written first, so a kill in between loses
    nothing. A file that is not here, or holds nothing, is left as it is."""
    if not path.is_file():
        return SetAside(path, None, None)
    data = path.read_bytes()
    if not data.strip():
        return SetAside(path, 0, None)
    copy = aside / path.resolve().relative_to(output_dir())
    copy = copy.with_name(copy.name + suffix)
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(data)
    path.write_bytes(b"")
    n = sum(1 for line in data.splitlines() if line.strip()) if path.suffix == ".jsonl" else 1
    return SetAside(path, n, copy)


def set_aside(path: Path) -> Path | None:
    """Copy a verdict file to outputs/.aside/<UTC stamp>/<its path under outputs>.aside and leave it empty
    (``--rejudge``). The copy is never synced, and its name does not end in .jsonl, so P04's verdict scan of
    outputs/ never reads it as current verdicts. Emptied, never removed: sync never deletes, so a removed file
    would come back from the runs repo, while an emptied one is sent as an update. Returns the copy, or None
    when the file is missing or already empty."""
    return _set_aside(path, aside_root(), suffix=ASIDE_SUFFIX).copy


def retire(variant: str, split: str, dtype: str | None = None, *, aside: Path | None = None,
           check: bool = True) -> list[SetAside]:
    """Retire what lift utility derived from ``variant``'s pair with stock on ``split`` (module docstring):
    both judge dirs' verdict files, the metrics .json and .md, the AuditKIT utility records (redacted and
    full) and, with ``check``, the variant's recorded answer check; ``dtype``'s files, or every precision's
    when ``dtype`` is None (a restart: the answers it sets aside are one file, whatever their precision).
    ``check`` is for ``lift restart`` of the cell, which sets aside the answers the check names. A refused
    pair passes False (``_retire_refused``; ``withdraw`` empties only the check of the side a GPU line
    refused): ``check_answers`` reads a kept check again on every run, so it never lets stale answers
    through, while only a GPU line could make it again, and a refusal this checkout settles with a git pull
    must not need one. Each file is copied under ``aside`` (default a new outputs/.aside/<UTC stamp>, never
    synced; a verdict file's copy named ``*.jsonl.aside``, so P04's scan of every .jsonl under outputs/
    never reads it as current verdicts) and left empty, never removed: sync never deletes, so a removed
    file would come back at the next pull, while an emptied one is a local edit the next push sends. An
    empty file is none to every reader (Gate 1, P04, ``reusable``, ``check_answers``). Returns what
    happened to each file (``SetAside``; records None for a file that is not here, no copy for an empty
    one)."""
    from lift.audit import TEXT_POLICIES, utility_record_path

    aside = aside or aside_root()
    files: dict[Path, str] = {}
    for d in ([dtype] if dtype else DTYPES):
        for judge_dir in JUDGE_DIRS:
            files[verdicts_path(judge_dir, variant, split, dtype=d)] = ASIDE_SUFFIX
        files.update(dict.fromkeys(metrics_paths(variant, split, dtype=d), ""))
        for judge_dir in JUDGE_DIRS:
            for text in TEXT_POLICIES:
                files[utility_record_path(judge_dir, variant, split, text=text, dtype=d)] = ""
        if check:
            files[checked_path(variant, split, dtype=d)] = ""
    return [_set_aside(path, aside, suffix=suffix) for path, suffix in files.items()]


def withdraw(side: str, candidate: str, split: str, dtype: str) -> list[SetAside]:
    """What a GPU line does when ``lift.models.generate.generate`` refuses ``side``'s answer file (stock's or
    ``candidate``'s) as stale: a StaleOutputError that is not an OtherPrecisionError, which the caller
    tells apart (a file made in another precision is not stale, and its own precision's check and files
    stand). ``side``'s recorded check in ``dtype`` is set aside and left empty, and so is what lift utility
    derived from ``candidate``'s pair with stock (``retire``, keeping ``candidate``'s check unless it is the
    side refused: a stock refusal says nothing of the candidate's answers). generate compares a file with
    more than a check names (decoding settings, the chat template, the cache-key version), so a check it
    contradicts must go, or ``--judge-only`` would still judge, and Gate 1 count, the answers it refused.
    The job's push carries the emptied files: every machine then reads ``side``'s answers as unchecked
    (stock's for every candidate) until the cell's ``lift restart`` line and a GPU line make them again.
    Logged by path and count; returns what happened to each file (``SetAside``)."""
    aside = aside_root()
    done = [_set_aside(checked_path(side, split, dtype=dtype), aside),
            *retire(candidate, split, dtype, aside=aside, check=False)]
    gone = [s for s in done if s.copy is not None]
    if gone:
        log.warning("%s vs %s, %s_%s_%s: %s's answers are refused as stale here, so its answer check and "
                    "what was derived from the pair, %d files, are set aside in %s and left empty: "
                    "--judge-only and Gate 1 refuse those answers until the cell is started over: %s",
                    candidate, REFERENCE, SET, LANG, split, side, len(gone), aside,
                    ", ".join(logical(s.path) for s in gone))
    return done


def _dirs(top: Path) -> list[Path]:
    return sorted(p for p in top.iterdir() if p.is_dir()) if top.is_dir() else []


def holders(split: str) -> list[str]:
    """Every candidate (any variant but stock) holding one of lift utility's files for ``split``, in any
    precision: a verdict file, metrics, an AuditKIT utility record or a recorded answer check. A restart of
    stock's utility cell retires all of theirs: each was paired with stock's answers."""
    tags = "|".join(DTYPES)
    cell = re.escape(f"{SET}_{LANG}_{split}") + rf"(?:-(?:{tags}))?"
    trees = [(utility_dir() / judge_dir, re.compile(cell + r"\.jsonl")) for judge_dir in JUDGE_DIRS]
    trees.append((utility_dir() / CHECKED_DIR, re.compile(cell + r"\.json")))
    trees += [(metrics_dir() / "auditkit_utility" / judge_dir, re.compile(cell + r"(?:\.full)?\.json"))
              for judge_dir in JUDGE_DIRS]
    found = {d.name for top, name in trees for d in _dirs(top)
             if any(p.is_file() and name.fullmatch(p.name) for p in d.iterdir())}
    metrics = re.compile(rf"(.+)_{re.escape(split)}(?:-(?:{tags}))?\.(?:json|md)")
    folder = metrics_dir() / "utility"
    if folder.is_dir():
        found.update(m.group(1) for p in folder.iterdir() if p.is_file() and (m := metrics.fullmatch(p.name)))
    return sorted(found - {REFERENCE})


# ---------------------------------------------------------------------------------------- metrics --


def _lid_results(texts: Sequence[str], lid: Any) -> list[tuple[str, bool]]:
    """(top GlotLID label, is_lang hi) of each text: ``lid.top(text, 1)`` and ``lid.is_lang(text, "hi")``."""
    out = []
    for text in texts:
        top = lid.top(text, 1)
        out.append((str(top[0][0]) if top else "", bool(lid.is_lang(text, LANG))))
    return out


def _fidelity(results: Sequence[tuple[str, bool]], accept: Collection[str]) -> dict[str, Any]:
    n = len(results)
    top_hi = sum(label in accept for label, _ in results)
    is_hi = sum(ok for _, ok in results)
    return {"top_hi": top_hi / n, "is_lang_hi": is_hi / n, "n": n, "n_top_hi": top_hi, "n_is_lang_hi": is_hi,
            "top_labels": dict(Counter(label for label, _ in results).most_common(5))}


def provenance(records: Sequence[GenerationRecord]) -> dict[str, list[str]]:
    """Each ``pipeline.PROVENANCE_META`` field of the records (dtype, run_config, model_rev, harness,
    template, decoding, torch, transformers, gpu) as its sorted distinct values, and the stack digests."""
    out = {name: sorted({str(r.meta[name]) for r in records if r.meta.get(name) is not None})
           for name in pipeline.PROVENANCE_META}
    out["stack"] = sorted({record_stack(r.meta) for r in records})
    return out


def _generation(records: Sequence[GenerationRecord], path: Path) -> dict[str, Any]:
    n = len(records)
    caps = sum(r.hit_cap for r in records)
    return {"variant": records[0].variant, **provenance(records), "n": n, "n_cap_hits": caps,
            "cap_hit_rate": caps / n, "mean_new_tokens": sum(r.n_new_tokens for r in records) / n,
            "path": logical(path), "file_sha256": sha256_file(path)}


def _failed(v: UtilityVerdict) -> tuple[bool, bool]:
    return v.error_ab is not None, v.error_ba is not None


def _bounded(verdicts: Sequence[UtilityVerdict], value: float) -> float:
    """The win rate with every failed order scored ``value`` (0: as a loss, 1: as a win) instead of 0.5."""
    total = 0.0
    for v in verdicts:
        ab, ba = _failed(v)
        total += ((value if ab else SCORE_AB[v.order_ab]) + (value if ba else SCORE_BA[v.order_ba])) / 2
    return total / len(verdicts)


def _position(verdicts: Sequence[UtilityVerdict]) -> dict[str, Any]:
    """Which position the judge picked over the orders it judged (failed ones left out), and how often a
    pair's two orders give the candidate the same outcome (both judged): a judge without position bias picks
    A in about half of its decisive verdicts."""
    judged = [winner for v in verdicts
              for winner, failed in zip((v.order_ab, v.order_ba), _failed(v), strict=True) if not failed]
    counts = Counter(judged)
    both = [v for v in verdicts if not any(_failed(v))]
    same = sum(OUTCOME_AB[v.order_ab] == OUTCOME_BA[v.order_ba] for v in both)
    decisive = counts["A"] + counts["B"]
    return {"n_orders": 2 * len(verdicts), "n_judged_orders": len(judged), "winner_A": counts["A"],
            "winner_B": counts["B"], "winner_TIE": counts["TIE"],
            "a_share_of_decisive": counts["A"] / decisive if decisive else None,
            "n_pairs_both_judged": len(both), "n_consistent": same, "n_split": len(both) - same,
            "consistency": same / len(both) if both else None}


def _scored(verdicts: Sequence[UtilityVerdict], pids: Sequence[str]) -> dict[str, Any]:
    """The win rate with its interval, and the failure counts and bounds, of one set of pairs."""
    wr = win_rate([v.score for v in verdicts], clusters=list(pids), n_boot=N_BOOT)
    failed = [_failed(v) for v in verdicts]
    return {"n": len(verdicts), "win_rate": wr["win_rate"], "lo": wr["lo"], "hi": wr["hi"],
            "p_vs_parity": wr["p_vs_parity"], "n_boot": N_BOOT,
            "failed_orders": sum(ab + ba for ab, ba in failed),
            "pairs_with_failure": sum(ab or ba for ab, ba in failed),
            "win_rate_failures_as_losses": _bounded(verdicts, 0.0),
            "win_rate_failures_as_wins": _bounded(verdicts, 1.0)}


def _check_summary(check: Checked) -> dict[str, Any]:
    """A side's recorded check as the metrics name it: its file and that file's sha256, and what it says the
    answers were made from."""
    named = ("run_config", "adapter_hash", "intervention", "model_rev", "harness")
    return {"path": logical(check.path), "sha256": check.sha256, **{k: check.doc[k] for k in named},
            "created": check.doc.get("created")}


def summarize(pairing: Pairing, verdicts: Sequence[UtilityVerdict], *, judge_id: str, judge_dir: str,
              verdicts_file: Path, lid: Any, checked: Mapping[str, Checked]) -> dict[str, Any]:
    """The utility metrics document (module docstring) of a pairing and its verdicts, in pairing order;
    ``checked`` is ``check_answers``' result for the pairing."""
    if [v.pid for v in verdicts] != [p.pid for p in pairing.pairs]:
        raise ValueError("the verdicts are not one per pair in pairing order")
    cand_check, stock_check = checked[pairing.variant], checked[REFERENCE]
    pids = [p.pid for p in pairing.pairs]
    accept = set(config.get("languages.accept_labels")[LANG])
    lids = {"candidate": _lid_results([p.candidate.output for p in pairing.pairs], lid),
            "stock": _lid_results([p.stock.output for p in pairing.pairs], lid)}
    errors = Counter(failure_kind(e) for v in verdicts for e in (v.error_ab, v.error_ba) if e is not None)
    by_source = {}
    for src in sorted({source_of(pid) for pid in pids}):
        idx = [i for i, pid in enumerate(pids) if source_of(pid) == src]
        fidelity = {side: _fidelity([res[i] for i in idx], accept) for side, res in lids.items()}
        by_source[src] = {"name": SOURCES[src], **_scored([verdicts[i] for i in idx], [pids[i] for i in idx]),
                          "fidelity": fidelity}
    return {
        "kind": "utility", "version": METRICS_VERSION, "variant": pairing.variant, "reference": REFERENCE,
        "split": pairing.split, "lang": LANG, "judge": judge_id, "judge_dir": judge_dir,
        "dtype": pairing.dtype,
        "candidate_adapter_hash": cand_check.doc["adapter_hash"],
        "candidate_intervention": cand_check.doc["intervention"],
        **_scored(verdicts, pids),
        "failure_reasons": dict(sorted(errors.items())),
        "outcomes": {k: sum(outcome(v.score) == k for v in verdicts) for k in ("win", "tie", "loss")},
        "scores": {str(s): sum(v.score == s for v in verdicts) for s in SCORES},
        "position": _position(verdicts),
        "fidelity": {side: _fidelity(res, accept) for side, res in lids.items()},
        "generation": {
            "candidate": {**_generation([p.candidate for p in pairing.pairs], pairing.candidate_path),
                          "checked": _check_summary(cand_check)},
            "stock": {**_generation([p.stock for p in pairing.pairs], pairing.stock_path),
                      "checked": _check_summary(stock_check)}},
        "by_source": by_source,
        "verdicts_path": logical(verdicts_file), "verdicts_sha256": sha256_file(verdicts_file),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _f(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def _both(doc: Mapping[str, Any], key: str) -> str:
    """"candidate / stock" of one fidelity or generation field."""
    return f"{_f(doc['candidate'][key])} / {_f(doc['stock'][key])}"


def render(doc: Mapping[str, Any]) -> str:
    """The metrics document as markdown: counts, rates and ids only."""
    fid, gen, pos = doc["fidelity"], doc["generation"], doc["position"]
    outcomes = " / ".join(str(doc["outcomes"][k]) for k in ("win", "tie", "loss"))
    scores = " / ".join(str(doc["scores"][str(s)]) for s in SCORES)
    lines = [
        f"# Utility: {doc['variant']} against stock, {doc['split']} split (Hindi)",
        "",
        f"Judge `{doc['judge']}` ({doc['judge_dir']}); answers in {doc['dtype']}. Made by `lift utility` on "
        f"{doc['created']} from `{doc['verdicts_path']}`.",
        "",
        "Both answer files were checked against configs/variants.yaml on a GPU runtime: the candidate's "
        f"adapter `{_short(doc['candidate_adapter_hash'])}`, intervention "
        f"`{_short(doc['candidate_intervention'])}`.",
        "",
        "Each pair is judged with the candidate shown as A, then as B; a win scores 1, a tie 0.5 and a loss "
        "0, and the pair's score is the mean of its two orders. A failed order counts as a tie.",
        "",
        "| measure | value |",
        "|---|---|",
        f"| win rate (n={doc['n']} pairs) | {_f(doc['win_rate'])} [{_f(doc['lo'])}, {_f(doc['hi'])}], "
        f"p vs parity {doc['p_vs_parity']:.3g} |",
        f"| failed orders | {doc['failed_orders']} of {2 * doc['n']} ({doc['pairs_with_failure']} pairs) |",
        f"| win rate, failures as losses / as wins | {_f(doc['win_rate_failures_as_losses'])} / "
        f"{_f(doc['win_rate_failures_as_wins'])} |",
        f"| pairs won / tied / lost | {outcomes} |",
        f"| pair scores 1 / 0.75 / 0.5 / 0.25 / 0 | {scores} |",
        f"| position: winner A / B / TIE over judged orders | {pos['winner_A']} / {pos['winner_B']} / "
        f"{pos['winner_TIE']} |",
        f"| position: pairs whose two orders agree | {pos['n_consistent']} of {pos['n_pairs_both_judged']} |",
        f"| Hindi top label: candidate / stock | {_both(fid, 'top_hi')} |",
        f"| is_lang hi: candidate / stock | {_both(fid, 'is_lang_hi')} |",
        f"| cap-hit rate: candidate / stock | {_both(gen, 'cap_hit_rate')} |",
        "",
        "By source:",
        "",
        "| source | pairs | win rate | failed orders | Hindi top label, candidate / stock |",
        "|---|---|---|---|---|",
    ]
    for src, s in doc["by_source"].items():
        lines.append(f"| {src}: {s['name']} | {s['n']} | {_f(s['win_rate'])} [{_f(s['lo'])}, "
                     f"{_f(s['hi'])}] | {s['failed_orders']} | {_both(s['fidelity'], 'top_hi')} |")
    if doc["failure_reasons"]:
        causes = ", ".join(f"{k} {n}" for k, n in doc["failure_reasons"].items())
        lines += ["", f"Failed orders by cause: {causes}."]
    lines.append("")
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    """UTF-8, LF and atomic (temp file + rename), like lift.io.write_jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(path: Path, doc: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n")


# ----------------------------------------------------------------------------------------- judges --


def make_judge(judge_dir: str, *, device: str = "cuda:0", dtype: str | None = None) -> Any:
    """The utility judge of ``judge_dir``: "frontier" is ``FrontierJudge.from_config("utility")`` with its own
    call cache (``frontier_cache_path``); "fallback" is the local Qwen3-8B (lift.judge.local_pairwise) on
    ``device``, computing in ``dtype`` (default ``generation.dtype``). Judge modules are imported here."""
    if _judge_dir(judge_dir) == "frontier":
        from lift.judge.frontier import FrontierJudge

        model = str(config.get("judges.utility.model"))
        return FrontierJudge.from_config("utility", cache_path=str(frontier_cache_path(model)))
    from lift.judge.local_pairwise import LocalPairwiseJudge

    return LocalPairwiseJudge.from_config(device=device, dtype=dtype or str(config.get("generation.dtype")))


def make_lid() -> Any:
    """GlotLID (lift.lang.LangID, about 1.6 GB in RAM), loaded once for both sides' answers."""
    from lift.lang import LangID

    return LangID()


def _stats(judge: Any) -> dict[str, Any] | None:
    stats = getattr(judge, "stats", None)
    return dict(stats) if stats is not None else None


def _judge_into(path: Path, pairing: Pairing, judge: Any, *, judge_dir: str, prefill: bool,
                wait: bool) -> tuple[list[UtilityVerdict], dict[str, Any] | None]:
    """Judge every pair, write the verdict file whole, and record the run in a manifest (its cost and the
    judge's counters; a local judge's ``info`` and ``stack``), also when the run stops: "pending" for batches
    still running, else "failed"."""
    from lift.manifest import finish_run, start_run

    pending: tuple[type[BaseException], ...] = ()
    if judge_dir == "frontier":
        # With or without the prefill: a synchronous pass also stops with FrontierBatchPending while an
        # earlier batch that holds its requests still runs (it collects such a batch rather than pay twice).
        from lift.judge import frontier

        pending = (frontier.FrontierBatchPending,)
    n = len(pairing.pairs)
    run = start_run("utility", variant=pairing.variant, reference=REFERENCE, set=SET, lang=LANG,
                    split=pairing.split, dtype=pairing.dtype, judge=judge.judge_id, judge_dir=judge_dir,
                    n_pairs=n, n_requests=2 * n, out_path=str(path), prefill=prefill, wait=wait,
                    device=getattr(judge, "device", None), devices=0 if judge_dir == "frontier" else 1,
                    judge_info=getattr(judge, "info", None), judge_stack=getattr(judge, "stack", None))
    log.info("%s vs %s, %s_%s_%s: judging %d pairs in both orders (%d requests) with %s", pairing.variant,
             REFERENCE, SET, LANG, pairing.split, n, 2 * n, judge.judge_id)
    try:
        verdicts, batch = judge_pairs(pairing, judge, prefill=prefill, wait=wait)
        write_verdicts(path, verdicts)
    except pending as exc:
        finish_run(run, status="pending", error=type(exc).__name__, judge_stats=_stats(judge))
        raise
    except BaseException as exc:
        finish_run(run, status="failed", error=type(exc).__name__, judge_stats=_stats(judge))
        raise
    failed = sum(ab + ba for ab, ba in map(_failed, verdicts))
    finish_run(run, status="ok", n_verdicts=len(verdicts), failed_orders=failed, prefill_summary=batch,
               judge_stats=_stats(judge))
    log.info("%s: wrote %d verdicts (%d failed orders of %d)", logical(path), len(verdicts), failed, 2 * n)
    return verdicts, batch


# --------------------------------------------------------------------------------------- the run --


def _retire_refused(variant: str, split: str, dtype: str) -> None:
    """``retire`` a refused pair's derived files in ``dtype``, logging what went: paths and counts only. Both
    sides' answer checks are kept (``retire``'s ``check``): the refusal may be this checkout's own, which a
    git pull settles with no GPU line, and ``check_answers`` reads every kept check again."""
    aside = aside_root()
    done = [s for s in retire(variant, split, dtype, aside=aside, check=False) if s.copy is not None]
    if done:
        log.warning("%s vs %s, %s_%s_%s: the pair is refused, so the %d files derived from it are set aside "
                    "in %s and left empty, and no reader takes them for current (the answer checks are kept, "
                    "and checked again on every run): %s", variant, REFERENCE, SET, LANG, split, len(done),
                    aside, ", ".join(logical(s.path) for s in done))


@dataclass(frozen=True)
class Report:
    """What ``evaluate`` did: the metrics document and the files it wrote."""

    metrics: dict[str, Any]
    verdicts_path: Path
    metrics_json: Path
    metrics_md: Path
    record_path: Path | None         # the AuditKIT record; None when it was refused
    record_error: str | None         # why it was refused
    judged: bool                     # False: a current verdict file was reused, with no judge call
    prefill: dict[str, Any] | None   # the frontier judge's Message Batches summary, when it ran
    set_aside: Path | None           # where --rejudge copied the previous verdict file


def evaluate(variant: str, split: str, *, judge_dir: str = "frontier", dtype: str | None = None,
             device: str = "cuda:0", prefill: bool = True, wait: bool = True, rejudge: bool = False,
             judge: Any = None, lid: Any = None) -> Report:
    """Pair, check (``check_answers``), judge (or reuse a current verdict file), report and record ``variant``
    against stock on the utility prompts of ``split`` (module docstring). ``judge`` and ``lid`` default to
    ``make_judge`` and ``make_lid`` (tests pass fakes). ``prefill`` applies to the frontier judge only. A pair
    refused at pairing or checking has its derived files in ``dtype`` (verdicts, metrics, AuditKIT records)
    retired before the UtilityRefused propagates, so nothing reads them as current, and both answer checks
    kept (``_retire_refused``): only ``lift restart`` of the cell retires a check, or a GPU line that refuses
    its answers (``withdraw``). A refused verdict file (another judge's, or unreadable) retires nothing, its
    remedy being --rejudge. A refused AuditKIT record (ValueError) is returned in ``record_error``, the
    metrics being written; the frontier judge's stops and FrontierBatchPending propagate."""
    from lift import audit

    _judge_dir(judge_dir)
    dtype = dtype or str(config.get("generation.dtype"))
    try:
        pairing = pair_generations(variant, split, dtype=dtype)
        checked = check_answers(pairing)
    except UtilityRefused:
        _retire_refused(variant, split, dtype)
        raise
    path = verdicts_path(judge_dir, variant, split, dtype=dtype)
    judge = judge if judge is not None else make_judge(judge_dir, device=device, dtype=dtype)
    aside = set_aside(path) if rejudge else None
    if aside is not None:
        log.info("%s set aside to %s and left empty (--rejudge)", logical(path), aside)
    verdicts = reusable(path, pairing, judge.judge_id)
    judged, batch = verdicts is None, None
    if verdicts is None:
        verdicts, batch = _judge_into(path, pairing, judge, judge_dir=judge_dir,
                                      prefill=prefill and judge_dir == "frontier", wait=wait)
    else:
        log.info("%s: every pair has a current verdict by %s; reused with no judge call", logical(path),
                 judge.judge_id)
    lid = lid if lid is not None else make_lid()
    doc = summarize(pairing, verdicts, judge_id=judge.judge_id, judge_dir=judge_dir, verdicts_file=path,
                    lid=lid, checked=checked)
    json_path, md_path = metrics_paths(variant, split, dtype=dtype)
    write_json(json_path, doc)
    _write_text(md_path, render(doc))
    record, error = None, None
    try:
        record = audit.record_utility(variant, split, judge_dir=judge_dir, dtype=dtype)
    except ValueError as exc:
        error = str(exc)
        log.error("%s vs %s, %s_%s_%s: no AuditKIT record: %s", variant, REFERENCE, SET, LANG, split, exc)
    return Report(metrics=doc, verdicts_path=path, metrics_json=json_path, metrics_md=md_path,
                  record_path=record, record_error=error, judged=judged, prefill=batch, set_aside=aside)
