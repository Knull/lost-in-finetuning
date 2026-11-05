"""``lift smoke|generate|judge|run-dev|restart``: Gate 0 and the generate -> judge harness (GPU runtimes).

Torch and transformers are imported inside the handlers, so ``lift --help`` works without them, and
only after the checks that need neither (pinned revisions, names), so a laptop gets their messages.
Devices: the generator takes ``--device`` (cuda:0). The judge takes ``--judge-device``; "auto" means
cuda:1 when there are two GPUs (Kaggle T4 x2: two separate 16 GB devices) and cuda:0 otherwise
(Colab A100: both models share the card, about 7 GB for the generator and 9 GB for the judge in bf16).

A cell lift generate, judge or run-dev cannot go on with (a StaleOutputError) is logged and skipped, and
the other cells go on. At the end each kind is listed apart with its remedy: stale (begun under another
stack, or made from other inputs): the ``lift restart`` line that starts it over, a job line right after
``lift sync pull``, with the lines that job needs under it; made in another precision than the run's
(whatever refusal came first): its own ``--dtype``, or leaving it out, never a restart (that would
discard it); another judge's judgements (MixedJudgesError): the file's judge, or the role's ``lift
restart --judge`` line. Exit codes: 0 done; 3 done, but some cells were
skipped or their AuditKIT record was refused (a GPU job goes on with its next lines and fails at the
end); 1 an error. A precision other than ``generation.dtype`` is refused on a GPU with native bf16
(compute capability 8 or more): other precisions exist only for the T4's licence (decisions.md 27).

lift judge with a judge that has a bulk API (``prefill``: the frontier judge's Message Batches) first
plans every cell (``lift.judge.pipeline.pending_items``, which writes nothing) and hands it all the
items they will send in one prefill, which only fills the judge's call cache; then the cell loop runs
as without it, finding the replies cached, so the judgements are the ones the one-at-a-time path gives.
A cell its planning refuses is left to the cell loop, which skips and lists it, or stops on it, as it
would anyway. The prefill writes its own run manifest (stage "judge_prefill", devices 0, its counts,
batch ids and spend): the cell loop's judge manifests count only what they send themselves.
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lift.prompts import SAFETY_SETS
from lift.schemas import LANGS

log = logging.getLogger("lift.run")

ALL_SETS = ("harmful", "easy_benign", "hard_benign", "redteam", "utility")
JUDGE_GB = 9.0            # Qwen3Guard-Gen-4B in bf16/fp16 (8.8 GB) with a little headroom
JUDGE_KV_HEADROOM_GB = 4.0


def _logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _names(values: Sequence[str] | None, allowed: Sequence[str], default: Sequence[str]) -> list[str]:
    """``--langs hi en`` or ``--langs hi,en``; checked against ``allowed``."""
    out = [v for item in (values or default) for v in str(item).split(",") if v]
    bad = sorted(set(out) - set(allowed))
    if bad:
        raise SystemExit(f"unknown {bad}; choose from {list(allowed)}")
    return list(dict.fromkeys(out))


def _judge_device(value: str) -> str:
    if value != "auto":
        return value
    import torch

    return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"


def _dtype(value: str | None) -> str:
    from lift import config

    return value or config.get("generation.dtype")


def _native_bf16(device: str) -> str | None:
    """The compute capability of ``device`` when it runs bf16 natively (8.0 or more: an A100, L4 or
    H100), else None (a T4, or a CPU)."""
    if not str(device).startswith("cuda"):
        return None
    from lift.models.generate import stack

    capability = str(stack(device)["gpu_capability"])
    major = capability.split(".")[0]
    return capability if major.isdigit() and int(major) >= 8 else None


def _refuse_other_dtype(dtype: str, *devices: str | None) -> None:
    """Refuse a precision other than ``generation.dtype`` on a GPU with native bf16 (decisions.md 27):
    another precision is read for reporting only under the T4's licence, which compared fp16 made on
    the T4, so an fp16 line reused on an A100 would make cells no Gate 0 compared. A CPU (the tests'
    fp32) is never refused."""
    from lift import config

    default = config.get("generation.dtype")
    if dtype == default:
        return
    for device in dict.fromkeys(d for d in devices if d):
        capability = _native_bf16(device)
        if capability:
            raise SystemExit(
                f"--dtype {dtype} on {device} (compute capability {capability}) is refused: this GPU runs "
                f"{default} natively, and every reported number is {default}. Another precision exists "
                "only for the T4's licence (Gate 0's fp16 check, decisions.md 27), which compared fp16 made "
                "on the T4: run this line on the T4, or here without --dtype.")


def require_pinned(*keys: str) -> None:
    """Refuse to produce study outputs from an unpinned revision ("main" moves: stock and an adapter
    could sit on different commits). Gate 0 resolves and prints the shas to pin. The GPU job reads the
    configs/project.yaml of the ref it runs, so a pin counts there only once it is committed and pushed."""
    import re

    from lift import config

    loose = [f"{k}.revision = {config.get(k + '.revision')!r}" for k in keys
             if not re.fullmatch(r"[0-9a-f]{40}", str(config.get(k + ".revision")))]
    if loose:
        raise SystemExit("unpinned revisions in configs/project.yaml: " + "; ".join(loose) + ". Pin each to "
                         "the commit sha Gate 0 printed (outputs/metrics/gate0.json, 'revisions') with a "
                         "dated docs/decisions.md entry, then commit the pin and push it to the ref the GPU "
                         "job runs (lift remote --ref, Kaggle BRANCH), before generating or judging: the "
                         "job reads that ref's configs/project.yaml, not this working tree.")


def _complete(variant: str, set_name: str, lang: str, split: str) -> bool:
    """True if every prompt of the split has a generation (the AuditKIT record needs all of them)."""
    from lift.models.generate import generations_path, read_generations
    from lift.prompts import load

    path = generations_path(variant, set_name, lang, split)
    if not path.exists():
        return False
    return {g.pid for g in read_generations(path)} >= {p.pid for p in load(set_name, lang, split)}


def _record(variant: str, set_name: str, lang: str, split: str, *, dtype: str,
            refused: dict[str, str]) -> None:
    """Write the AuditKIT record of a judged cell. A record the checked loader refuses (another
    precision, a stale judgement) is logged and kept in ``refused`` with its error rather than stopping
    the run: the judgements are already written, and the command exits 3 at the end."""
    from lift.audit import record_safety
    from lift.models.generate import StaleOutputError

    if not _complete(variant, set_name, lang, split):
        log.info("%s %s_%s_%s: generations do not cover every prompt yet; no AuditKIT record",
                 variant, set_name, lang, split)
        return
    try:
        log.info("AuditKIT record: %s", record_safety(variant, set_name, lang, split, dtype=dtype))
    except (ValueError, StaleOutputError) as exc:
        log.error("%s %s_%s_%s: no AuditKIT record: %s", variant, set_name, lang, split, exc)
        refused[f"{set_name}_{lang}_{split}"] = str(exc)


# Each kind of skipped cell (contract I), and the head of its end-of-run listing.
SKIPPED = {
    "stale": ("were skipped as stale (see the errors above). To start one over, run its line once as a job "
              "line right after `lift sync pull`, followed by the lines under it where they say and the line "
              "that failed: a `lift judge` line needs its cell generated first, as the restart empties the "
              "generations too, and the restart also sets aside other judges' judgements, which only their "
              "own line fills again"),
    "precision": ("were made in another precision and left as they are (see the errors above): a restart "
                  "would discard them"),
    "judges": ("hold another judge's judgements and were left as they are (see the errors above): judges are "
               "never mixed in one file. Judge each with the file's judge (its id names its precision and "
               "Controversial mode), or judge it again with its line, run once as a job line right after "
               "`lift sync pull`, followed by the line that failed"),
}
_SKIPPED_AS = {"stale": "stale", "precision": "made in another precision",
               "judges": "judged by another judge"}
REDRAW = "after P04 draws the key list again (lift validate second-judge-sample, then lift sync push)"


def _kind(exc: Exception) -> str:
    from lift.judge.pipeline import MixedJudgesError
    from lift.models.generate import OtherPrecisionError

    if isinstance(exc, OtherPrecisionError):
        return "precision"
    return "judges" if isinstance(exc, MixedJudgesError) else "stale"


def _skip(skipped: dict[str, dict[str, str]], variant: str, set_name: str, lang: str, split: str,
          exc: Exception, *, judged: str | None, dtype: str, generates: bool = True,
          keys: bool = False) -> None:
    """Log a cell the command cannot go on with and keep its remedy under its kind (``SKIPPED``), so the
    command goes on with the other cells and lists it at the end. A cell made in another precision than
    the run's ``dtype`` is listed as such whatever refusal came first (an API judge has no precision for
    judge_file to check): a restart line run here would discard it. A stale cell's whole-cell restart
    gets the lines its job needs under it (``_after_restart``)."""
    from lift.models.generate import made_in

    found = _kind(exc)
    if found == "precision":
        made = exc.dtype
    else:
        made = "/".join(sorted(made_in(variant, set_name, lang, split) - {dtype}))
    kind = "precision" if made else found
    cell = f"{set_name}_{lang}_{split}"
    if kind == found:
        log.error("%s %s: %s, skipped: %s", variant, cell, _SKIPPED_AS[kind], exc)
    else:                                      # its error's remedy is for a run in the cell's precision
        log.error("%s %s: made in another precision, skipped: its generations were made in %s, and this run "
                  "is %s. It was also found %s, which a run in %s settles: a restart from here would discard "
                  "its %s outputs", variant, cell, made, dtype, _SKIPPED_AS[found], made, made)
    command = getattr(exc, "command", None)
    if kind == "precision":
        remedy = (f"made in {made}: run it with --dtype {made} where {made} runs, or leave it out with "
                  "--sets/--langs")
    elif command is None:
        remedy = "no lift restart line covers it (see its error above)"
    elif " --judge " in command:
        remedy = command
    else:
        remedy = command + _after_restart(variant, set_name, lang, split, judged=judged, dtype=dtype,
                                          generates=generates, keys=keys,
                                          redraw=getattr(exc, "keys_changed", True))
    skipped.setdefault(kind, {})[cell] = remedy


def _after_restart(variant: str, set_name: str, lang: str, split: str, *, judged: str | None, dtype: str,
                   generates: bool, keys: bool, redraw: bool) -> str:
    """The lines a whole-cell restart's job needs, each under the restart line saying when it runs. The
    restart empties the generations too, so a line that does not generate (``generates`` False: lift
    judge) needs lift generate first; and it sets aside every judge's judgements, while the line that
    failed refills only ``judged``'s: the other roles' lines refill theirs. When the cause may have
    changed the cell's generation keys (``redraw``: anything but a stack move), a line reading P04's key
    list (a refill, or the line that failed: ``keys``) runs only after P04 draws it again."""
    from lift.models.generate import cell_judgements

    lines = []
    if not generates:
        generate = _generate_line(variant, set_name, lang, split, dtype=dtype)
        lines.append(f"first generate it again: {generate}")
    if keys and redraw:
        lines.append(f"then the line that failed, {REDRAW}")
    held = [p for p in cell_judgements(variant, set_name, lang, split) if p.stat().st_size]
    for _path, line in _refills(variant, set_name, lang, split, held, judged=judged, dtype=dtype):
        when = f", {REDRAW}" if redraw and " --keys " in line else ""
        lines.append(f"and once it is generated again{when}: {line}")
    return "".join(f"\n    {line}" for line in lines)


def _listing(skipped: Mapping[str, Mapping[str, str]]) -> list[str]:
    """One message per kind of skipped cell (in ``SKIPPED``'s order), each cell with its remedy."""
    return [f"{len(cells)} cells {SKIPPED[kind]}:\n" + "".join(f"  {cell}: {remedy}\n"
                                                                for cell, remedy in cells.items())
            for kind in SKIPPED if (cells := skipped.get(kind))]


def _refused_exit(refused: Mapping[str, str], skipped: Mapping[str, Mapping[str, str]] | None = None) -> int:
    """3 when an AuditKIT record was refused or a cell was skipped ("done, but listed": a GPU job goes on
    with its next lines), else 0. Each is listed with its remedy."""
    if refused:
        log.error("%d AuditKIT records were refused; the judgements are written, and each record needs what "
                  "its error says:\n%s", len(refused),
                  "".join(f"  {cell}: {why}\n" for cell, why in refused.items()))
    for message in _listing(skipped or {}):
        log.error("%s", message)
    return 3 if refused or any((skipped or {}).values()) else 0


# --- lift smoke ---------------------------------------------------------------------------------------


def _smoke(args: argparse.Namespace) -> int:
    _logging()
    import torch

    if not torch.cuda.is_available():
        print("lift smoke is Gate 0 and needs a CUDA GPU", file=sys.stderr)
        return 2
    from lift.gate0 import CHECKS, check_status, pin_hint, report_name, run_gate0, verdict, write_report
    from lift.manifest import finish_run, start_run

    skip = _names(args.skip, CHECKS, ()) if args.skip else []
    sizes = [int(s) for s in _names(args.batch_sizes, [str(n) for n in range(1, 513)], ())]
    judge_device = _judge_device(args.judge_device)
    devices = len({args.device, judge_device})
    run = start_run("smoke", device=args.device, judge_device=judge_device, compare_dtype=args.compare_dtype,
                    skip=skip, devices=devices)
    try:
        result = run_gate0(device=args.device, judge_device=judge_device, compare_dtype=args.compare_dtype,
                           skip=skip, n_precision=args.n_precision, n_dry=args.n_dry, batch_sizes=sizes)
    except ValueError as exc:            # a refused configuration (e.g. --compare-dtype on an A100)
        finish_run(run, status="refused", error=str(exc))
        print(f"lift smoke: {exc}", file=sys.stderr)
        return 2
    except BaseException as exc:
        finish_run(run, status="failed", error=type(exc).__name__)
        raise
    json_path, md_path = write_report(result, name=report_name(result))
    finish_run(run, status="ok", passed=result["pass"], incomplete=result["incomplete"],
               check_seconds={k: c.get("seconds") for k, c in result["checks"].items()})
    for check, c in result["checks"].items():
        why = c.get("error") or c.get("reason") or c.get("failure")
        print(f"{check_status(c)}  {check}" + (f"  ({why})" if why else ""))
    print(f"\n{verdict(result)}: {result['decision']}")
    print(f"\n{pin_hint(result)}\n\nwrote {json_path} and {md_path}")
    return 0 if result["pass"] else 1


# --- lift generate ------------------------------------------------------------------------------------


def _generate(args: argparse.Namespace) -> int:
    _logging()
    require_pinned("models.global")
    sets = _names(args.sets, ALL_SETS, SAFETY_SETS)
    langs = _names(args.langs, LANGS, LANGS)
    dtype = _dtype(args.dtype)
    _refuse_other_dtype(dtype, args.device)
    from lift.models.generate import StaleOutputError, generate, generations_path
    from lift.models.variants import build_variant, get_spec
    from lift.prompts import load

    spec = get_spec(args.variant)
    variant = build_variant(spec, device=args.device, dtype=dtype)
    skipped: dict[str, dict[str, str]] = {}
    for set_name in sets:
        for lang in langs:
            prompts = load(set_name, lang, args.split)
            if args.limit:
                prompts = prompts[: args.limit]
            if not prompts:
                log.warning("%s_%s has no %s prompts", set_name, lang, args.split)
                continue
            try:
                generate(variant, prompts, out_path=generations_path(spec.name, set_name, lang, args.split),
                         batch_size=args.batch_size)
            except StaleOutputError as exc:
                _skip(skipped, spec.name, set_name, lang, args.split, exc, judged=None, dtype=dtype)
    return _refused_exit({}, skipped)


# --- lift judge ---------------------------------------------------------------------------------------


def _judge(args: argparse.Namespace) -> int:
    _logging()
    from lift.judge.pipeline import judge_dir, judge_file, make_judge, read_keys_file

    if (args.sample is not None or args.keys) and args.judge != "second":
        raise SystemExit("--sample and --keys select a subset, which only the second judge may judge; "
                         "the primary judge covers every generation")
    if args.sample is not None and args.keys:
        raise SystemExit("pass either --sample or --keys, not both")
    sets = _names(args.sets, ALL_SETS, SAFETY_SETS)
    langs = _names(args.langs, LANGS, LANGS)
    local = judge_dir(args.judge) == "qwen3guard"
    if local:
        require_pinned(_judge_entry(args.judge))
    from lift.models.generate import StaleOutputError, generations_path

    keys = read_keys_file(Path(args.keys)) if args.keys else None
    if keys is None:
        cells = [(s, lang) for s in sets for lang in langs]
    else:
        cells = _key_cells(args.variant, args.split, keys, sets=sets if args.sets else None,
                           langs=langs if args.langs else None, allow_missing=args.allow_missing)
    paths = {(s, lang): generations_path(args.variant, s, lang, args.split) for s, lang in cells}
    _generated_first(paths.values())
    dtype = _dtype(args.dtype)
    device = _judge_device(args.device) if local else None
    if local:                                  # an API judge has no precision and runs on no GPU of ours
        _refuse_other_dtype(dtype, device)
    judge = make_judge(args.judge, device=device, dtype=dtype if local else None)
    if callable(getattr(judge, "prefill", None)):
        _prefill(judge, paths, variant=args.variant, split=args.split, sample=args.sample, keys=keys)
    refused: dict[str, str] = {}
    skipped: dict[str, dict[str, str]] = {}
    for (set_name, lang), path in paths.items():
        try:
            judge_file(path, judge=judge, sample=args.sample, keys=keys)
        except StaleOutputError as exc:
            _skip(skipped, args.variant, set_name, lang, args.split, exc, judged=args.judge, dtype=dtype,
                  generates=False, keys=keys is not None)
            continue
        if args.judge == "primary":
            _record(args.variant, set_name, lang, args.split, dtype=dtype, refused=refused)
    return _refused_exit(refused, skipped)


# What a bulk prefill's summary holds (FrontierJudge.prefill_safety): counts, batch ids and spend, no text.
PREFILL_FIELDS = ("mode", "n_requests", "n_cached", "n_submitted", "n_collected", "n_errored", "n_expired",
                  "n_canceled", "batch_ids", "usd")


def _prefill(judge: Any, paths: Mapping[tuple[str, str], Path], *, variant: str, split: str,
             sample: float | None, keys: Collection[str] | None) -> None:
    """Hand ``judge.prefill`` every item the cells of ``paths`` will send (``pending_items``, under the cell
    loop's selection) in one call, before the cell loop judges them: a judge with a bulk API fills its call
    cache, and the loop's calls find the replies there. A cell whose planning raises what judge_file would
    raise for it (a StaleOutputError, ValueError or FileNotFoundError) is left out, logged by name: the loop
    meets the same error and skips and lists the cell, or stops on it, as it would without the prefill.
    Nothing pending, no call. The call writes its own manifest (stage "judge_prefill", devices 0: an API
    judge runs on no GPU of ours) with the summary and the judge's cost counters, also when it fails; the
    summary is logged, counts, ids and spend only."""
    from lift.judge.pipeline import pending_items, stats_delta
    from lift.manifest import finish_run, start_run
    from lift.models.generate import StaleOutputError

    items: list[Any] = []
    planned: dict[str, int] = {}
    left_out: list[str] = []
    for (set_name, lang), path in paths.items():
        cell = f"{set_name}_{lang}_{split}"
        try:
            found = pending_items(path, judge, sample=sample, keys=keys)
        except (StaleOutputError, ValueError, FileNotFoundError) as exc:
            left_out.append(cell)
            log.info("%s %s: left out of the bulk prefill (%s); the cell loop meets the same error", variant,
                     cell, type(exc).__name__)
            continue
        planned[cell] = len(found)
        items.extend(found)
    if not items:
        log.info("%s %s: nothing left to judge in the %d cells planned, so no bulk prefill", variant, split,
                 len(planned))
        return
    counters = getattr(judge, "stats", None)
    before = dict(counters) if counters is not None else None
    run = start_run("judge_prefill", variant=variant, split=split, judge=judge.judge_id,
                    judge_dir=judge.short, cells=planned, cells_left_out=left_out, n_items=len(items),
                    devices=0)
    log.info("%s %s: %d items pending in %d cells go to %s's bulk prefill first", variant, split, len(items),
             sum(1 for n in planned.values() if n), judge.judge_id)
    try:
        result = judge.prefill(items)
        if not isinstance(result, Mapping):
            raise TypeError(f"{judge.judge_id}'s prefill returned a {type(result).__name__}, not its summary")
    except BaseException as exc:
        finish_run(run, status="failed", error=type(exc).__name__,
                   judge_stats=stats_delta(before, getattr(judge, "stats", None)))
        raise
    summary = {field: result[field] for field in PREFILL_FIELDS if field in result}
    finish_run(run, status="ok", prefill=summary,
               judge_stats=stats_delta(before, getattr(judge, "stats", None)))
    log.info("%s %s: bulk prefill %s; the cells are judged next, each request the cache does not hold sent "
             "on its own", variant, split, summary)


def _generated_first(paths: Iterable[Path]) -> None:
    """Refuse a lift judge run over generations files that are missing or empty: lift generate comes
    first (lift restart leaves a cell's files empty)."""
    missing = [str(p) for p in paths if not p.is_file() or not p.stat().st_size]
    if missing:
        raise SystemExit(f"{len(missing)} generations files are missing or empty, e.g. {missing[:3]}; "
                         "run lift generate first (lift restart leaves a cell's files empty)")


def _key_cells(variant: str, split: str, keys: Sequence[str], *, sets: Collection[str] | None,
               langs: Collection[str] | None, allow_missing: bool) -> list[tuple[str, str]]:
    """The (set, lang) generations files a --keys run judges: every file of ``variant``'s ``split`` that
    holds a listed key, within --sets/--langs when they were given. A listed key in no file at all may be
    in a file within them that is empty (lift restart leaves a cell so) or, named by an explicit --sets
    and --langs, missing: then the command is refused with lift generate's message, which --allow-missing
    never skips, as that says nothing about the list. Otherwise such a key was regenerated under a new key
    (a fixed prompt) or is a typo, and stops the command unless ``allow_missing``: its item would silently
    lose its label. Keys in files outside an explicit --sets/--langs are expected (P04 prints one line per
    group over one list) and are logged by "{set}_{lang}" of the file, counts only."""
    from lift.models.generate import generations_path, read_generations
    from lift.paths import generations_dir

    files: dict[tuple[str, str], Path] = {}
    for path in sorted((generations_dir() / variant).glob(f"*_{split}.jsonl")):
        set_name, lang, _ = path.stem.rsplit("_", 2)
        files[(set_name, lang)] = path
    wanted = set(keys)
    holder: dict[str, tuple[str, str]] = {}
    for cell, path in files.items():
        holder.update({g.key: cell for g in read_generations(path) if g.key in wanted})
    unknown = sorted(wanted - set(holder))
    if unknown:                                 # they may be the keys of a cell not generated (again) yet
        within = [p for (s, lang), p in files.items()
                  if (sets is None or s in sets) and (langs is None or lang in langs)]
        named = [generations_path(variant, s, lang, split) for s in sets or () for lang in langs or ()]
        _generated_first(dict.fromkeys([*within, *named]))
    if unknown and not allow_missing:
        raise SystemExit(f"{len(unknown)} listed keys are in none of {variant}'s {split} generations files, "
                         f"e.g. {[k[:12] for k in unknown[:5]]}; regenerate the key list "
                         "(lift validate second-judge-sample) or pass --allow-missing")
    if unknown:
        log.warning("%d listed keys are in no %s %s generations file", len(unknown), variant, split)
    held = Counter(holder.values())
    chosen = [c for c in held if (sets is None or c[0] in sets) and (langs is None or c[1] in langs)]
    outside = {f"{s}_{lang}": n for (s, lang), n in sorted(held.items()) if (s, lang) not in chosen}
    if outside:
        log.warning("%d listed keys are in %s %s generations files outside --sets/--langs; this run does "
                    "not judge them (count by set_lang): %s", sum(outside.values()), variant, split, outside)
    return sorted(chosen, key=lambda c: (_rank(c[0], ALL_SETS), _rank(c[1], LANGS)))


def _rank(name: str, order: Sequence[str]) -> tuple[int, str]:
    return (order.index(name) if name in order else len(order), name)


def _judge_entry(role: str) -> str:
    return {"primary": "judges.safety_primary", "second": "judges.safety_second"}[role]


def _p04_list(variant: str, split: str) -> Path:
    """P04's second-judge key list of a variant and split (decisions.md 31)."""
    from lift.paths import validation_dir

    return validation_dir() / "keys" / "second_judge" / f"{variant}_{split}.txt"


def _p04_keys(variant: str, split: str) -> str:
    """``_p04_list`` as a job line names it: relative to the repo, where every job line runs, unless
    LIFT_VALIDATION_DIR moved validation/."""
    import shlex

    from lift.paths import REPO_ROOT

    path = _p04_list(variant, split)
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return shlex.quote(str(path))


def _dtype_flag(dtype: str) -> str:
    """A job line's ``--dtype``, named when it is not ``generation.dtype``."""
    from lift import config

    return f" --dtype {dtype}" if dtype != config.get("generation.dtype") else ""


def _generate_line(variant: str, set_name: str, lang: str, split: str, *, dtype: str) -> str:
    """The job line that generates one cell again (in ``dtype``)."""
    line = f"lift generate --variant {variant} --split {split} --sets {set_name} --langs {lang}"
    return line + _dtype_flag(dtype)


def _judge_line(variant: str, set_name: str, lang: str, split: str, *, role: str, dtype: str) -> str:
    """The job line that judges one cell again with ``role``'s judge (in ``dtype``): the second judge's
    over P04's key list, or with decisions.md 31's --sample 0.10 while P04 has drawn none (the second
    judge ran with --sample then, and a line naming a missing list would only fail)."""
    line = f"lift judge --variant {variant} --split {split} --sets {set_name} --langs {lang} --judge {role}"
    if role == "second":
        listed = _p04_list(variant, split).is_file()
        line += f" --keys {_p04_keys(variant, split)}" if listed else " --sample 0.10"
    return line + _dtype_flag(dtype)


def _refills(variant: str, set_name: str, lang: str, split: str, paths: Sequence[Path], *,
             judged: str | None, dtype: str) -> list[tuple[Path, str]]:
    """(file, line) for each judgements file in ``paths`` of a role other than ``judged`` (the role the
    line that failed judges with; None for lift generate): the job line that fills it again once the
    cell is generated again. A whole-cell restart sets every judge's file aside, while the line that
    failed refills only its own, and run-dev never runs the second judge."""
    from lift.judge.pipeline import ROLES, judge_dir

    roles: dict[str, str] = {}
    for role in ROLES:
        try:
            roles.setdefault(judge_dir(role), role)
        except (KeyError, ValueError):           # an entry that names no judge kind
            continue
    out = []
    for path in paths:
        role = roles.get(path.parent.parent.name)
        if role != judged:
            out.append((path, _judge_line(variant, set_name, lang, split, role=role, dtype=dtype) if role
                        else "no configured judge writes it now"))
    return out


# --- lift run-dev -------------------------------------------------------------------------------------


def run_cells(holder: dict[str, Any], make_judge: Callable[[], Any], cells: Sequence[tuple[str, str]], *,
              split: str = "dev", concurrent: bool, interleave: bool = True, limit: int | None = None,
              batch_size: int | None = None, lid: Any = None,
              record: Callable[[str, str], None] | None = None,
              skipped: dict[str, dict[str, str]] | None = None) -> list[Path]:
    """Generate then judge every (set, lang) cell with the primary judge (run-dev's); returns the
    judgements paths.

    ``holder["variant"]`` is the only reference to the generator, so it can really be freed.
    ``concurrent``: the judge runs in its own thread on its own GPU and judges each file as soon as
    it is written. Otherwise both share one device: with ``interleave`` each file is judged right
    after it is generated (both models loaded); without, every file is generated first, the
    generator is dropped and its memory released, then the judge is loaded and judges them all.
    A cell whose generations or judgements file is refused (StaleOutputError: stale, made in another
    precision, judged by another judge) is logged and skipped, and the other cells go on: a cut-short
    session must not block the rest. Its remedy goes into ``skipped[kind]`` (``SKIPPED``) under
    "{set}_{lang}_{split}"; without ``skipped``, one StaleOutputError listing them is raised once every
    other cell is done.
    """
    from lift.judge.pipeline import judge_file
    from lift.models.generate import StaleOutputError, generate, generations_path
    from lift.prompts import load

    name, dtype = holder["variant"].spec.name, holder["variant"].dtype
    done: list[Path] = []
    left: dict[str, dict[str, str]] = {} if skipped is None else skipped

    def finished() -> list[Path]:
        if skipped is None and any(left.values()):
            raise StaleOutputError("".join(_listing(left)))
        return done

    def skip(set_name: str, lang: str, exc: Exception) -> None:
        _skip(left, name, set_name, lang, split, exc, judged="primary", dtype=dtype)

    def gen(set_name: str, lang: str) -> Path | None:
        prompts = load(set_name, lang, split)[:limit] if limit else load(set_name, lang, split)
        if not prompts:
            log.warning("%s_%s has no %s prompts", set_name, lang, split)
            return None
        out_path = generations_path(name, set_name, lang, split)
        try:
            return generate(holder["variant"], prompts, out_path=out_path, batch_size=batch_size)
        except StaleOutputError as exc:
            skip(set_name, lang, exc)
            return None

    def jud(judge: Any, path: Path, set_name: str, lang: str) -> None:
        try:
            done.append(judge_file(path, judge=judge, lid=lid))
        except StaleOutputError as exc:
            skip(set_name, lang, exc)
            return
        if record is not None:
            record(set_name, lang)

    if concurrent:
        work: queue.Queue = queue.Queue()
        failure: list[BaseException] = []

        def worker() -> None:
            try:
                judge = make_judge()
                while (item := work.get()) is not None:
                    jud(judge, *item)
            except BaseException as exc:  # logged here, re-raised (or chained) by the main thread
                log.exception("the judge thread failed")
                failure.append(exc)

        thread = threading.Thread(target=worker, name="lift-judge", daemon=True)
        thread.start()
        try:
            for set_name, lang in cells:
                if failure:
                    break
                path = gen(set_name, lang)
                if path is not None:
                    work.put((path, set_name, lang))
        except BaseException as exc:
            work.put(None)
            thread.join()
            if failure:                    # both sides failed: neither cause may be lost
                raise exc from failure[0]
            raise
        work.put(None)
        thread.join()
        if failure:
            raise failure[0]
        return finished()

    if interleave:
        judge = make_judge()
        for set_name, lang in cells:
            path = gen(set_name, lang)
            if path is not None:
                jud(judge, path, set_name, lang)
        return finished()

    written = [(gen(s, lang), s, lang) for s, lang in cells]
    holder.clear()
    _release_memory()
    judge = make_judge()
    for path, set_name, lang in written:
        if path is not None:
            jud(judge, path, set_name, lang)
    return finished()


def _release_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _on_cuda(device: str) -> bool:
    if not device.startswith("cuda"):
        return False
    import torch

    return torch.cuda.is_available()


def _free_gb(device: str) -> float:
    import torch

    free, _total = torch.cuda.mem_get_info(torch.device(device))
    return free / 1e9


def _run_dev(args: argparse.Namespace) -> int:
    _logging()
    from lift.judge.pipeline import judge_dir, make_judge

    # As in _judge, only a local judge has a revision to pin: once the judge decision makes the frontier
    # judge primary, the judges.safety_primary block names a provider and model instead.
    local = judge_dir("primary") == "qwen3guard"
    require_pinned("models.global", *([_judge_entry("primary")] if local else []))
    langs = _names(args.langs, LANGS, LANGS)
    sets = _names(args.sets, SAFETY_SETS, SAFETY_SETS)
    dtype = _dtype(args.dtype)
    gen_device, judge_device = args.device, _judge_device(args.judge_device)
    _refuse_other_dtype(dtype, gen_device, judge_device if local else None)
    from lift.models.variants import build_variant, get_spec

    spec = get_spec(args.variant)
    # The holder is the only reference to the generator, so the one-GPU path can free it.
    holder = {"variant": build_variant(spec, device=gen_device, dtype=dtype)}
    concurrent = gen_device != judge_device
    interleave = True
    if not concurrent and _on_cuda(gen_device):
        free = _free_gb(gen_device)
        interleave = free >= JUDGE_GB + JUDGE_KV_HEADROOM_GB
        log.info("one GPU: %.1f GB free after loading the generator; %s", free,
                 "judge loaded alongside" if interleave else "judging after all generation, generator freed")

    cells = [(s, lang) for s in sets for lang in langs]
    refused: dict[str, str] = {}
    skipped: dict[str, dict[str, str]] = {}

    def record(set_name: str, lang: str) -> None:
        _record(args.variant, set_name, lang, "dev", dtype=dtype, refused=refused)

    run_cells(holder, lambda: make_judge("primary", device=judge_device, dtype=dtype), cells,
              concurrent=concurrent, interleave=interleave, limit=args.limit, batch_size=args.batch_size,
              record=None if args.limit else record, skipped=skipped)
    return _refused_exit(refused, skipped)


# --- lift restart -------------------------------------------------------------------------------------


RESTART_HELP = """\
Start cells over that lift generate, judge or run-dev skipped as stale (begun under another stack, or
made from other inputs; the error names the exact line). For each --sets x --langs cell, the
generations file, every judgements file and every AuditKIT record of the cell (with --judge, only that
judge's judgements file and records) is copied to outputs/.aside/<UTC stamp>/, which is never synced,
and left EMPTY: the next lift sync push replaces the runs repo's copies (an empty record is retired),
and run-dev (or generate, then judge) fills the cell again. The log names the line that refills any
other judge's judgements it set aside, a judge run-dev never runs: the second judge's, over P04's key
list (drawn again first when the cell's generation keys changed: a fixed prompt text or another
configuration), or with --sample 0.10 while P04 has drawn none.

A utility cell (--sets utility --langs hi) holds lift utility's answers. Its whole-cell restart also
retires what lift utility derived from them, in every precision, copied aside and left empty in the same
way: both judge dirs' verdict files (their copies named *.jsonl.aside, which P04's verdict scan passes
over), the metrics, the AuditKIT utility records and the answer check (outputs/utility/checked/), of the
variant and, for stock, of every candidate that holds any (each was paired with stock's answers), when
the cell's answer file is here (one that is not was not started over). The log names the line that fills
them again for each: lift utility --variant <candidate> --split <split>.

--dtype is the precision this job makes the cells in (default: generation.dtype; the T4 passes
--dtype fp16, as its restart lines say). A cell made in another precision is refused, and nothing is
set aside: a restart would discard its outputs, which this precision cannot make again. --force
starts it over anyway.

Run it once, as a job line right after `lift sync pull` and before the line that failed, then leave
it out of later jobs (each run would start the cell over again). Moving a file aside by hand does not
work: sync never deletes, so the next pull brings the runs repo's copy back. A file that is not here
is logged and skipped.

example job (lift remote --job, or the Kaggle runner's JOB):
  lift sync pull
  lift restart --variant stock --split dev --sets harmful --langs hi
  lift run-dev --variant stock
"""


def _other_precision_refused(variant: str, split: str, cells: Sequence[tuple[str, str]], dtype: str, *,
                             force: bool) -> bool:
    """True (each such cell logged; the caller sets nothing aside) when a cell was made in another
    precision than ``dtype`` and not ``force``: a restart would discard its outputs, and the job (in
    ``dtype``) cannot make them again. With ``force`` such cells are only logged as discarded."""
    from lift.models.generate import made_in

    other = {c: "/".join(sorted(made)) for c in cells if (made := made_in(variant, *c, split) - {dtype})}
    for (set_name, lang), was in other.items():
        cell = f"{variant} {set_name}_{lang}_{split}"
        if force:
            log.warning("%s: made in %s; --force: its %s outputs are discarded (set aside in "
                        "outputs/.aside/, which is never synced)", cell, was, was)
        else:
            log.error("%s: made in %s, not %s (--dtype): a restart would discard its %s outputs, which a run "
                      "in %s cannot make again. Nothing was set aside. Restart it where %s runs, with "
                      "--dtype %s; leave it out of --sets/--langs; or add --force to discard them.",
                      cell, was, dtype, was, dtype, was, was)
    return bool(other) and not force


UTILITY_CELL = ("utility", "hi")       # lift.utility's SET and LANG: the answers lift utility judges


def _retire_utility(variant: str, split: str, *, aside: Path) -> tuple[int, list[str]]:
    """Retire what lift utility derived from the answers a restart of ``variant``'s utility cell sets aside
    (``lift.utility.retire``, every precision): the variant's own files (a candidate's verdicts, metrics,
    AuditKIT utility records and answer check; stock's answer check) and, for stock, every candidate's that
    holds any (``lift.utility.holders``), each having been paired with stock's answers. Logged by path;
    returns how many files were retired and the candidates whose ``lift utility`` line fills them again."""
    from lift import utility

    candidates = utility.holders(split) if variant == utility.REFERENCE else [variant]
    n = 0
    for name in dict.fromkeys([variant, *candidates]):
        for path, _records, copy in utility.retire(name, split, aside=aside):
            if copy is not None:
                n += 1
                log.info("%s: lift utility's file retired: copied to %s and left empty", path, aside)
    return n, candidates


def _restart(args: argparse.Namespace) -> int:
    _logging()
    from lift.models.generate import aside_root, generations_path, restart_cell
    from lift.models.variants import get_spec
    from lift.paths import judgements_dir

    try:
        get_spec(args.variant)
    except KeyError as exc:
        raise SystemExit(exc.args[0]) from None
    sets = _names(args.sets, ALL_SETS, ())
    langs = _names(args.langs, LANGS, ())
    dtype = _dtype(args.dtype)
    judge = None
    if args.judge:
        from lift.judge.pipeline import judge_dir

        judge = judge_dir(args.judge)
    cells = [(s, lang) for s in sets for lang in langs]
    if _other_precision_refused(args.variant, args.split, cells, dtype, force=args.force):
        return 1
    aside = aside_root()
    emptied, retired, retired_utility = 0, 0, 0
    refills: list[tuple[Path, str]] = []
    utility_refills: list[str] | None = None        # set when a utility cell was started over whole
    for set_name, lang in cells:
        done = restart_cell(args.variant, set_name, lang, args.split, aside=aside, judge_dir=judge)
        for path, n, copy in done:
            if n is None:
                log.warning("%s: not here, skipped (lift restart runs right after lift sync pull: a file "
                            "only the runs repo holds is left as it is there)", path)
            elif copy is None:
                log.info("%s: already empty", path)
            elif path.suffix == ".json":
                retired += 1
                log.info("%s: AuditKIT record retired: copied to %s and left empty", path, aside)
            else:
                emptied += 1
                log.info("%s: %d record(s) set aside in %s; the file is left empty", path, n, aside)
        set_aside = [p for p, _n, copy in done if copy is not None and p.is_relative_to(judgements_dir())]
        refills += _refills(args.variant, set_name, lang, args.split, set_aside,
                            judged=args.judge or "primary", dtype=dtype)
        answers = generations_path(args.variant, set_name, lang, args.split)
        if judge is None and (set_name, lang) == UTILITY_CELL and answers.is_file():   # here: started over
            n_files, utility_refills = _retire_utility(args.variant, args.split, aside=aside)
            retired_utility += n_files
    if not emptied + retired + retired_utility:
        return 0
    precision = f", with --dtype {dtype}" if dtype != _dtype(None) else ""
    fills = []
    if args.judge:
        fills.append(f"the line that failed judges the cell again (lift judge --judge {args.judge}, or lift "
                     f"run-dev for the primary judge{precision})")
    elif any(cell != UTILITY_CELL for cell in cells):
        fills.append("the line that failed (lift run-dev, or lift generate then lift judge"
                     f"{precision}) fills the generations, the primary judge's judgements and its AuditKIT "
                     "record again")
    if utility_refills is not None:
        tail = f" --split {args.split}{_dtype_flag(dtype)}`"
        lines = [f"`lift utility --variant {c}{tail}" for c in utility_refills]
        lines = lines or [f"`lift utility --variant <candidate>{tail} (any candidate's line makes stock's "
                          "answers again)"]
        fills.append(f"the utility cell's answers, verdicts, metrics and AuditKIT utility records are filled "
                     f"again by {', '.join(lines)}, one line per candidate (the line that failed among them)")
    log.info("%d file(s) emptied, %d AuditKIT record(s) and %d of lift utility's files retired: the next "
             "lift sync push replaces the runs repo's copies (an empty record is retired there too), and %s",
             emptied, retired, retired_utility, "; and ".join(fills))
    if refills:
        redraw = ("; when its generation keys changed (a fixed prompt text or another configuration, as the "
                  f"listing that named this restart says), run a --keys line only {REDRAW}")
        keyed = any(" --keys " in line for _path, line in refills)
        log.warning("%d judgements file(s) of another judge were set aside, which that line never fills "
                    "again. Once the cell is generated again, add each one's line to the job%s:\n%s",
                    len(refills), redraw if keyed else "",
                    "".join(f"  {path}: {line}\n" for path, line in refills))
    return 0


# --- registration -------------------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("smoke", help="Gate 0: precision, throughput, dry run, preamble, merge parity (GPU)")
    p.add_argument("--device", default="cuda:0", help="generator device (default cuda:0)")
    p.add_argument("--judge-device", default="auto",
                   help="judge device; auto = cuda:1 if two GPUs, else cuda:0")
    p.add_argument("--compare-dtype", choices=["fp16"], default=None,
                   help="run in fp16 (a T4) and compare with the stored bf16 run from the runs repo")
    p.add_argument("--skip", nargs="+", default=None, metavar="CHECK",
                   help="checks to skip: preamble precision fp16 throughput dry_run parity")
    p.add_argument("--n-precision", type=int, default=200, help="prompts in the precision check (200)")
    p.add_argument("--n-dry", type=int, default=20, help="prompts per language and set in the dry run (20)")
    p.add_argument("--batch-sizes", nargs="+", default=["8", "16", "32"], help="throughput batch sizes")
    p.set_defaults(func=_smoke)

    p = sub.add_parser("generate", help="generate one variant's outputs for prompt sets and languages")
    p.add_argument("--variant", required=True, help="a name from configs/variants.yaml")
    p.add_argument("--sets", nargs="+", default=None, help=f"default: {' '.join(SAFETY_SETS)}")
    p.add_argument("--langs", nargs="+", default=None, help="default: all six")
    p.add_argument("--split", required=True, choices=["dev", "final"])
    p.add_argument("--limit", type=int, default=None, help="first N prompts per set and language (by pid)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None,
                   help="default: generation.dtype")
    p.add_argument("--batch-size", type=int, default=None, help="default: chosen from free memory")
    p.set_defaults(func=_generate)

    p = sub.add_parser("judge", help="judge one variant's generations (every one, or a second-judge subset)")
    p.add_argument("--variant", required=True)
    p.add_argument("--sets", nargs="+", default=None, help=f"default: {' '.join(SAFETY_SETS)}")
    p.add_argument("--langs", nargs="+", default=None, help="default: all six")
    p.add_argument("--split", required=True, choices=["dev", "final"])
    p.add_argument("--judge", choices=["primary", "second"], default="primary")
    p.add_argument("--sample", type=float, default=None,
                   help="second judge: a seeded random fraction (e.g. 0.10) plus every validated item")
    p.add_argument("--keys", default=None,
                   help="second judge: judge exactly the generation keys listed in this file, in every "
                        "generations file that holds one (within --sets/--langs when given)")
    p.add_argument("--allow-missing", action="store_true",
                   help="with --keys: go on when listed keys are in none of the variant's generations files")
    p.add_argument("--device", default="auto", help="judge device; auto = cuda:1 if two GPUs, else cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None,
                   help="default: generation.dtype")
    p.set_defaults(func=_judge)

    p = sub.add_parser("run-dev", help="generate, then judge, every dev safety set in all six languages")
    p.add_argument("--variant", required=True)
    p.add_argument("--langs", nargs="+", default=None, help="default: all six")
    p.add_argument("--sets", nargs="+", default=None, help=f"default: {' '.join(SAFETY_SETS)}")
    p.add_argument("--limit", type=int, default=None, help="first N prompts per set and language (no record)")
    p.add_argument("--device", default="cuda:0", help="generator device")
    p.add_argument("--judge-device", default="auto", help="auto = cuda:1 if two GPUs, else cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default=None, help="default: generation.dtype")
    p.add_argument("--batch-size", type=int, default=None)
    p.set_defaults(func=_run_dev)

    p = sub.add_parser("restart", help="start stale cells over: set their files aside, leave them empty "
                                       "(a job line right after lift sync pull)",
                       description=RESTART_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, help="a name from configs/variants.yaml")
    p.add_argument("--split", required=True, choices=["dev", "final"])
    p.add_argument("--sets", nargs="+", required=True, help=f"one or more of: {' '.join(ALL_SETS)}")
    p.add_argument("--langs", nargs="+", required=True, help=f"one or more of: {' '.join(LANGS)}")
    p.add_argument("--judge", choices=["primary", "second"], default=None,
                   help="only this judge's judgements file and records (default: the generations file, "
                        "every judgements file and every AuditKIT record of each cell)")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None,
                   help="the precision this job makes the cells in (default: generation.dtype); a cell made "
                        "in another one is refused")
    p.add_argument("--force", action="store_true",
                   help="also start over cells made in another precision than --dtype (discards them)")
    p.set_defaults(func=_restart)
