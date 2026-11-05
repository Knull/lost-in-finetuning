"""``lift utility``: the Hindi utility win rate of a variant against stock (P03); lift.utility has the rules.

It follows lift.commands.run's conventions and shares its helpers. Every run but ``--judge-only`` checks the
pin, then loads stock and then the candidate (one model at a time, torch imported after the pin check) and
passes each side's answer file through ``lift.models.generate.generate``, also when the file answers every
prompt: generate appends nothing to a current file and refuses one made under another configuration than
the variant as registered now (another adapter, registry entry, model revision, template, decoding or
harness), so a replaced adapter's answers are never judged as the new one's (rule 4). A side it refuses (a
StaleOutputError: stale, or made in another precision) is listed with its remedy, and the pair is not
judged; a side it accepts gets its check recorded (``lift.utility.record_checked``). Only ``--judge-only``,
which loads no model, works on a laptop with no GPU stack (and needs no pin): it judges answers only as a
GPU line checked them, and refuses a side whose registry entry, harness or pinned revision has moved since
(``lift.utility.check_answers``); such a refusal keeps the checks, so a git pull settles one this checkout
is behind on, with no GPU line. Exit codes: 0 done; 3 done, but something was listed with its remedy (a
side skipped, the pair or the verdict file refused, Message Batches still running, the AuditKIT record
refused), so a GPU job goes on with its next lines; 1 an error; 2 a refused combination of options.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import Any

from lift.commands.run import (
    _dtype,
    _logging,
    _refuse_other_dtype,
    _refused_exit,
    _release_memory,
    _skip,
    require_pinned,
)

log = logging.getLogger(__name__)

HELP = """\
The Hindi utility win rate of a variant against stock (Gate 1's gain check, the selection rule's utility
condition). Both sides answer every utility_hi prompt of the split through the frozen harness (stock
first, then the candidate, one model loaded at a time). A side whose file already answers every prompt is
still loaded and its file checked against the variant as registered now, and nothing is generated again:
a file made under another adapter, registry entry, model revision, template, decoding or harness is
listed with its lift restart line, and the pair is not judged. Each check is recorded in
outputs/utility/checked/. Each pair is then judged in both orders, the candidate shown as A and then as B,
scored 1 / 0.5 / 0 per order for a win / tie / loss and averaged: the win rate is the mean pair score,
with a bootstrap interval. A failed order counts as a tie; the failures, and the win rate with them
counted as losses and as wins, are reported beside it.

--judge-only loads no model (the laptop, after lift sync pull): it judges the answers only as a GPU line
checked them, and refuses a side with no check, one that changed since, or one whose registry entry,
harness or pinned model revision has moved since (each listed with the line that settles it).

Writes outputs/utility/{frontier|fallback}/{variant}/utility_hi_{split}.jsonl (the verdicts, rewritten
whole), outputs/metrics/utility/{variant}_{split}.json and .md, and the AuditKIT record under
outputs/metrics/auditkit_utility/. A verdict file that already describes every pair as it stands is reused
with no judge call; one another judge wrote is refused until --rejudge sets it aside. When the pair is
refused, its verdicts, metrics and AuditKIT record are set aside (outputs/.aside/, never synced) and left
empty, so nothing reads them as current. The answer checks are kept, and checked again on every run: a
refusal this checkout settles (a git pull, when it is behind the ref the GPU job ran) needs no GPU line.
Only lift restart of the cell retires a check.

The frontier judge (judges.utility) first sends its uncached requests as Message Batches when
judges.utility.bulk is batch, waits for them and then judges from its call cache. --no-wait submits the
batches and exits 3 (rerun the same line to collect them). --sync submits no batch and sends the uncached
requests synchronously through the Messages API, except those an earlier batch holds (one left by
--no-wait or by a stopped wait), which are never paid for twice: such a batch is collected first once it
has ended, and while it still runs the line judges nothing and exits 3. Then rerun it once the batch has
ended, cancel the batch in the Console and rerun, or set LIFT_ABANDON_BATCHES=<batch id> to give it up
and pay for those requests again. The fallback judge (Qwen3-8B in 4-bit, on --device) runs only when
asked for.

examples:
  lift utility --variant adapter-aya-std-s1 --split dev                  (a GPU job: generate, then judge)
  lift utility --variant adapter-aya-std-s1 --split dev --generate-only  (a GPU job line)
  lift utility --variant adapter-aya-std-s1 --split dev --judge-only     (after lift sync pull)
"""


def _refused_options(args: argparse.Namespace) -> str | None:
    """Why a combination of options makes no sense, or None."""
    if args.variant == "stock":
        return "--variant is the candidate; stock is the reference every pair is judged against"
    if args.batch_size is not None and args.batch_size < 1:
        return "--batch-size must be at least 1"
    if args.generate_only and (args.rejudge or args.sync or args.no_wait):
        return "--generate-only judges nothing, so --rejudge, --sync and --no-wait do not apply"
    if args.judge == "fallback" and (args.sync or args.no_wait):
        return ("--sync and --no-wait choose how the frontier judge sends its requests; the fallback judge "
                "runs locally")
    return None


def _answer(args: argparse.Namespace, prompts: Sequence[Any], dtype: str,
            skipped: dict[str, dict[str, str]]) -> None:
    """Pass both sides' answer files through the frozen harness: stock first, then the candidate, one model
    loaded at a time and freed before the next. Always, also for a file that answers every prompt: a check
    of the pids alone would reuse the answers of an adapter replaced since (rule 4), while ``generate``
    compares every record with the variant as built now, appends only what is missing and refuses a file of
    another configuration or stack. A side ``generate`` refuses is listed in ``skipped`` under "{variant}
    {cell}"; a side it accepts gets its check recorded (``utility.record_checked``), which --judge-only
    reads."""
    from lift import utility

    require_pinned("models.global")
    _refuse_other_dtype(dtype, args.device)
    from lift.models.generate import StaleOutputError, generate, generations_path
    from lift.models.variants import build_variant, get_spec

    names = (utility.REFERENCE, args.variant)
    specs = {}
    for name in names:                    # every name checked before any model loads
        try:
            specs[name] = get_spec(name)
        except KeyError as exc:
            raise SystemExit(exc.args[0]) from None
    for name in names:
        holder = {"variant": build_variant(specs[name], device=args.device, dtype=dtype)}
        try:
            path = generate(holder["variant"], prompts,
                            out_path=generations_path(name, utility.SET, utility.LANG, args.split),
                            batch_size=args.batch_size)
            checked = utility.record_checked(holder["variant"], args.split)
            log.info("%s: checked against %s as registered now (%s)", utility.logical(path), name,
                     utility.logical(checked))
        except StaleOutputError as exc:
            mine: dict[str, dict[str, str]] = {}
            _skip(mine, name, utility.SET, utility.LANG, args.split, exc, judged=None, dtype=dtype)
            for kind, cells in mine.items():   # both sides share the cell name: key them by variant too
                skipped.setdefault(kind, {}).update({f"{name} {cell}": why for cell, why in cells.items()})
        finally:
            holder.clear()
            _release_memory()


def _summary(report: Any) -> None:
    """Counts, rates and paths only."""
    from lift.utility import logical

    m = report.metrics
    fid = m["fidelity"]
    print(f"{m['variant']} vs stock, utility_hi_{m['split']}: {m['n']} pairs, judge {m['judge']}")
    print(f"  win rate {m['win_rate']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}], p vs parity "
          f"{m['p_vs_parity']:.3g}")
    print(f"  failed orders {m['failed_orders']} of {2 * m['n']} ({m['pairs_with_failure']} pairs); win rate "
          f"with them as losses {m['win_rate_failures_as_losses']:.3f}, as wins "
          f"{m['win_rate_failures_as_wins']:.3f}")
    print(f"  Hindi fidelity, top label / is_lang: candidate {fid['candidate']['top_hi']:.3f} / "
          f"{fid['candidate']['is_lang_hi']:.3f}, stock {fid['stock']['top_hi']:.3f} / "
          f"{fid['stock']['is_lang_hi']:.3f}")
    if report.prefill is not None:
        print("  Message Batches: " + ", ".join(f"{k} {v}" for k, v in report.prefill.items()
                                                  if k != "batch_ids"))
    if report.set_aside is not None:
        print(f"  previous verdicts set aside: {report.set_aside}")
    print(f"  verdicts ({'judged now' if report.judged else 'reused'}): {logical(report.verdicts_path)}")
    print(f"  metrics: {logical(report.metrics_json)} and {report.metrics_md.name}")
    if report.record_path is not None:
        print(f"  AuditKIT record: {logical(report.record_path)}")
    else:
        print("  AuditKIT record: refused (see the error above)")


def _utility(args: argparse.Namespace) -> int:
    _logging()
    refused = _refused_options(args)
    if refused:
        print(f"lift utility: {refused}", file=sys.stderr)
        return 2
    from lift import prompts, utility

    dtype = _dtype(args.dtype)
    cell = f"{args.variant} vs stock, {utility.SET}_{utility.LANG}_{args.split}"
    texts = prompts.load(utility.SET, utility.LANG, args.split)
    skipped: dict[str, dict[str, str]] = {}
    if not args.judge_only:
        _answer(args, texts, dtype, skipped)
    if skipped:
        log.error("%s: not judged, as a side's answers were skipped (listed below)", cell)
    if args.generate_only or skipped:
        return _refused_exit({}, skipped)
    if args.judge == "fallback":
        _refuse_other_dtype(dtype, args.device)
    pending: tuple[type[BaseException], ...] = ()
    if args.judge == "frontier":
        # Also under --sync: a synchronous pass first collects an earlier batch that holds its requests
        # (they are paid for), and stops with FrontierBatchPending while that batch still runs.
        from lift.judge import frontier

        pending = (frontier.FrontierBatchPending,)
    try:
        report = utility.evaluate(args.variant, args.split, judge_dir=args.judge, dtype=dtype,
                                  device=args.device, prefill=not args.sync, wait=not args.no_wait,
                                  rejudge=args.rejudge)
    except utility.UtilityRefused as exc:
        log.error("%s: not judged: %s", cell, exc)
        return 3
    except pending as exc:
        log.warning("%s: %s. The batches go on server-side: rerun the same line to collect them and judge",
                    cell, exc)
        return 3
    _summary(report)
    return 3 if report.record_error else 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("utility", help="Hindi utility win rate of a variant against stock (answers, both "
                                       "orders judged, metrics, AuditKIT record)",
                       description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True,
                   help="the candidate: a non-stock name from configs/variants.yaml")
    p.add_argument("--split", required=True, choices=["dev", "final"])
    p.add_argument("--judge", choices=["frontier", "fallback"], default="frontier",
                   help="frontier: judges.utility (default); fallback: the local Qwen3-8B, only when "
                        "asked for")
    p.add_argument("--device", default="cuda:0", help="the generator's device, and the fallback judge's")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None,
                   help="default: generation.dtype (another counts only under Gate 0's licence)")
    p.add_argument("--batch-size", type=int, default=None, help="generation batch size (default: from free "
                                                                "memory)")
    stage = p.add_mutually_exclusive_group()
    stage.add_argument("--generate-only", action="store_true",
                       help="stop after generating and checking both sides' answers (a GPU job line)")
    stage.add_argument("--judge-only", action="store_true",
                       help="load no model: judge and report the answers as a GPU line checked them (no GPU "
                            "stack or pin needed with the frontier judge)")
    bulk = p.add_mutually_exclusive_group()
    bulk.add_argument("--sync", action="store_true",
                      help="submit no Message Batch: judge every uncached request synchronously through the "
                           "Messages API (small or urgent runs). An earlier batch that holds some of them is "
                           "collected first once it has ended; while it still runs the line judges nothing "
                           "and exits 3, rather than pay for those requests twice: rerun once it has ended, "
                           "cancel it in the Console and rerun, or set LIFT_ABANDON_BATCHES=<batch id> to "
                           "give it up and pay again")
    bulk.add_argument("--no-wait", action="store_true",
                      help="submit the Message Batches and exit 3 without waiting; rerun the same line to "
                           "collect them")
    p.add_argument("--rejudge", action="store_true",
                   help="set the verdict file aside (outputs/.aside/, never synced) and judge every pair "
                        "again: the remedy for a file another judge wrote. Run it once, then leave it out")
    p.set_defaults(func=_utility)
