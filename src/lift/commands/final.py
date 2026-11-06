"""``lift final``: the final evaluation of frozen variants (P07). The rules are in lift.final.

Order (prompts/P07_final_report.md):
  1. lift final --all                       GPU job line (or --variant NAME, one variant per job)
  2. lift final --all --judge-only          laptop, after lift sync pull: second judge's 10% and utility
  3. lift stats && lift report              laptop (lift.commands.report)
"""

from __future__ import annotations

import argparse
import logging
import sys

log = logging.getLogger("lift.final")

HELP = """\
The final evaluation of one frozen variant (--variant) or of the 13 core variants in priority order (--all:
stock, the three adapter seeds, the overall winner on seeds 1-3, then the other two families), on the
final split only. Refused unless a recipe-frozen-* git tag exists and the variant is frozen (stock, a
registered adapter, or a setting outputs/metrics/selection.json froze, on seeds 1-3).

Per variant: the safety cells (harmful, easy and hard benign in six languages and Aya Red-teaming Hindi,
generated and judged by the primary judge, on two GPUs when there are two), the utility answers (lift
utility --generate-only), Belebele in six languages and the ar/th back-translation sample. Resumes after a
crash (outputs/metrics/final/state/{variant}.json); each run writes a manifest with its GPU-hours.
Exit 0 done, 3 some stage left cells listed with their remedies (or a variant was not run), 1 some stage
failed, 2 refused.

--judge-only (laptop, after lift sync pull): the second judge's 10% sample (P04's key list when drawn)
and lift utility --judge-only for each variant.
"""


def _final(args: argparse.Namespace) -> int:
    from lift import final

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if bool(args.variant) == bool(args.all):
        print("lift final: pass --variant NAME or --all", file=sys.stderr)
        return 2
    stages = [s for item in (args.stages or final.STAGES) for s in str(item).split(",") if s]
    bad = sorted(set(stages) - set(final.STAGES))
    if bad:
        print(f"lift final: unknown stages {bad}; choose from {list(final.STAGES)}", file=sys.stderr)
        return 2
    opts = final.Options(
        device=args.device,
        judge_device=args.judge_device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        recheck=args.recheck,
        stages=stages,
    )
    try:
        if args.judge_only:
            if args.all:
                names = [p.name for p in final.core_plan(final.read_selection() or {})]
            else:
                final.check_final(args.variant)
                names = [args.variant]
            return final.judge_only(names, no_wait=args.no_wait, sync=args.sync)
        if args.all:
            rc, summary = final.run_all(opts)
            for name, stages_rc in summary["results"].items():
                print(f"  {name}: " + ", ".join(f"{s} {rc_}" for s, rc_ in stages_rc.items()))
            for name, why in summary["not_run"].items():
                print(f"  {name}: not run ({why})")
            return rc
        tag = final.check_final(args.variant)
        results = final.run_variant(args.variant, opts, tag=tag)
    except final.Refused as exc:
        print(f"lift final: {exc}", file=sys.stderr)
        return 2
    print(f"{args.variant}: " + ", ".join(f"{s} {rc}" for s, rc in results.items()))
    return final.exit_code({args.variant: results})


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "final",
        help="the final evaluation of frozen variants (GPU; --judge-only on the laptop)",
        description=HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--variant", default=None, help="a frozen variant from configs/variants.yaml")
    p.add_argument("--all", action="store_true", help="the 13 core variants, primary endpoint first")
    p.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="safety utility belebele backtranslation (default: all, in that order)",
    )
    p.add_argument("--device", default="cuda:0", help="generator device")
    p.add_argument("--judge-device", default="auto", help="auto = cuda:1 if two GPUs, else cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default=None, help="default: generation.dtype")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--recheck", action="store_true", help="run every stage again (each resumes by cache key)")
    p.add_argument("--judge-only", action="store_true", help="laptop: the second judge's sample and utility")
    p.add_argument("--no-wait", action="store_true", help="with --judge-only: submit the batches and exit 3")
    p.add_argument("--sync", action="store_true", help="with --judge-only: no Message Batches for utility")
    p.set_defaults(func=_final)
