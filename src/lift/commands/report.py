"""``lift stats|report``: the final numbers and the report built from them (P07). Laptop, no torch.

  lift sync pull && lift stats && lift report

``lift stats`` (lift.report.analysis) writes outputs/metrics/final_stats.json; ``lift report``
(lift.report.tables, figures, cost) writes outputs/metrics/final_report.json (the file P08 reads), the
markdown and CSV tables in outputs/metrics/report/, outputs/metrics/final_report.md and the figures in
outputs/figures/. An arm that was not run is shown as "not run", never estimated. Both read final results of
repaired variants only under the recipe the newest ``recipe-frozen-*`` tag froze (lift final's check,
``lift.final.reported_recipe``); otherwise they refuse (exit 2) and write nothing.
"""

from __future__ import annotations

import argparse
import logging
import sys


def _stats(args: argparse.Namespace) -> int:
    from lift.final import Refused
    from lift.report import analysis

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        doc = analysis.run(dtype=args.dtype)
    except (analysis.Unpairable, Refused) as exc:
        print(f"lift stats: refused: {exc}", file=sys.stderr)
        return 2
    for v in doc["verdicts"]:
        print(f"{v['adapter']} ({v['recipe']}): {v['verdict'] or 'not run'}. {v['reason']}")
    for key in ("drift", "repair"):
        pooled = doc["primary"][key]["pooled"]
        print(
            f"primary {key}: "
            + (
                "not run"
                if pooled is None
                else f"{100 * pooled['est']:+.1f} pp [{100 * pooled['lo']:+.1f}, "
                f"{100 * pooled['hi']:+.1f}], p {pooled['p']:.3g}, seeds {pooled['seeds']}"
            )
        )
    print(f"{len(doc['not_run'])} arms or cells not run")
    print(f"wrote {analysis.stats_path()}")
    return 0 if not doc["not_run"] else 3


def _report(args: argparse.Namespace) -> int:
    from lift.final import Refused
    from lift.report import tables

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        doc = tables.build_all(figures=not args.no_figures)
    except FileNotFoundError as exc:
        print(f"lift report: {exc}")
        return 2
    except Refused as exc:
        print(f"lift report: refused: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {tables.report_path()}, {len(doc['written']) - 1} table files and "
        f"{len(doc.get('figures', []))} figures"
    )
    if doc["not_run"]:
        print(f"{len(doc['not_run'])} arms or cells are shown as not run")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("stats", help="every final-split number: endpoints, corrections, verdicts (laptop)")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default=None, help="default: generation.dtype")
    p.set_defaults(func=_stats)
    p = sub.add_parser("report", help="final_report.json, tables, figures and the cost section (laptop)")
    p.add_argument("--no-figures", action="store_true", help="skip the figures (no matplotlib needed)")
    p.set_defaults(func=_report)
