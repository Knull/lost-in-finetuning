"""``lift data build|stats|spotcheck-sheets|apply-spotchecks``: the data pipeline (lift.data.build).

Handlers import lift.data lazily, so ``lift --help`` needs no data packages. Each one logs INFO to
stderr and writes stdout/stderr with errors="backslashreplace", so a cp1252 console never crashes.
``lift data build`` exits 0 when complete (listed failures are expected and reported) and 3 when
translation was blocked, a file was skipped for translation, or items a rerun can translate are
missing: rerun to resume from the cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# lift.data.build.SETS, repeated so registering the parser imports nothing from lift.data (a test
# checks the two agree).
SETS = ("harmful", "easy_benign", "hard_benign", "redteam", "utility", "belebele", "sft")
EXIT_INCOMPLETE = 3

log = logging.getLogger(__name__)


def _setup() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="backslashreplace")
            except (ValueError, OSError):
                pass
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _build(args: argparse.Namespace) -> int:
    _setup()
    from lift.data.build import BuildContext, build_all
    from lift.paths import prompts_dir

    ctx = BuildContext.default(skip_translation=args.skip_translation, cache_only=args.cache_only,
                               retranslate_failed=args.retranslate_failed,
                               fallback_engine=args.fallback_engine)
    summary = build_all(only=args.only, langs=args.langs, ctx=ctx)
    totals = summary["totals"]
    print(f"{totals['files']} files, {totals['records']} records; {totals['lid_failures']} GlotLID "
          f"rejections, {totals['translation_failures']} translation failures ({totals['open_failures']} "
          f"open), {totals['exclusions']} excluded")
    for x in summary["skipped"]:
        print(f"skipped {x['set']}_{x['lang']}: {x['reason']}")
    if summary["incomplete"]:
        print("items a rerun can translate are missing from: " + ", ".join(summary["incomplete"]))
    if summary["sft"]:
        d = summary["sft"]["decision"]
        print(f"sft: {d['rows']} rows per candidate (target {d['rows_target']}, "
              f"limited by {d['limited_by']})")
    if summary["blocked"]:
        print("translation BLOCKED (Google refused this IP): wait 30 minutes or more, then rerun; the cache "
              "keeps everything done")
    state = "complete" if summary["complete"] else "INCOMPLETE: rerun to resume from the cache"
    print(f"{state}; summary in {prompts_dir() / 'reports' / 'build.json'}")
    return 0 if summary["complete"] else EXIT_INCOMPLETE


def _stats(args: argparse.Namespace) -> int:
    _setup()
    from lift.data.build import BuildContext, format_stats, stats

    st = stats(cfg=BuildContext.default().cfg)
    print(json.dumps(st, indent=1, ensure_ascii=False, sort_keys=True) if args.json else format_stats(st))
    return 0


def _sheets(args: argparse.Namespace) -> int:
    _setup()
    from lift.data.build import BuildContext, write_spotcheck_sheets

    try:
        written = write_spotcheck_sheets(langs=args.langs, n=args.n, overwrite=args.overwrite,
                                         ctx=BuildContext.default())
    except (FileExistsError, FileNotFoundError) as e:
        log.error("%s", e)
        return 1
    for path in written:
        print(path)
    return 0


def _apply(args: argparse.Namespace) -> int:
    _setup()
    from lift.data.build import BuildContext, SheetError, apply_spotchecks

    try:
        result = apply_spotchecks(args.file, ctx=BuildContext.default(), dry_run=args.dry_run)
    except (SheetError, FileNotFoundError) as e:
        log.error("%s", e)
        return 1
    for name, t in result["files"].items():
        print(f"{name}: ok {t['ok']}, fixed {len(t['fixed'])} {t['fixed']}, "
              f"bad {len(t['bad'])} {t['bad']}, unreviewed {t['unreviewed']}")
    for x in result["not_applied"]:
        print(f"not applied: {x['lang']}-{x['pid']} {x['verdict']} ({x['why']})")
    if result["dry_run"]:
        print(f"dry run: {result['reviewed']} verdicts checked, nothing written")
    else:
        print(f"{result['reviewed']} verdicts applied; rebuilt {', '.join(result['rebuilt']) or 'nothing'}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    from lift import config

    langs = list(config.load("project")["languages"]["all"])
    translated = [lang for lang in langs if lang != "en"]
    p = sub.add_parser("data", help="build the prompt sets, splits and SFT candidates; spot checks",
                       description="The data pipeline: every prompt set, translation and frozen split, "
                                   "the SFT candidates and the native readers' spot checks.")
    data = p.add_subparsers(dest="data_command", required=True, metavar="<data command>")

    b = data.add_parser("build", help="build data/prompts, data/splits and data/sft",
                        description="Build every prompt set (resumable: translations are cached), freeze "
                                    "the splits and make the SFT candidates. Exit 0 when complete, 3 when "
                                    "translation was blocked or skipped (rerun to resume).")
    b.add_argument("--only", nargs="+", choices=SETS, metavar="SET",
                   help=f"build only these sets: {', '.join(SETS)}")
    b.add_argument("--langs", nargs="+", choices=langs, metavar="LANG",
                   help=f"build only these languages: {', '.join(langs)} (redteam, utility and sft "
                        "are Hindi)")
    b.add_argument("--skip-translation", action="store_true",
                   help="skip every file that needs a translation (never writes a partial file)")
    cache = b.add_mutually_exclusive_group()
    cache.add_argument("--cache-only", action="store_true",
                       help="use cached translations only; never call a translation engine")
    cache.add_argument("--retranslate-failed", action="store_true",
                       help="request again the cached translations that fail the checks")
    b.add_argument("--fallback-engine", choices=["nllb"],
                   help="translate the hard_benign items Google failed with NLLB-200 (needs torch)")
    b.set_defaults(func=_build)

    s = data.add_parser("stats", help="counts per set, language and split; failures and exclusions")
    s.add_argument("--json", action="store_true", help="print the counts as JSON")
    s.set_defaults(func=_stats)

    c = data.add_parser("spotcheck-sheets", help="write the translation spot-check sheets to "
                                                 "validation/sheets")
    c.add_argument("--langs", nargs="+", choices=translated, metavar="LANG",
                   help=f"languages (default: hi zh es; any of {', '.join(translated)})")
    c.add_argument("--n", type=int,
                   help="items per translated set (default validation.translation_spot_check)")
    c.add_argument("--overwrite", action="store_true", help="replace sheets that already exist")
    c.set_defaults(func=_sheets)

    a = data.add_parser("apply-spotchecks", help="apply a filled spot-check sheet (splits untouched)")
    a.add_argument("--file", required=True, type=Path, help="the filled CSV sheet")
    a.add_argument("--dry-run", action="store_true", help="check the sheet and report, write nothing")
    a.set_defaults(func=_apply)
