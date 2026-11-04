"""``lift data build|stats|spotcheck-sheets|apply-spotchecks``: the data pipeline (lift.data.build).

Handlers import lift.data lazily, so ``lift --help`` needs no data packages. Each one logs INFO to
stderr and writes stdout/stderr with errors="backslashreplace", so a cp1252 console never crashes.

Exit status. ``lift data build``: 0 when nothing a rerun could add is missing (listed failures are
expected and reported; files left out by --skip-translation are listed and need a run without it);
3 (EXIT_INCOMPLETE) when the summary's rerun_needed is set: translation was blocked (HTTP 429:
Google refused this IP, wait 30 minutes or more; a streak: ten items in a row failed with network
or service errors, check the network), a file was skipped because of the block, or items a rerun can
translate are missing: rerun to resume from the cache; 1 on an ImportError (a missing package: torch
and transformers for --fallback-engine nllb, checked before anything is written; deep-translator,
fasttext), on a TranslationSetupError (the Cloud Translation API key unset, or the API refusing it:
one line, no traceback; the items before it are cached), and on any other error, which stops the
build with a traceback (a split file that no longer matches the sources, a PolyRefuse clone at another
commit, an NLLB failure).
``lift data apply-spotchecks``: 0 when applied (or nothing new), 1 when the sheet or the
translation cache is refused (SheetError), a file is missing or a package (GlotLID's fasttext) is,
all before a verdict is stored; 3 when the rebuild lost records (listed; the verdicts are stored,
and the files' reports say why). ``spotcheck-sheets``: 1 when a sheet exists (no --overwrite), a
file is missing or the translation cache lacks a failure's engine output.
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


def _import_error(e: ImportError) -> int:
    hint = "" if "pip install" in str(e) else " (pip install -e \".[data]\" installs the data packages)"
    log.error("%s%s", e, hint)
    return 1


def _build(args: argparse.Namespace) -> int:
    _setup()
    from lift.paths import prompts_dir

    try:
        from lift.data.build import BuildContext, blocked_advice, build_all, build_state
        from lift.data.translate import TranslationSetupError
    except ImportError as e:
        return _import_error(e)
    try:
        ctx = BuildContext.default(skip_translation=args.skip_translation, cache_only=args.cache_only,
                                   retranslate_failed=args.retranslate_failed,
                                   fallback_engine=args.fallback_engine)
        summary = build_all(only=args.only, langs=args.langs, ctx=ctx)
    except ImportError as e:                 # check_nllb's (with its install hint), deep-translator, fasttext
        return _import_error(e)
    except TranslationSetupError as e:       # no Cloud Translation key, or the API refusing it
        log.error("translation cannot run: %s", e)
        return 1
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
        if d.get("rows") is not None:
            print(f"sft: {d['rows']} rows per candidate (target {d['rows_target']}, "
                  f"limited by {d['limited_by']})")
        else:
            print(f"sft: rows per variant (target {d['rows_target']}, no equal-size rule): "
                  + ", ".join(f"{k} {v}" for k, v in sorted(d["rows_by_variant"].items())))
    if summary["blocked"]:
        print(blocked_advice(summary["blocked_kind"]))
        print(f"blocked: {summary['blocked_reason']}")
    print(f"{build_state(summary)}; summary in {prompts_dir() / 'reports' / 'build.json'}")
    return EXIT_INCOMPLETE if summary["rerun_needed"] else 0


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
    except (SheetError, FileNotFoundError, ImportError) as e:      # ImportError: before a verdict is stored
        log.error("%s", e)
        return 1
    for name, t in result["files"].items():
        print(f"{name}: ok {t['ok']}, fixed {len(t['fixed'])} {t['fixed']}, bad {len(t['bad'])} {t['bad']}, "
              f"unchanged {t['unchanged']}, unreviewed {t['unreviewed']}")
    for x in result["not_applied"]:
        print(f"not applied: {x['lang']}-{x['pid']} {x['verdict']} ({x['why']})")
    if result["dry_run"]:
        print(f"dry run: {result['reviewed']} verdicts checked ({result['new']} new, {result['unchanged']} "
              "unchanged), nothing written")
        return 0
    for name, entry in result["files_rebuilt"].items():
        print(f"rebuilt {name}: " + ("not written" if entry is None else
                                     f"{entry['records']} records, {entry['open_failures']} open failures, "
                                     f"{entry['errors']} errors"))
    for name, pids in result["lost"].items():
        print(f"LOST from {name}: {len(pids)} records {pids}")
    print(f"{result['new']} verdicts applied, {result['unchanged']} unchanged; rebuilt "
          f"{', '.join(result['rebuilt']) or 'nothing'}")
    if result["lost"]:
        print("INCOMPLETE: the rebuild lost records (listed above); the verdicts are stored. The files' "
              "reports (translation_failures) and the not applied lines say why")
        return EXIT_INCOMPLETE
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
                                    "the splits and make the SFT candidates. Exit 0 when built (files "
                                    "--skip-translation left out are listed), 3 when translation was "
                                    "blocked or items a rerun can translate are missing (rerun to resume "
                                    "from the cache), 1 when a package is missing or the build stops on "
                                    "an error.")
    b.add_argument("--only", nargs="+", choices=SETS, metavar="SET",
                   help=f"build only these sets: {', '.join(SETS)}")
    b.add_argument("--langs", nargs="+", choices=langs, metavar="LANG",
                   help=f"build only these languages: {', '.join(langs)} (redteam, utility and sft "
                        "are Hindi)")
    b.add_argument("--skip-translation", action="store_true",
                   help="skip every file that needs a translation (never writes a partial file; exit 0)")
    cache = b.add_mutually_exclusive_group()
    cache.add_argument("--cache-only", action="store_true",
                       help="use cached translations only; never call a translation engine")
    cache.add_argument("--retranslate-failed", action="store_true",
                       help="request again the cached translations that fail the checks")
    b.add_argument("--fallback-engine", choices=["nllb"],
                   help="translate the hard_benign items Google failed with NLLB-200 (needs torch and "
                        "transformers)")
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

    a = data.add_parser("apply-spotchecks", help="apply a filled spot-check sheet (splits untouched)",
                        description="Check a filled sheet against the build and the stored verdicts, store "
                                    "the new verdicts and rebuild their files from the translation cache. "
                                    "Exit 0 when applied, 1 when the sheet or the cache is refused (nothing "
                                    "applied), 3 when the rebuild lost records.")
    a.add_argument("--file", required=True, type=Path,
                   help="the filled sheet, saved as CSV UTF-8 (commas, semicolons or tabs)")
    a.add_argument("--dry-run", action="store_true", help="check the sheet and report, write nothing")
    a.set_defaults(func=_apply)
