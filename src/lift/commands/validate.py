"""``lift validate ...``: blind human validation (P04).

    lift validate sample --round N [--check]     blind sheets + keys for round 1 or 2
    lift validate import --round N --lang L --file filled.csv
    lift validate score [--round N]              agreement of each judge with the readers
    lift validate decide-judge [--dry-run]       the Nov 6 judge decision rule, on round 1
    lift validate second-judge-sample            keys the frontier judge must cover
    lift validate utility-sheet --variant NAME   60 blind stock-vs-adapter Hindi pairs
    lift validate utility-import --file filled.csv
    lift validate utility-score

Outputs: validation/{sheets,keys,labels}/ (private), outputs/metrics/validation/ and
outputs/metrics/judge_decision.json. P03's Gate 1 reads outputs/metrics/validation/gate1.json.
Nothing here prints prompt or response text: only paths, ids and counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _setup() -> None:
    # Paths and ids are printed; a console in a legacy code page must not turn them into a crash.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")


def _fail(err: Exception) -> int:
    print(f"error: {err}")
    return 1


def _no_prompt_check(args: argparse.Namespace) -> None:
    if not args.prompt_check:
        print("  WARNING: --no-prompt-check: outputs left stale by a fixed translation could not be told "
              "apart from current ones")


def _check(args: argparse.Namespace) -> int:
    from lift.validation.sample import check_round  # noqa: PLC0415

    out = check_round(args.round_no, seed=args.seed, variants=args.variants,
                      hindi_overlap=args.hindi_overlap, prompt_check=args.prompt_check, force=args.force,
                      dtype=args.dtype)
    print(f"round {args.round_no} dry run (nothing written)")
    _no_prompt_check(args)
    for group, names in out["variants"].items():
        print(f"  group {group}: {', '.join(names) or '-'}")
    if out["ignored_variants"]:
        print(f"  not in this round: {', '.join(out['ignored_variants'])}")
    print("  file: current/expected, stale pids, missing, other dtype")
    for f in out["files"]:
        print(f"    {f['variant']}/{f['set']}_{f['lang']}_{f['split']}: {f['current']}/{f['expected']}, "
              f"{f['stale_pids']}, {f['missing']}, {f['other_dtype']}")
    for w in out["warnings"]:
        print(f"  warning: {w}")
    if out["problems"]:
        print("not drawable yet:")
        for p in out["problems"]:
            print(f"  {p}")
        return 1
    print("drawable")
    return 0


def _sample(args: argparse.Namespace) -> int:
    _setup()
    from lift.validation.sample import load_round, make_sheets  # noqa: PLC0415
    from lift.validation.store import ValidateError, round_keys_path  # noqa: PLC0415

    try:
        if args.check:
            return _check(args)
        sheets = make_sheets(args.round_no, seed=args.seed, variants=args.variants,
                             hindi_overlap=args.hindi_overlap, force=args.force,
                             prompt_check=args.prompt_check, dtype=args.dtype)
    except ValidateError as e:
        return _fail(e)
    doc = load_round(args.round_no)
    print(f"round {args.round_no}: {len(doc['items'])} items, seed {doc['seed']}, drawn {doc['created']}")
    if not doc.get("prompt_check", True):
        print("  WARNING: drawn with --no-prompt-check: outputs left stale by a fixed translation "
              "could not be told apart")
    for group, names in doc["variants"].items():
        print(f"  group {group}: {', '.join(names)}")
    if doc["ignored_variants"]:
        print(f"  not in this round: {', '.join(doc['ignored_variants'])}")
    stale = sum(f["stale_pids"] for f in doc["files"])
    other = sum(f["other_dtype"] for f in doc["files"])
    if stale or other:
        print(f"  left out: {stale} prompts with only stale outputs, {other} records in another dtype")
    print(f"  left out as already seen by the reader: {doc['excluded_seen_pids']}")
    for w in doc.get("warnings", []):
        print(f"  warning: {w}")
    for lang, alloc in doc["allocation"].items():
        cells = [f"{stratum} {k}/{doc['pools'][lang][stratum]}" for stratum, k in alloc.items()]
        print(f"  {lang} (items/outputs): " + ", ".join(cells))
    for path in sheets:
        info = next(i for i in doc["sheets"].values() if i["file"] == path.name)
        reader = "the second Hindi reader (overlap)" if info["slot"] == "b" else f"the {info['lang']} reader"
        print(f"  sheet {path}  ({info['n']} items, for {reader})")
    print(f"  keys  {round_keys_path(args.round_no)}  (never send this file to a reader)")
    return 0


def _import(args: argparse.Namespace) -> int:
    _setup()
    from lift.validation.labels import import_labels  # noqa: PLC0415
    from lift.validation.store import ValidateError  # noqa: PLC0415

    try:
        out = import_labels(args.round_no, args.lang, Path(args.file), annotator=args.annotator)
    except ValidateError as e:
        return _fail(e)
    counts = ", ".join(f"{k} {v}" for k, v in out["label_counts"].items())
    print(f"{out['sheet']}: {out['labelled']} labels from {out['annotator']} ({counts}); "
          f"{len(out['skipped'])} skipped, {len(out['unlabelled'])} unlabelled, "
          f"{len(out['missing'])} missing, of {out['items_on_sheet']}")
    if out["replaced"]:
        print(f"  replaced an earlier import ({out['replaced']['changed']} labels changed, "
              f"{out['replaced']['dropped']} no longer labelled)")
    for w in out["warnings"]:
        print(f"  warning: {w}")
    print(f"  wrote {out['labels_path']}")
    return 0


def _score(args: argparse.Namespace) -> int:
    _setup()
    from lift.validation.score import bar_text, gate1_path, score  # noqa: PLC0415
    from lift.validation.store import ValidateError, metrics_dir  # noqa: PLC0415

    try:
        doc = score(args.round_no, qwen3guard_judge=args.qwen3guard_judge,
                    frontier_judge=args.frontier_judge)
    except ValidateError as e:
        return _fail(e)
    print(f"rounds {', '.join(map(str, doc['rounds']))}: bar agreement >= {doc['bar']['agreement']} "
          f"or kappa >= {doc['bar']['kappa']}; primary judge {doc['primary_view']}")
    for view, j in doc["judges"].items():
        cells = []
        for lang, b in [*j["langs"].items(), ("pooled", j["pooled"])]:
            agree = "-" if b["agreement"] is None else f"{b['agreement']:.3f}"
            kappa = "-" if b["kappa"] is None else f"{b['kappa']:.3f}"
            cells.append(f"{lang} {agree}/{kappa} {bar_text(b)} ({b['n_judged']}/{b['n']})")
        print(f"  {view}: " + "; ".join(cells))
    checks = doc["checks"]
    if checks.get("labels_changed_since_decision"):
        print(f"  warning: labels changed since the judge decision: "
              f"{', '.join(checks['labels_changed_since_decision'])}")
    if checks["loose_disabled"]:
        print("  warning: Qwen3Guard native fields disagree with the stored labels; loose view withheld")
    stem = "all" if args.round_no is None else f"round{args.round_no}"
    print(f"  wrote {metrics_dir() / (stem + '.json')} and .md"
          + (f", and {gate1_path()} (Gate 1)" if 1 in doc["rounds"] else ""))
    return 0


def _decide(args: argparse.Namespace) -> int:
    _setup()
    from lift.validation.score import (  # noqa: PLC0415
        config_change,
        decide_primary_judge,
        decision_path,
        decisions_entry,
    )
    from lift.validation.store import ValidateError  # noqa: PLC0415

    try:
        doc = decide_primary_judge(qwen3guard_judge=args.qwen3guard_judge,
                                   frontier_judge=args.frontier_judge, force=args.force,
                                   allow_incomplete=args.allow_incomplete, dry_run=args.dry_run)
    except ValidateError as e:
        return _fail(e)
    a, d = doc["agreement"], doc["diagnostics"]
    side = "exactly at" if doc["boundary"] else ("above" if doc["rule_met"] else "below")
    print(f"round 1 pooled, n={doc['n']}: frontier {doc['n_agree']['frontier']} right "
          f"({a['frontier']:.4f}), Qwen3Guard strict {doc['n_agree']['qwen3guard_strict']} "
          f"({a['qwen3guard_strict']:.4f}); difference {doc['difference_exact']} = "
          f"{100 * doc['difference']:+.2f} points, {side} the threshold of {100 * doc['threshold']:.0f}")
    print(f"  the judges' labels differ on {d['judges_differ']} items; exactly one matches the reader on "
          f"{d['discordant']['frontier_only_right']} (frontier) + {d['discordant']['qwen3guard_only_right']} "
          f"(Qwen3Guard), McNemar p {d['discordant']['mcnemar_p']:.4f}; bootstrap "
          f"{100 * d['bootstrap']['lo']:+.1f} to {100 * d['bootstrap']['hi']:+.1f} points")
    by_class = ", ".join(f"{k} {100 * v:+.1f}" for k, v in d["margin_by_human_class"].items())
    print(f"  margin by human label (points): {by_class}")
    without = d["margin_without_unclear"]
    print(f"  margin without unclear {'-' if without is None else f'{100 * without:+.1f}'}, "
          f"harmful vs not {100 * d['margin_binary']:+.1f}, weighted by dev set size "
          f"{100 * d['margin_weighted_by_dev_set_size']:+.1f}")
    s = d["skips"]
    print(f"  {s['not_labelled']} items without a label could move it to {100 * s['margin_lo']:+.1f} .. "
          f"{100 * s['margin_hi']:+.1f} points: " + ("the decision holds either way" if s["robust_to_skips"]
                                                  else "WARNING: the decision could flip"))
    if doc["boundary"]:
        print("  the margin is exactly the threshold: the README's 'at least' settles it")
    if doc["allowed_incomplete"]:
        print(f"  WARNING: decided with --allow-incomplete (not labelled {doc['not_labelled']}, "
              f"superseded {doc['superseded']})")
    if doc["final_judgements_exist"]:
        print("  warning: final-split judgements already exist; the rule is meant to be settled first")
    if not doc["rule_applies"]:
        print(f"  the second judge is the fallback ({doc['frontier_judge']}), not the frontier LLM the "
              "rule is about: Qwen3Guard stays primary and nothing is proposed")
    print("dry run: nothing written" if args.dry_run else f"wrote {decision_path()}")
    if not doc["switch_primary"]:
        if doc["config_primary"] == "frontier":
            print("Decision: Qwen3Guard is the primary judge, but configs/project.yaml makes the frontier "
                  "judge primary (an earlier decision applied?). Nothing has been edited. A person swaps "
                  "the blocks back and logs it in docs/decisions.md:\n")
            print(config_change(doc))
            return 0
        print("Decision: Qwen3Guard stays the primary judge. No config or docs/decisions.md change.")
        return 0
    print("Decision: the frontier judge becomes the primary judge.")
    if doc["config_primary"] == "frontier":
        print("configs/project.yaml already names the frontier judge as primary: nothing more to change.")
        return 0
    print("Nothing has been edited. A person makes these two changes:\n")
    print("1. " + config_change(doc))
    print("2. Add this entry at the top of docs/decisions.md (newest first):\n")
    print(decisions_entry(doc))
    print("Then run `lift validate score --round 1` again: gate1.json follows the primary judge in the "
          "config.")
    return 0


def _second(args: argparse.Namespace) -> int:
    _setup()
    from lift.validation.score import config_primary  # noqa: PLC0415
    from lift.validation.second_judge import second_judge_sample  # noqa: PLC0415
    from lift.validation.store import ValidateError  # noqa: PLC0415

    try:
        doc = second_judge_sample(fraction=args.fraction, seed=args.seed,
                                  require_validated=args.validated, prompt_check=args.prompt_check,
                                  dtype=args.dtype)
    except ValidateError as e:
        return _fail(e)
    _no_prompt_check(args)
    print(f"{doc['n_keys']} keys: {doc['n_random']} in the {doc['fraction']:.0%} random sample, "
          f"{doc['n_validated']} validated ({doc['n_both']} in both), from {len(doc['files'])} files")
    if doc["incomplete_files"]:
        print(f"  skipped {len(doc['incomplete_files'])} incomplete files; run this again when they finish")
    if doc["regenerate_first"]:
        print(f"  skipped {len(doc['regenerate_first'])} files that hold outputs of an older prompt text, "
              "which the harness will not judge until they are regenerated: "
              + ", ".join(doc["regenerate_first"][:10]))
    if doc["validated_waiting"]:
        print(f"  held back {len(doc['validated_waiting'])} validated keys whose files must finish or be "
              "regenerated first; run this again afterwards")
    if doc["validated_not_current"]:
        print(f"  left {len(doc['validated_not_current'])} validated keys out of the lists: the outputs "
              "the readers saw are gone from the generations files (they score as superseded)")
    print(f"  wrote {doc['dir']} (random.txt, validated.txt, all.txt, one file per variant and split)")
    if config_primary() == "frontier":
        print("  the config makes the frontier judge primary, so it runs on every output anyway; these "
              "lists now only matter for a second judge that does not already label everything")
    print("  second-judge runs (check P02's `lift judge --help` for the exact flags):")
    for line in doc["commands"]:
        print(f"    {line}")
    return 0


def _utility_sheet(args: argparse.Namespace) -> int:
    _setup()
    from lift.validation.store import ValidateError  # noqa: PLC0415
    from lift.validation.utility import load_utility_keys, make_utility_sheet  # noqa: PLC0415

    try:
        path = make_utility_sheet(args.variant, split=args.split, seed=args.seed, force=args.force,
                                  prompt_check=args.prompt_check, dtype=args.dtype)
    except ValidateError as e:
        return _fail(e)
    _no_prompt_check(args)
    doc = load_utility_keys()
    print(f"{doc['n']} pairs of stock vs {doc['candidate']} ({doc['split']}), candidate shown as A "
          f"in {doc['candidate_as_a']}")
    print(f"  sheet {path}  (for the Hindi reader)")
    return 0


def _utility_import(args: argparse.Namespace) -> int:
    _setup()
    from lift.validation.store import ValidateError  # noqa: PLC0415
    from lift.validation.utility import import_preferences  # noqa: PLC0415

    try:
        out = import_preferences(Path(args.file), annotator=args.annotator)
    except ValidateError as e:
        return _fail(e)
    counts = ", ".join(f"{k} {v}" for k, v in out["preference_counts"].items())
    print(f"utility_hi: {out['labelled']} preferences ({counts}); {len(out['skipped'])} skipped, "
          f"{len(out['unlabelled'])} unlabelled, {len(out['missing'])} missing, of {out['items_on_sheet']}")
    for w in out["warnings"]:
        print(f"  warning: {w}")
    print(f"  wrote {out['labels_path']}")
    return 0


def _utility_score(args: argparse.Namespace) -> int:
    _setup()
    from lift.validation.store import ValidateError, metrics_dir  # noqa: PLC0415
    from lift.validation.utility import score_utility  # noqa: PLC0415

    try:
        out = score_utility(verdict_files=args.verdicts, judge_id=args.judge_id)
    except ValidateError as e:
        return _fail(e)
    if not out["n_judged"]:
        print(f"no verdicts found for {out['candidate']} on the sheet's pairs "
              f"({out['n_labelled']} labelled); pass --verdicts FILE")
    else:
        d = out["judge_minus_human"]
        print(f"{out['n_judged']} pairs: agreement {out['agreement']:.3f}, kappa {out['kappa']:.3f}; "
              f"win rate reader {out['human_win_rate']['win_rate']:.3f}, judge "
              f"{out['judge_win_rate']['win_rate']:.3f} (judge minus reader {d['difference']:+.3f}, "
              f"{d['lo']:+.3f} to {d['hi']:+.3f})"
              + ("" if out["complete"] else f"; incomplete: {out['n_missing']} without a verdict"))
    if out["n_superseded"]:
        print(f"  left out {out['n_superseded']} pairs whose outputs were remade after the sheet was made")
    print(f"  wrote {metrics_dir() / 'utility.json'} and .md")
    return 0


def _judge_flags(s: argparse.ArgumentParser) -> None:
    s.add_argument("--qwen3guard-judge", default=None, metavar="ID",
                   help="the Qwen3Guard judge id (or id@revision) to use if several are found")
    s.add_argument("--frontier-judge", default=None, metavar="ID",
                   help="the frontier judge id (or id@revision) to use if several are found")


def _prompt_flag(s: argparse.ArgumentParser) -> None:
    s.add_argument("--no-prompt-check", dest="prompt_check", action="store_false",
                   help="escape hatch: treat every output as current even if its prompt_sha does not "
                        "match the prompt text (printed as a warning; recorded in what it writes)")
    s.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"],
                   help="the precision of the outputs to use (default: generation.dtype; another is "
                        "refused unless Gate 0's licence, gate0-<dtype>.json, covers the outputs)")


def register(sub: argparse._SubParsersAction) -> None:
    from lift.validation.sample import DEFAULT_SEED, ROUNDS  # noqa: PLC0415

    rounds = sorted(ROUNDS)
    p = sub.add_parser("validate", help="human validation: blind sheets, label import, judge agreement",
                       description="Blind human validation: sheets, label import, judge agreement, "
                                   "the judge decision, the second-judge sample and the utility check.")
    steps = p.add_subparsers(dest="step", required=True, metavar="<step>")

    s = steps.add_parser("sample", help="draw a round and write its blind sheets and keys")
    s.add_argument("--round", dest="round_no", type=int, required=True, choices=rounds)
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--variants", nargs="+", metavar="NAME",
                   help="only these variants (default: every variant with outputs that fits the round; "
                        "round 1 takes standard-recipe adapters unless others are named)")
    s.add_argument("--hindi-overlap", action=argparse.BooleanOptionalAction, default=None,
                   help="20 of the Hindi items again on a second sheet, for a second Hindi reader "
                        "(default: validation.hindi_overlap in configs/project.yaml, else off)")
    s.add_argument("--check", action="store_true",
                   help="dry run with the same options: report what each generations file holds, the "
                        "draw's warnings and anything that would stop it; writes nothing")
    s.add_argument("--force", action="store_true",
                   help="replace a round drawn before with a different sample (archives the old one)")
    _prompt_flag(s)
    s.set_defaults(func=_sample)

    s = steps.add_parser("import", help="import a filled sheet as HumanLabel records")
    s.add_argument("--round", dest="round_no", type=int, required=True, choices=rounds)
    s.add_argument("--lang", required=True, choices=["hi", "zh", "es", "en"])
    s.add_argument("--file", required=True, help="the CSV the reader sent back")
    s.add_argument("--annotator", default=None,
                   help="annotator id (default: {lang}-a, or {lang}-b for the overlap sheet)")
    s.set_defaults(func=_import)

    s = steps.add_parser("score", help="agreement, kappa, bar check and confusion matrices per judge")
    s.add_argument("--round", dest="round_no", type=int, choices=rounds, default=None,
                   help="one round (default: every round drawn so far, pooled)")
    _judge_flags(s)
    s.set_defaults(func=_score)

    s = steps.add_parser("decide-judge", help="apply judges.decision_rule to round 1 (pooled)")
    _judge_flags(s)
    s.add_argument("--dry-run", action="store_true", help="print the decision, write nothing")
    s.add_argument("--allow-incomplete", action="store_true",
                   help="decide although some rows have no label and no 'skipped' note, or are superseded "
                        "(their output changed after the draw); recorded in the decision")
    s.add_argument("--force", action="store_true",
                   help="replace a different recorded decision (the old one stays in its history)")
    s.set_defaults(func=_decide)

    s = steps.add_parser("second-judge-sample",
                         help="list the generation keys the second judge must cover")
    s.add_argument("--fraction", type=float, default=None,
                   help="share of each set and split (default: second_judge_fraction)")
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--no-validated", dest="validated", action="store_false",
                   help="write the random sample although no validation round keys are here")
    _prompt_flag(s)
    s.set_defaults(func=_second)

    s = steps.add_parser("utility-sheet", help="blind stock-vs-adapter Hindi pairs for the utility check")
    s.add_argument("--variant", required=True, help="the adapter compared with stock")
    s.add_argument("--split", default="dev", choices=["dev", "final"])
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--force", action="store_true",
                   help="replace a different utility sample (archives the old one)")
    _prompt_flag(s)
    s.set_defaults(func=_utility_sheet)

    s = steps.add_parser("utility-import", help="import the filled utility sheet")
    s.add_argument("--file", required=True)
    s.add_argument("--annotator", default="hi-a")
    s.set_defaults(func=_utility_import)

    s = steps.add_parser("utility-score", help="score the reader's preferences against the utility judge")
    s.add_argument("--verdicts", nargs="+", default=None, metavar="FILE",
                   help="UtilityVerdict JSONL files (default: found under outputs/, except what lift "
                        "sync never syncs: outputs/.aside, outputs/.sync-backup, caches)")
    s.add_argument("--judge-id", default=None, help="the utility judge to use if several are found")
    s.set_defaults(func=_utility_score)
