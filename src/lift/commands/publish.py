"""``lift publish model|datasets|collections|article|space|preflight``: the HF artifacts (P08).

Every upload command stages its files under outputs/publish/, prints what it would upload, and uploads
only with ``--yes`` (a person runs the real uploads, with HF_TOKEN set by a hidden prompt). ``--yes``
refuses while the HF namespace is unset (LIFT_HF_NAMESPACE or configs/publish.yaml), while something the
plan needs is not built, while a staged text still holds a ``[PENDING ...]`` placeholder (an input that
does not exist yet; ``--allow-pending`` uploads such a repo as a private draft instead), or while a file
that would be public fails the public-text scan, which needs this machine's outputs (``lift sync pull``).
``--yes`` also refuses every plan that names the repaired model or its numbers (model, parameters, logs,
collections) while configs/frozen_recipe.json is not the one the newest ``recipe-frozen-*`` tag froze
(the dry run prints it BLOCKED); ``article`` then writes a draft marked blocked and exits 1. A recipe
that differs from the tag's is refused outright.
A dry run runs that scan too and exits 1 when a file that would be public holds a run of any model
output's text, whatever its label (decisions item 62).
``article`` writes hf/article.md and the same text as README.md.
``preflight`` checks the git repo can go public: it fetches every
branch, tag and pull-request ref of origin first, and refuses a machine with no model outputs to check
against. Exit codes: 0 done (or dry run), 1 refused or failed (preflight or a dry run's scan: findings).
"""

from __future__ import annotations

import argparse
import sys

_EPILOG = "Dry run by default: nothing is uploaded without --yes."


def _fail(exc: Exception) -> int:
    print(f"lift publish: {exc}", file=sys.stderr)
    return 1


def _finish(uploads: list, args: argparse.Namespace, message: str) -> int:
    from lift.publish import hub

    clean = hub.report(uploads, yes=args.yes)
    if args.yes:
        hub.execute(uploads, message=message, allow_pending=args.allow_pending)
    return 0 if clean else 1


def _model(args: argparse.Namespace) -> int:
    from lift.publish import PublishError, model

    try:
        uploads = model.build_all(
            variant=args.variant, build_weights=args.build, device=args.device, adapters=not args.no_adapters
        )
        return _finish(uploads, args, "lift publish model")
    except PublishError as exc:
        return _fail(exc)


def _datasets(args: argparse.Namespace) -> int:
    from lift.publish import PublishError, datasets

    try:
        return _finish(datasets.build(args.only), args, "lift publish datasets")
    except PublishError as exc:
        return _fail(exc)


def _space(args: argparse.Namespace) -> int:
    from lift.publish import PublishError, space

    try:
        return _finish([space.build()], args, "lift publish space")
    except PublishError as exc:
        return _fail(exc)


def _collections(args: argparse.Namespace) -> int:
    from lift.publish import PublishError, collect
    from lift.publish.inputs import Facts, recipe_block

    try:
        facts = Facts()
        plan = collect.plan(facts)
        for line in collect.describe(plan):
            print(line)
        block = recipe_block(facts)
        if block:
            print(f"BLOCKED: {block}")
        if args.yes:
            collect.execute(plan)
        else:
            print("dry run: nothing was created. Add --yes to create the collections.")
        return 0
    except PublishError as exc:
        return _fail(exc)


def _article(args: argparse.Namespace) -> int:
    from lift.publish import PublishError, article
    from lift.publish.inputs import recipe_block, shown

    try:
        path, facts = article.build(readme=True)
        block = recipe_block(facts)
    except PublishError as exc:
        return _fail(exc)
    print(
        f"wrote {shown(path)} and the same text as {shown(article.readme_path())} ({len(facts.sources)} "
        f"values, sources in {shown(article.sources_path())})"
    )
    pend = article.placeholders(path.read_text(encoding="utf-8"))
    if pend:
        print(f"{len(pend)} placeholders still pending (inputs not made yet, or for a person to fill in):")
        for p in pend:
            print(f"  {p}")
    if block:
        print(f"lift publish: BLOCKED, a draft not for posting: {block}", file=sys.stderr)
        return 1
    print(
        "Commit README.md, figures/ and hf/article.md (with hf/article.sources.json): the README is the "
        "write-up the submission links to."
    )
    return 0


def _preflight(args: argparse.Namespace) -> int:
    from lift.publish import PublishError, preflight

    fetched = False
    try:
        if not args.no_history and not args.no_fetch:
            fetched = preflight.fetch_public_refs()
        windows, n_outputs, n_gone = preflight.load_windows()
    except PublishError as exc:
        return _fail(exc)
    if n_gone:
        print(
            f"note: {n_gone} judged outputs are no longer on this machine (regenerated): "
            "their text cannot be looked for"
        )
    findings = preflight.run(history=not args.no_history, windows=windows, log=print)
    if findings:
        print(f"preflight FAILED: {len(findings)} finding(s) (matched text is never shown):")
        for f in findings[:200]:
            print(f"  {f}")
        if len(findings) > 200:
            print(f"  ... and {len(findings) - 200} more")
        if any(f.where.startswith("history") for f in findings):
            print(
                "A finding in the history stays public even after the file is fixed: rewrite the history "
                "(git filter-repo) or publish a fresh repo, and revoke any secret found. Model-output text "
                "found in our own code (a person read the blob) can instead be listed in "
                f"{preflight.ALLOW_FILE} with a decisions.md entry."
            )
        return 1
    print(
        "preflight passed: no secret, token, raw completion, notebook output or text of the "
        f"{n_outputs} model outputs on this machine ({len(windows)} windows) in the tracked files"
        + ("" if args.no_history else " or the git history of every ref")
        + "."
    )
    partial = []
    if args.no_history:
        partial.append("the git history was not scanned (--no-history)")
    elif not fetched:
        partial.append(
            "only local refs were scanned (--no-fetch, or no origin remote): GitHub also publishes every "
            "remote branch, tag and pull-request ref"
        )
    if partial:
        print(f"NOT ENOUGH TO MAKE THE REPO PUBLIC: {'; '.join(partial)}. Run lift publish preflight alone.")
    print(preflight.PUBLIC_STEPS)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "publish",
        help="stage and upload the HF artifacts (dry run unless --yes); preflight",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cmds = p.add_subparsers(dest="publish_command", required=True, metavar="<what>")

    def upload_flags(q: argparse.ArgumentParser) -> None:
        q.add_argument("--yes", action="store_true", help="upload for real (default: dry run)")
        q.add_argument(
            "--allow-pending",
            action="store_true",
            help="upload a repo whose staged text still holds a [PENDING ...] placeholder as a PRIVATE draft "
            "(it stays private until uploaded again without placeholders)",
        )

    q = cmds.add_parser(
        "model", help="the repaired model, the repair parameters and the gated adapters", epilog=_EPILOG
    )
    q.add_argument(
        "--variant",
        default=None,
        help="the overall winner in selection.json, seed 1 (the default); any other variant is refused",
    )
    q.add_argument(
        "--build",
        action="store_true",
        help="materialise the weights through build_variant (GPU VM: needs the model in memory)",
    )
    q.add_argument("--device", default="cuda:0", help="device for --build (default cuda:0)")
    q.add_argument("--no-adapters", action="store_true", help="leave the gated adapters out")
    upload_flags(q)
    q.set_defaults(func=_model)

    q = cmds.add_parser(
        "datasets", help="logs (public), raw completions (gated), translated prompts (public)", epilog=_EPILOG
    )
    q.add_argument("--only", nargs="+", choices=["logs", "raw", "prompts"], default=None)
    upload_flags(q)
    q.set_defaults(func=_datasets)

    q = cmds.add_parser("space", help="the Gradio Space over the public aggregate dataset", epilog=_EPILOG)
    upload_flags(q)
    q.set_defaults(func=_space)

    q = cmds.add_parser("collections", help="the two HF collections (models, data)", epilog=_EPILOG)
    q.add_argument("--yes", action="store_true", help="create the collections for real (default: dry run)")
    q.set_defaults(func=_collections)

    q = cmds.add_parser(
        "article", help="write hf/article.md and README.md from the template and outputs/metrics"
    )
    q.set_defaults(func=_article)

    q = cmds.add_parser(
        "preflight",
        help="fail if a tracked file or the git history (every ref of origin) holds a secret, a token or "
        "a raw completion",
    )
    q.add_argument(
        "--no-history", action="store_true", help="check the tracked files only, not the git history"
    )
    q.add_argument(
        "--no-fetch",
        action="store_true",
        help="do not fetch origin's branches, tags and pull-request refs first (local refs only: not enough "
        "to make the repo public)",
    )
    q.set_defaults(func=_preflight)
