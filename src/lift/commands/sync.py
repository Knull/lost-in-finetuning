"""``lift sync push|pull|status``: move data, outputs and validation files between machines through
the PRIVATE HF dataset repo named by LIFT_RUNS_REPO (rules in lift.sync).

Exit codes, which the job runners act on: 0 done; 3 done, but some files conflicted (push: their local
copies went under conflicts/<utc-timestamp>/ in the runs repo; pull: the local copies were kept;
status: a pull or push would conflict), so a rerun changes nothing and runners report it without
retrying; 1 an error: not everything was transferred, and a rerun is safe (sync compares content, so
it resumes). huggingface_hub is imported only when a handler runs, so ``lift --help`` works without it.
"""

from __future__ import annotations

import argparse
import sys

EXIT_OK, EXIT_ERROR, EXIT_CONFLICT = 0, 1, 3

_EXIT_CODES = ("exit codes: 0 done; 3 done, but some files conflicted (push: their local copies went under "
               "conflicts/<utc-timestamp>/ in the runs repo; pull: the local copies were kept; status: a "
               "pull or push would conflict), so do not retry; 1 error: not everything was transferred, "
               "safe to rerun")

_ONLY = ("limit the sync to this path (repeatable): data/prompts, data/sft, data/interim/translations, "
         "outputs, validation/sheets, validation/keys, validation/labels, a folder under one (e.g. "
         "outputs/generations) or a parent of some (data, validation)")

_COMMANDS = {
    "push": ("upload new and changed local files. Append-only JSONL changed on both sides is merged by "
             "key; any other file changed on both sides, or a merge that would pick between two records "
             "of one key, stays as it is on the Hub and its local copy goes to "
             "conflicts/<utc-timestamp>/<path>",
             "on a conflict, upload the local copy over the Hub's"),
    "pull": ("download new and changed files from the Hub, never overwriting a local edit. Append-only "
             "JSONL changed on both sides is merged by key; any other conflict keeps the local copy",
             "on a conflict, replace the local copy with the Hub's and keep the local one under "
             "outputs/.sync-backup/<utc-timestamp>/<path>"),
    "status": ("dry run: show what pull and push would do", None),
}


def _sync(args: argparse.Namespace) -> int:
    from lift import sync

    only = tuple(args.only)
    try:
        if args.sync_command == "status":
            reports = list(sync.status(only).values())
        elif args.sync_command == "push":
            reports = [sync.push(only, force=args.force)]
        else:
            reports = [sync.pull(only, force=args.force)]
    except sync.SyncError as e:
        print(f"lift sync {args.sync_command}: {e}", file=sys.stderr)
        return EXIT_ERROR
    for report in reports:
        print("\n".join(report.summary()), flush=True)
    return EXIT_OK if all(report.ok for report in reports) else EXIT_CONFLICT


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("sync", help="push or pull data, outputs and validation files (private runs repo)",
                       description="Move data/prompts, data/sft, data/interim/translations, outputs and "
                                   "validation/{sheets,keys,labels} between machines through the PRIVATE "
                                   "HF dataset repo named by LIFT_RUNS_REPO.",
                       epilog=_EXIT_CODES)
    commands = p.add_subparsers(dest="sync_command", required=True, metavar="<push|pull|status>")
    for name, (text, force_help) in _COMMANDS.items():
        q = commands.add_parser(name, help=text.split(".")[0], description=text, epilog=_EXIT_CODES)
        q.add_argument("--only", action="append", default=[], metavar="PREFIX", help=_ONLY)
        if force_help:
            q.add_argument("--force", action="store_true", help=force_help)
        q.set_defaults(func=_sync)
