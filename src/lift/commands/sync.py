"""``lift sync push|pull|status|check``: move data, outputs and validation files between machines
through the PRIVATE HF dataset repo named by LIFT_RUNS_REPO (rules in lift.sync).

Exit codes, which the job runners act on: 0 done; 3 done, but some files conflicted (push: their local
copies went under conflicts/<utc-timestamp>/ in the runs repo; pull: the local copies were kept;
status: a pull or push would conflict), or judgements files or AuditKIT records here or on the Hub do
not pair with their cell's generations (the summary names the line that settles them: the lift restart
line that sets them aside, or the lift sync push that sends the copies here when those pair), or a
pull could not check them because --only left them, or the files they were made from, out (pull
without --only), so a rerun changes nothing and runners report it without retrying; 1 a transient or unknown
error: not everything was transferred, and a rerun is safe (sync compares content, so it resumes); 2 a
problem no rerun fixes (SyncError.permanent: a token missing, rejected or not allowed, the runs repo
missing or not private, LIFT_RUNS_REPO unset or malformed, HF_DEBUG set, a bad path; argparse's own
usage errors exit 2 too), so runners stop the job instead of retrying. No message shows
LIFT_RUNS_REPO's value before the Hub has answered for it as a repo.

``lift sync check [--write]`` runs lift.sync.preflight before a job spends GPU time: 0 ok, 1 a
transient error, 2 a problem no rerun fixes. huggingface_hub is imported only when a handler runs, so
``lift --help`` works without it.
"""

from __future__ import annotations

import argparse
import sys

EXIT_OK, EXIT_ERROR, EXIT_PERMANENT, EXIT_CONFLICT = 0, 1, 2, 3

_EXIT_CODES = ("exit codes: 0 done; 3 done, but some files conflicted (push: their local copies went under "
               "conflicts/<utc-timestamp>/ in the runs repo; pull: the local copies were kept; status: a "
               "pull or push would conflict), or judgements or AuditKIT records do not pair with their "
               "cell's generations (the summary names the line that settles them: a lift restart line, or "
               "a lift sync push that sends the copies here), or a pull could not check them because --only "
               "left them, or what they were made from, out, so do not retry; 1 a transient or unknown "
               "error: not everything was transferred, safe to rerun; 2 "
               "a problem no rerun fixes (a token missing, rejected or not allowed, the runs repo missing or "
               "not private, LIFT_RUNS_REPO unset or malformed, HF_DEBUG set, a bad argument): fix it, do "
               "not retry")

_CHECK_EXIT_CODES = ("exit codes: 0 ok; 1 a transient Hub or network error, safe to rerun; 2 a problem no "
                     "rerun fixes (as for push and pull): fix it before starting a job")

_ONLY = ("limit the sync to this path (repeatable): data/prompts, data/sft, data/interim/translations, "
         "outputs, validation/sheets, validation/keys, validation/labels, a folder under one (e.g. "
         "outputs/generations) or a parent of some (data, validation). A pull does not check judgements "
         "or AuditKIT records against generations it left out that differ from the Hub's (it says so)")

_COMMANDS = {
    "push": ("upload new and changed local files. Append-only JSONL changed on both sides is merged by "
             "key; any other file changed on both sides, or a merge that would pick between two records "
             "of one key, stays as it is on the Hub and its local copy goes to "
             "conflicts/<utc-timestamp>/<path>. Judgements and AuditKIT records go up only with the "
             "generations they pair with, and so does a merge of them",
             "on a conflict, upload the local copy over the Hub's (never judgements or AuditKIT records "
             "that do not pair with their cell's generations)"),
    "pull": ("download new and changed files from the Hub, never overwriting a local edit. Append-only "
             "JSONL changed on both sides is merged by key; any other conflict keeps the local copy, and so "
             "do the judgements and AuditKIT records of a cell whose generations conflict",
             "on a conflict, replace the local copy with the Hub's and keep the local one under "
             "outputs/.sync-backup/<utc-timestamp>/<path>"),
    "status": ("dry run: show what pull and push would do", None),
}

_CHECK = ("check, before a job spends GPU time, what every sync needs: LIFT_RUNS_REPO and a token set, "
          "HF_DEBUG unset, the runs repo there and private. Moves nothing")


def _exit_code(error: Exception) -> int:
    return EXIT_PERMANENT if getattr(error, "permanent", False) else EXIT_ERROR


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
        return _exit_code(e)
    for report in reports:
        print("\n".join(report.summary()), flush=True)
    return EXIT_OK if all(report.ok for report in reports) else EXIT_CONFLICT


def _check(args: argparse.Namespace) -> int:
    from lift import sync

    try:
        sync.preflight(write=args.write)
    except sync.SyncError as e:
        print(f"lift sync check: {e}", file=sys.stderr)
        return _exit_code(e)
    may = "read it, and whoami shows nothing that stops a push" if args.write else "read it"
    print(f"lift sync check: {sync.runs_repo()} is private and this token can {may}", flush=True)
    return EXIT_OK


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("sync", help="push or pull data, outputs and validation files (private runs repo)",
                       description="Move data/prompts, data/sft, data/interim/translations, outputs and "
                                   "validation/{sheets,keys,labels} between machines through the PRIVATE "
                                   "HF dataset repo named by LIFT_RUNS_REPO.",
                       epilog=_EXIT_CODES)
    commands = p.add_subparsers(dest="sync_command", required=True, metavar="<push|pull|status|check>")
    for name, (text, force_help) in _COMMANDS.items():
        q = commands.add_parser(name, help=text.split(".")[0], description=text, epilog=_EXIT_CODES)
        q.add_argument("--only", action="append", default=[], metavar="PREFIX", help=_ONLY)
        if force_help:
            q.add_argument("--force", action="store_true", help=force_help)
        q.set_defaults(func=_sync)
    q = commands.add_parser("check", help=_CHECK.split(":")[0], description=_CHECK, epilog=_CHECK_EXIT_CODES)
    q.add_argument("--write", action="store_true",
                   help="also check that the token may write the runs repo (HfApi.whoami: a read token, or a "
                        "fine-grained one without repo.write on it or its owner, is refused)")
    q.set_defaults(func=_check)
