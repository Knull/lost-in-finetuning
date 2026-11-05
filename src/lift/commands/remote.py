"""``lift remote``: run lift lines on a Colab GPU VM through the Colab CLI, then stop the VM.

``lift remote --gpu A100 --job "lift sync pull; lift smoke"`` launches a job and streams its log;
``lift remote watch NAME`` re-attaches to one (and stops its VM when it ends); ``lift remote stop NAME``
releases a VM. The logic is in lift.remote; colab/README.md has the setup and what happens to secrets.
"""

from __future__ import annotations

import argparse
import sys

EXAMPLES = """\
examples (PowerShell, repo root, venv active; --ref must hold P01 and P02, see colab/README.md):
  lift remote --gpu A100 --job "lift sync pull; lift smoke"                  # Gate 0
  #   then pin: copy the 3 shas Gate 0 printed (gate0.json, 'revisions') into configs/project.yaml,
  #   add a dated docs/decisions.md entry, commit, and push to the --ref the VM runs
  lift remote --gpu A100 --job "lift sync pull; lift run-dev --variant stock"  # stock dev eval
  lift remote --gpu A100 --job "lift sync pull; lift info" --dry-run
  lift remote watch lift-20261104-091500-ab12
  lift remote stop lift-20261104-091500-ab12
  lift remote stop lift-20261104-091500-ab12 --endpoint <a [?] endpoint from colab sessions>

Before any VM, the runs repo is checked as `lift sync check --write` does. The job pushes its outputs
after every line and once more at the end, also after a failure; a push that exits 1 is tried again,
and once more from this laptop before the VM is stopped; one that exits 2 (a problem no retry can
fix) never is. A generate, judge, run-dev, utility, belebele or heldout-loss line that exits 3 (done,
but some cells were skipped or left for a rerun, each listed with its remedy) does not stop the job:
its next lines run, and lift remote exits 1. lift train and lift gate never exit 3. The
job stops itself 30 minutes before --max-hours. Ctrl+C asks the job to stop and push, waits up to 15
minutes for it (up to 55 while its log shows a push running), then stops the VM; a stop request that
cannot be uploaded is tried again while it waits, never a reason to stop sooner. A second Ctrl+C
stops the VM at once (the outputs since the last push are then lost). While lift remote waits for
the job's final push at --max-hours, or runs its push retry, the first Ctrl+C only says that the
wait goes on. One lift remote process at a time follows a job: watch is refused while another one
does, with its pid.

"""


def _remote(args: argparse.Namespace) -> int:
    from lift import remote

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # a cp1252 console must not crash on a Hindi log line
        except (AttributeError, ValueError):
            pass
    try:
        if args.remote_action and (args.gpu or args.job):
            raise remote.RemoteError(f"lift remote {args.remote_action} takes only a session name")
        if args.remote_action == "watch":
            return remote.watch(args.name, poll=args.poll, max_hours=args.watch_max_hours,
                                stop=not args.no_stop)
        if args.remote_action == "stop":
            return remote.stop(args.name, endpoint=args.endpoint)
        if not args.gpu or not args.job:
            raise remote.RemoteError("--gpu and --job are required (or: lift remote watch|stop NAME)")
        return remote.launch(remote.Launch(
            gpu=args.gpu, job=args.job, ref=args.ref, repo=args.repo, high_mem=args.high_mem,
            max_hours=args.max_hours, poll=args.poll, name=args.name, dry_run=args.dry_run,
            max_silent_minutes=args.max_silent_minutes))
    except remote.RemoteError as e:
        print(f"lift remote: error: {e}", file=sys.stderr)
        return e.code


def register(sub: argparse._SubParsersAction) -> None:
    from lift.remote import DEFAULT_REF, DEFAULT_REPO, EXIT_CODES, GPUS, MAX_SILENT_MINUTES

    p = sub.add_parser("remote", help="run lift lines on a Colab GPU VM, stream the log, stop the VM",
                       description=__doc__.split("\n\n")[0], epilog=EXAMPLES + EXIT_CODES,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", help=f"{', '.join(GPUS)} (checked here: the Colab CLI turns an unknown value "
                                 "into an A100)")
    p.add_argument("--job", help='lift lines separated by ";" or newlines, e.g. "lift sync pull; lift smoke"')
    p.add_argument("--ref", default=DEFAULT_REF,
                   help=f"branch, tag or full commit sha for the VM to run (default {DEFAULT_REF}); push it. "
                        "It must hold both P01 and P02 (colab/README.md, 'Code version')")
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub owner/name (default {DEFAULT_REPO})")
    p.add_argument("--high-mem", action="store_true", help="ask for Colab's high-RAM machine shape")
    p.add_argument("--max-hours", type=float, default=12.0,
                   help="stop the VM after this many hours, finished or not (default 12); the job stops "
                        "itself and pushes 30 minutes before")
    p.add_argument("--max-silent-minutes", type=float, default=MAX_SILENT_MINUTES,
                   help=f"kill a job line that prints nothing for this long, as hung (default "
                        f"{MAX_SILENT_MINUTES:g})")
    p.add_argument("--poll", type=float, default=45.0, help="seconds between log downloads (default 45)")
    p.add_argument("--name", help="session name (default lift-YYYYmmdd-HHMMSS-xxxx); never reuse one")
    p.add_argument("--dry-run", action="store_true", help="print the plan with secrets masked; call nothing")
    p.set_defaults(func=_remote)
    actions = p.add_subparsers(dest="remote_action", metavar="{watch,stop}",
                               title="recovery (for a job started from this machine)")
    w = actions.add_parser("watch", help="re-attach to a job: print its log, stop its VM when it ends",
                           description="Re-attach to a job started from this machine: print its log, and "
                                       "stop its VM when it ends, after one push retry if its final push "
                                       "failed. That retry needs a WRITE token for the runs repo in "
                                       "HF_TOKEN (or `hf auth login` with a WRITE token) and LIFT_RUNS_REPO "
                                       "in this shell, not GITHUB_TOKEN, and checks the runs repo first. "
                                       "Without them, when that check fails, or when the retry's push "
                                       "exits 2, the VM is kept, with the outputs on it (exit 6). A VM "
                                       "the Colab CLI has lost gets no retry, and is stopped. "
                                       "Refused (exit 2) while another lift remote process on this "
                                       "machine follows the job: check first whether the original lift "
                                       "remote still runs.")
    w.add_argument("name", help="the session name lift remote printed")
    w.add_argument("--poll", type=float, default=argparse.SUPPRESS, help="seconds between log downloads")
    w.add_argument("--max-hours", dest="watch_max_hours", type=float, default=None,
                   help="stop the VM after this many more hours (default: the launch's cap)")
    w.add_argument("--no-stop", action="store_true", help="leave the VM running when the job ends")
    s = actions.add_parser("stop", help="stop a job's VM and check it is gone (also when the CLI lost it)")
    s.add_argument("name", help="the session name lift remote printed")
    s.add_argument("--endpoint", help="the VM's endpoint, for a VM the CLI has no record of: a [?] line of "
                                      "`colab sessions`, as the alarm lists it")
