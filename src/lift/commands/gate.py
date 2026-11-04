"""``lift gate 1`` and ``lift gate 2 [--candidate NAME]``: Gates 1 and 2 from the dev metrics (P03).

Laptop side, after ``lift sync pull``; nothing here imports torch. Gate 1 (judge and gain) reads P04's
outputs/metrics/validation/gate1.json and each seed-1 standard adapter's dev utility metrics; Gate 2
(drift) reads the Hindi dev harmful pairs of stock and each seed-1 standard adapter, and of the stress
adapter as a fact, through lift.judge.pipeline.paired_labels. Each prints the README rule, every check
and adapter with its numbers, and the verdict, and writes outputs/metrics/gate{1,2}.json and .md
(lift.gates has the rules). Output names ids, paths, counts and rates only.

``--candidate`` (Gate 2 only) is the candidate a person picked at Gate 1, by dev utility win rate only,
never by drift; without it every candidate's branch is listed. Nothing here picks a candidate.

Exit codes: 0 when the report is written, whatever the verdict (PASS, FAIL, PENDING and INCOMPLETE are
outcomes, not errors: both branches of a gate are legitimate, and a job line must not stop on one);
1 when the report cannot be written; 2 for a refused invocation (--candidate with Gate 1, or a name
that is not a configs/train.yaml candidate). Never 3: a gate skips no cell, it reports it.
"""

from __future__ import annotations

import argparse
import logging
import sys

DESCRIPTION = (
    "Gate 1 (judge and gain): P04's validation round 1 verdict in Hindi and each seed-1 standard adapter's "
    "dev utility win rate against the 0.55 bar. Gate 2 (drift): each seed-1 standard adapter's Hindi dev "
    "harmful rate against stock (at least 8 points, McNemar p < 0.05 on 172 prompts), with the stress "
    "adapter as a fact, and the branch the rule gives. Writes outputs/metrics/gate{1,2}.json and .md and "
    "exits 0 whatever the verdict. Runs on the laptop after lift sync pull; no GPU, no torch.")


def _setup() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Paths and ids are printed; a console in a legacy code page must not turn them into a crash.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")


def _refuse(message: str) -> int:
    print(f"lift gate: error: {message}", file=sys.stderr)
    return 2


def _gate(args: argparse.Namespace) -> int:
    _setup()
    from lift import gates  # noqa: PLC0415

    if args.number == 1 and args.candidate is not None:
        return _refuse("--candidate is for Gate 2 only: Gate 1 lists every candidate, and a person picks "
                       "from it")
    names = gates.candidates()
    if args.candidate is not None and args.candidate not in names:
        return _refuse(f"unknown candidate {args.candidate!r}; choose from configs/train.yaml candidates "
                       f"{names}")
    result = gates.gate1() if args.number == 1 else gates.gate2(args.candidate)
    for line in gates.summary_lines(result):
        print(line)
    try:
        json_path, md_path = gates.write_report(result)
    except OSError as exc:
        print(f"error: the Gate {args.number} report cannot be written ({type(exc).__name__}: {exc}). If a "
              "report file is open in another program, close it and run the command again.", file=sys.stderr)
        return 1
    print(f"\nwrote {json_path} and {md_path}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("gate", help="Gate 1 (judge and gain) or Gate 2 (drift) from the dev metrics "
                                    "(laptop, no GPU)", description=DESCRIPTION)
    p.add_argument("number", type=int, choices=(1, 2), metavar="{1,2}",
                   help="1: judge and gain; 2: drift")
    p.add_argument("--candidate", default=None, metavar="NAME",
                   help="Gate 2 only: the candidate picked at Gate 1 (a configs/train.yaml candidate, e.g. "
                        "aya or indicalign); without it, every candidate's branch is listed")
    p.set_defaults(func=_gate)
