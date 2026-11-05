"""``lift train`` and ``lift register-adapter``: the Hindi LoRA adapters (P03; the rules are in
lift.train.lora).

``lift train --candidate aya|indicalign --recipe standard|stress --seed N`` trains one adapter on a GPU
runtime, a job line after ``lift sync pull``: outputs/adapters/adapter-{candidate}-{std|stress}-s{seed}/,
pushed to the runs repo at once, checked for merge parity and registered in that machine's
configs/variants.yaml. Rerunning the line never trains again: it finishes the steps still missing
(``--retrain`` sets the folder aside first and trains anew). Exit codes: 0 done (trained now, or
finished from an earlier run); 1 an error, or fp32 merge parity failed (the adapter is saved and pushed
but not registered, and nothing may evaluate it until the cause is understood: CLAUDE.md rule 9), or the
registry already holds another entry under the name; 2 a refused request, before any GPU time (a
candidate or seed the recipe does not allow, a per-device batch that does not divide the effective
batch, a precision the GPU must not use, a stale or malformed training file, rows the template leaves no
answer token in, a push that cannot work). Unpinned revisions stop it like lift generate (exit 1). It
never exits 3: a job runner treats any other exit as the line failing.

``lift register-adapter NAME [NAME ...]`` is the laptop's half: a GPU runtime's configs/ edit dies with
the session (configs/ travels by git, not by lift sync). After ``lift sync pull`` it recomputes each
adapter's hash from the pulled files, checks it against the adapter's registration.json and that parity
passed, and writes the entry the job wrote; then commit configs/variants.yaml and push it to the ref the
GPU jobs run. Exit 0 when every name is registered (or already was), 1 when one could not be.

Torch, transformers, peft and trl are imported only inside the handlers, after the checks that need
none of them, so ``lift --help`` and ``lift register-adapter`` work without the GPU stack.
"""

from __future__ import annotations

import argparse
import sys

TRAIN_HELP = """\
Train one Hindi LoRA adapter of Tiny Aya Global (configs/train.yaml; rules in lift.train.lora): chat
template on, loss on the answer tokens only, the effective batch 16 on every GPU. The adapter goes to
outputs/adapters/adapter-{candidate}-{std|stress}-s{seed}/ and to the runs repo at once; then merged and
unmerged greedy outputs are compared in fp32 (merge parity) and the adapter is registered in this
machine's configs/variants.yaml. A GPU runtime loses that edit with the session: on the laptop run
`lift sync pull`, then `lift register-adapter NAME`, and commit configs/variants.yaml.

Every training row is tokenized before the model loads; rows whose prompt alone (the ~366-token
preamble and a long user turn) fills max_seq_len stop the run with their ids, unless --drop-unfit drops
exactly those rows (recorded; it needs a dated docs/decisions.md entry).

Rerunning the same line never trains again: it pushes, checks parity (unless already recorded for that
adapter in that precision) and registers. --retrain sets the folder aside (outputs/.aside/, never synced)
and trains anew; every output made from the old adapter then goes stale.

examples (job lines; on a T4 add --dtype fp16 --per-device-batch 1, and --parity-dtype fp16 if the fp32
parity does not fit):
  lift train --candidate aya --recipe standard --seed 1 --per-device-batch 4
  lift train --candidate indicalign --recipe stress --seed 1 --per-device-batch 4
"""

REGISTER_HELP = """\
Register adapters a GPU job trained, on the laptop after `lift sync pull`: for each NAME, recompute the
hash of outputs/adapters/NAME/, check it against the job's registration.json and that merge parity
passed, and write the entry the job wrote into configs/variants.yaml (its comments are kept). Then commit
configs/variants.yaml and push it to the ref the GPU jobs run (lift remote --ref, Kaggle BRANCH) before
any job evaluates the adapter.
"""


def _train(args: argparse.Namespace) -> int:
    from lift.commands.run import _logging

    _logging()
    from lift.train.lora import ParityFailed, RegistryConflict, TrainRefused, summary, train_lora

    try:
        path = train_lora(args.candidate, args.recipe, args.seed, device=args.device, dtype=args.dtype,
                          per_device_batch=args.per_device_batch, parity_dtype=args.parity_dtype,
                          push=not args.no_push, retrain=args.retrain, drop_unfit=args.drop_unfit)
    except TrainRefused as exc:
        print(f"lift train: refused: {exc}", file=sys.stderr)
        return 2
    except (ParityFailed, RegistryConflict) as exc:
        print(f"lift train: {exc}", file=sys.stderr)
        return 1
    print("\n".join(summary(path)))
    return 0


def _register(args: argparse.Namespace) -> int:
    from lift.commands.run import _logging

    _logging()
    from lift.train.lora import register_pulled

    failed, changed = [], []
    for name in dict.fromkeys(args.names):
        try:
            outcome = register_pulled(name, replace=args.replace)
        except (FileNotFoundError, ValueError) as exc:     # RegistryConflict is a ValueError
            print(f"{name}: not registered: {exc}", file=sys.stderr)
            failed.append(name)
            continue
        print(f"{name}: {outcome} in configs/variants.yaml")
        if outcome != "unchanged":
            changed.append(name)
    if changed:
        print("now commit configs/variants.yaml and push it to the ref the GPU jobs run (lift remote --ref, "
              "Kaggle BRANCH):\n  git add configs/variants.yaml; "
              f"git commit -m \"Register {', '.join(changed)}\"; git push")
    return 1 if failed else 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="train one Hindi LoRA adapter, push it, check merge parity, register "
                                     "it (GPU)",
                       description=TRAIN_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", required=True, help="a candidate of configs/train.yaml (aya, indicalign)")
    p.add_argument("--recipe", required=True, choices=["standard", "stress"],
                   help="standard (the std adapters) or stress (indicalign only)")
    p.add_argument("--seed", required=True, type=int,
                   help="one of the recipe's seeds (standard 1 2 3, stress 1)")
    p.add_argument("--device", default="cuda:0", help="cuda:0 (default), or cpu for the tests' tiny model")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None,
                   help="default: configs/train.yaml precision (bf16); a T4 takes fp16")
    p.add_argument("--per-device-batch", type=int, default=None,
                   help="sequences per micro-batch; must divide the effective batch 16, and gradient "
                        "accumulation makes up the rest (A100: 4; T4: 1). Default: configs/train.yaml")
    p.add_argument("--parity-dtype", choices=["fp32", "fp16"], default="fp32",
                   help="merge parity precision: fp32 (must match token for token) or fp16 (a T4 only: "
                        "the share that match is recorded and the adapter registered with a warning)")
    p.add_argument("--no-push", action="store_true",
                   help="do not push the adapter from inside the run (the runner's pushes still send it)")
    p.add_argument("--retrain", action="store_true",
                   help="set an existing adapter folder aside (outputs/.aside/, never synced), train anew")
    p.add_argument("--drop-unfit", action="store_true",
                   help="drop exactly the rows that leave no answer token within max_seq_len (recorded; "
                        "needs a dated docs/decisions.md entry); without it such rows stop the run")
    p.set_defaults(func=_train)

    p = sub.add_parser("register-adapter",
                       help="register adapters a GPU job trained, after lift sync pull (laptop)",
                       description=REGISTER_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="+", metavar="NAME", help="adapter names, e.g. adapter-aya-std-s1")
    p.add_argument("--replace", action="store_true",
                   help="replace another entry already registered under the name (every output made from "
                        "it goes stale: its cache keys change)")
    p.set_defaults(func=_register)
