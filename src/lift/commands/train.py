"""``lift train`` and ``lift register-adapter``: the Hindi LoRA adapters (P03; the rules are in
lift.train.lora).

``lift train --candidate aya|indicalign --recipe standard|stress --seed N`` trains one adapter on a GPU
runtime, a job line after ``lift sync pull``: outputs/adapters/adapter-{candidate}-{std|stress}-s{seed}/,
pushed to the runs repo at once, checked for merge parity and registered in that machine's
configs/variants.yaml. Rerunning the line never trains again: it finishes the steps still missing
(``--retrain`` sets the folder aside first and trains anew). Exit codes: 0 done (trained now, or
finished from an earlier run); 1 an error, or fp32 merge parity failed (the adapter is saved and pushed
but not registered, an fp16 parity's registration is withdrawn, and nothing may evaluate it until the
cause is understood: CLAUDE.md rule 9), or the registry already holds another entry under the name; 2 a
refused request, before any GPU time (a candidate or seed the recipe does not allow, a per-device batch
that does not divide the effective batch, a precision the GPU must not use, a stale or malformed
training file, rows the template leaves no answer token in, a push that cannot work) or before the first
training step (a Trainer that would not take the recipe's batch: a torchrun launch, say). Unpinned
revisions stop it like lift generate (exit 1).
It never exits 3: a job runner treats any other exit as the line failing.

``lift train ... --check`` runs anywhere, the laptop included: it loads Global's tokenizer only, builds
every example as training would (lift.train.lora.check_training), prints the counts and the rows left
without a supervised token, writes nothing and exits 0 (2 when the request or the training file is
refused, 1 on unpinned revisions).

``lift register-adapter NAME [NAME ...]`` is the laptop's half: a GPU runtime's configs/ edit dies with
the session (configs/ travels by git, not by lift sync). After ``lift sync pull`` it recomputes each
adapter's hash from the pulled files, checks it against the adapter's registration.json and that parity
passed, and writes the entry the job wrote; then commit configs/variants.yaml and push it to the ref the
GPU jobs run. An adapter whose fp32 parity failed after an fp16 parity (a T4) registered it has that
entry removed instead ("removed", committed the same way): nothing may evaluate it (CLAUDE.md rule 9).
Exit 0 when every name is registered (or already was), 1 when one could not be, a removed one included.

Torch, transformers, peft and trl are imported only inside the handlers, after the checks that need
none of them, so ``lift --help`` and ``lift register-adapter`` work without the GPU stack.
"""

from __future__ import annotations

import argparse
import sys

TRAIN_HELP = """\
Train one Hindi LoRA adapter of Tiny Aya Global (configs/train.yaml; rules in lift.train.lora): chat
template on, loss on the answer tokens only, the effective batch 16 on every GPU, one process on one GPU
(cuda:0: a second visible GPU, as on a Kaggle T4 x2, stays idle). The adapter goes to
outputs/adapters/adapter-{candidate}-{std|stress}-s{seed}/ and to the runs repo at once; then merged and
unmerged greedy outputs are compared in fp32 (merge parity: every token must match, or the adapter is
not registered) and the adapter is registered in this machine's configs/variants.yaml. The one exception
is a GPU without native bf16 (a T4): there --parity-dtype fp16 records the share of prompts that match
and registers the adapter with a warning, whatever that share. A later fp32 parity of the same adapter
(the line rerun where the fp32 base fits) decides instead: if it fails, the adapter is un-registered
(its entry removed from this machine's registry) and must not be evaluated. A GPU runtime loses the
registry edit with the session: on the laptop run `lift sync pull`, then `lift register-adapter NAME`
(which also removes the entry of an adapter whose fp32 parity failed), and commit configs/variants.yaml.

Every training row is tokenized before the model loads; rows whose prompt alone (the ~366-token
preamble and a long user turn) fills max_seq_len stop the run with their ids, unless --drop-unfit drops
exactly those rows (recorded; it needs a dated docs/decisions.md entry). --check shows them first, on
any machine with transformers (the laptop too): it loads the tokenizer only (no model, no GPU, no push,
no registry), tokenizes every row as training does, prints the counts and those rows (id, prompt tokens,
limit), writes nothing, exits 0.

Rerunning the same line never trains again: it pushes, checks parity (unless already recorded for that
adapter in that precision) and registers. --retrain sets the folder aside (outputs/.aside/, never synced)
and trains anew; every output made from the old adapter then goes stale.

examples (job lines; on a T4 add --dtype fp16 --per-device-batch 1, and --parity-dtype fp16 if the fp32
parity does not fit):
  lift train --candidate aya --recipe standard --seed 1 --per-device-batch 4
  lift train --candidate indicalign --recipe stress --seed 1 --per-device-batch 4
before a GPU run (the laptop):
  lift train --candidate indicalign --recipe stress --seed 1 --check
"""

REGISTER_HELP = """\
Register adapters a GPU job trained, on the laptop after `lift sync pull`: for each NAME, recompute the
hash of outputs/adapters/NAME/, check it against the job's registration.json and that merge parity
passed, and write the entry the job wrote into configs/variants.yaml (its comments are kept). Then commit
configs/variants.yaml and push it to the ref the GPU jobs run (lift remote --ref, Kaggle BRANCH) before
any job evaluates the adapter. When an adapter's fp32 merge parity failed after an fp16 parity (a T4) had
registered it, its entry is removed instead (printed as removed, exit 1): commit that removal the same
way, since nothing may evaluate the adapter (CLAUDE.md rule 9).
"""


def _train(args: argparse.Namespace) -> int:
    from lift.commands.run import _logging

    _logging()
    if args.check:
        return _check(args)
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


def _check(args: argparse.Namespace) -> int:
    """``lift train --check``: the examples the line would train on (counts and ids only), nothing written."""
    from lift.train.lora import TrainRefused, check_lines, check_training

    try:
        report = check_training(args.candidate, args.recipe, args.seed, device=args.device, dtype=args.dtype,
                                per_device_batch=args.per_device_batch, drop_unfit=args.drop_unfit)
    except TrainRefused as exc:
        print(f"lift train --check: refused: {exc}", file=sys.stderr)
        return 2
    print("\n".join(check_lines(report)))
    return 0


def _register(args: argparse.Namespace) -> int:
    from lift.commands.run import _logging

    _logging()
    from lift.train.lora import register_pulled

    failed, changed, removed = [], [], []
    for name in dict.fromkeys(args.names):
        try:
            outcome = register_pulled(name, replace=args.replace)
        except (FileNotFoundError, ValueError) as exc:     # RegistryConflict is a ValueError
            print(f"{name}: not registered: {exc}", file=sys.stderr)
            failed.append(name)
            continue
        if outcome == "removed":            # not registered either: exit 1, and the removal is committed
            print(f"{name}: removed from configs/variants.yaml: its fp32 merge parity FAILED after an fp16 "
                  "parity had registered it, and it must not be evaluated (CLAUDE.md rule 9)",
                  file=sys.stderr)
            failed.append(name)
            removed.append(name)
            continue
        print(f"{name}: {outcome} in configs/variants.yaml")
        if outcome != "unchanged":
            changed.append(name)
    if changed or removed:
        parts = [f"register {', '.join(changed)}"] if changed else []
        if removed:
            parts.append(f"unregister {', '.join(removed)} (fp32 merge parity failed)")
        what = "; ".join(parts)
        print("now commit configs/variants.yaml and push it to the ref the GPU jobs run (lift remote --ref, "
              "Kaggle BRANCH):\n  git add configs/variants.yaml; "
              f"git commit -m \"{what[0].upper()}{what[1:]}\"; git push")
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
                   help="merge parity precision: fp32 (default: every token must match, or the adapter is "
                        "not registered) or fp16, only on a GPU without native bf16 (a T4, where the fp32 "
                        "base may not fit; refused on an A100 or L4): the share of prompts that match is "
                        "recorded and the adapter registered with a warning, whatever that share")
    p.add_argument("--no-push", action="store_true",
                   help="do not push the adapter from inside the run (the runner's pushes still send it)")
    p.add_argument("--retrain", action="store_true",
                   help="set an existing adapter folder aside (outputs/.aside/, never synced), train anew")
    p.add_argument("--drop-unfit", action="store_true",
                   help="drop exactly the rows that leave no answer token within max_seq_len (recorded; "
                        "needs a dated docs/decisions.md entry); without it such rows stop the run")
    p.add_argument("--check", action="store_true",
                   help="report what this line would train on and stop: load the tokenizer only (no model, "
                        "no GPU, no push, no registry), tokenize every row exactly as training does, print "
                        "the counts and the rows left without a supervised token (id, prompt tokens, "
                        "limit), write nothing, exit 0. Run it before a GPU run to decide on --drop-unfit")
    p.set_defaults(func=_train)

    p = sub.add_parser("register-adapter",
                       help="register adapters a GPU job trained, after lift sync pull (laptop)",
                       description=REGISTER_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="+", metavar="NAME", help="adapter names, e.g. adapter-aya-std-s1")
    p.add_argument("--replace", action="store_true",
                   help="replace another entry already registered under the name (every output made from "
                        "it goes stale: its cache keys change)")
    p.set_defaults(func=_register)
