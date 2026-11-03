"""``lift smoke|generate|judge|run-dev``: Gate 0 and the generate -> judge harness (GPU runtimes).

Torch and transformers are imported inside the handlers, so ``lift --help`` works without them.
Devices: the generator takes ``--device`` (cuda:0). The judge takes ``--judge-device``; "auto" means
cuda:1 when there are two GPUs (Kaggle T4 x2: two separate 16 GB devices) and cuda:0 otherwise
(Colab A100: both models share the card, about 7 GB for the generator and 9 GB for the judge in bf16).
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from lift.prompts import SAFETY_SETS
from lift.schemas import LANGS

log = logging.getLogger("lift.run")

ALL_SETS = ("harmful", "easy_benign", "hard_benign", "redteam", "utility")
JUDGE_GB = 9.0            # Qwen3Guard-Gen-4B in bf16/fp16 (8.8 GB) with a little headroom
JUDGE_KV_HEADROOM_GB = 4.0


def _logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _names(values: Sequence[str] | None, allowed: Sequence[str], default: Sequence[str]) -> list[str]:
    """``--langs hi en`` or ``--langs hi,en``; checked against ``allowed``."""
    out = [v for item in (values or default) for v in str(item).split(",") if v]
    bad = sorted(set(out) - set(allowed))
    if bad:
        raise SystemExit(f"unknown {bad}; choose from {list(allowed)}")
    return list(dict.fromkeys(out))


def _judge_device(value: str) -> str:
    if value != "auto":
        return value
    import torch

    return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"


def _dtype(value: str | None) -> str:
    from lift import config

    return value or config.get("generation.dtype")


def require_pinned(*keys: str) -> None:
    """Refuse to produce study outputs from an unpinned revision ("main" moves: stock and an adapter
    could sit on different commits). Gate 0 resolves and prints the shas to pin."""
    import re

    from lift import config

    loose = [f"{k}.revision = {config.get(k + '.revision')!r}" for k in keys
             if not re.fullmatch(r"[0-9a-f]{40}", str(config.get(k + ".revision")))]
    if loose:
        raise SystemExit("unpinned revisions in configs/project.yaml: " + "; ".join(loose) + ". Pin each to "
                         "the commit sha Gate 0 printed (outputs/metrics/gate0.json, 'revisions'), with a "
                         "dated docs/decisions.md entry, before generating or judging.")


def _complete(variant: str, set_name: str, lang: str, split: str) -> bool:
    """True if every prompt of the split has a generation (the AuditKIT record needs all of them)."""
    from lift.models.generate import generations_path, read_generations
    from lift.prompts import load

    path = generations_path(variant, set_name, lang, split)
    if not path.exists():
        return False
    return {g.pid for g in read_generations(path)} >= {p.pid for p in load(set_name, lang, split)}


def _record(variant: str, set_name: str, lang: str, split: str, *, dtype: str, refused: list[str]) -> None:
    """Write the AuditKIT record of a judged cell. A record the checked loader refuses (another
    precision, a stale judgement) is logged and collected in ``refused`` rather than stopping the run:
    the judgements are already written, and the command exits non-zero at the end."""
    from lift.audit import record_safety

    if not _complete(variant, set_name, lang, split):
        log.info("%s %s_%s_%s: generations do not cover every prompt yet; no AuditKIT record",
                 variant, set_name, lang, split)
        return
    try:
        log.info("AuditKIT record: %s", record_safety(variant, set_name, lang, split, dtype=dtype))
    except ValueError as exc:
        log.error("%s %s_%s_%s: no AuditKIT record: %s", variant, set_name, lang, split, exc)
        refused.append(f"{set_name}_{lang}_{split}")


def _refused_exit(refused: list[str]) -> int:
    if refused:
        log.error("%d AuditKIT records were refused (see the errors above): %s", len(refused), refused)
        return 1
    return 0


# --- lift smoke ---------------------------------------------------------------------------------------


def _smoke(args: argparse.Namespace) -> int:
    _logging()
    import torch

    if not torch.cuda.is_available():
        print("lift smoke is Gate 0 and needs a CUDA GPU", file=sys.stderr)
        return 2
    from lift.gate0 import CHECKS, pin_hint, report_name, run_gate0, write_report
    from lift.manifest import finish_run, start_run

    skip = _names(args.skip, CHECKS, ()) if args.skip else []
    sizes = [int(s) for s in _names(args.batch_sizes, [str(n) for n in range(1, 513)], ())]
    judge_device = _judge_device(args.judge_device)
    devices = len({args.device, judge_device})
    run = start_run("smoke", device=args.device, judge_device=judge_device, compare_dtype=args.compare_dtype,
                    skip=skip, devices=devices)
    try:
        result = run_gate0(device=args.device, judge_device=judge_device, compare_dtype=args.compare_dtype,
                           skip=skip, n_precision=args.n_precision, n_dry=args.n_dry, batch_sizes=sizes)
    except ValueError as exc:            # a refused configuration (e.g. --compare-dtype on an A100)
        finish_run(run, status="refused", error=str(exc))
        print(f"lift smoke: {exc}", file=sys.stderr)
        return 2
    except BaseException as exc:
        finish_run(run, status="failed", error=type(exc).__name__)
        raise
    json_path, md_path = write_report(result, name=report_name(result))
    finish_run(run, status="ok", passed=result["pass"], incomplete=result["incomplete"],
               check_seconds={k: c.get("seconds") for k, c in result["checks"].items()})
    for check, c in result["checks"].items():
        error = f"  ({c['error']})" if c.get("error") else ""
        print(f"{'PASS' if c.get('pass') else 'FAIL'}  {check}{error}")
    print(f"\n{'PASS' if result['pass'] else 'FAIL'}: {result['decision']}")
    print(f"\n{pin_hint(result)}\n\nwrote {json_path} and {md_path}")
    return 0 if result["pass"] else 1


# --- lift generate ------------------------------------------------------------------------------------


def _generate(args: argparse.Namespace) -> int:
    _logging()
    from lift.models.generate import generate, generations_path
    from lift.models.variants import build_variant, get_spec
    from lift.prompts import load

    require_pinned("models.global")
    sets = _names(args.sets, ALL_SETS, SAFETY_SETS)
    langs = _names(args.langs, LANGS, LANGS)
    spec = get_spec(args.variant)
    variant = build_variant(spec, device=args.device, dtype=_dtype(args.dtype))
    for set_name in sets:
        for lang in langs:
            prompts = load(set_name, lang, args.split)
            if args.limit:
                prompts = prompts[: args.limit]
            if not prompts:
                log.warning("%s_%s has no %s prompts", set_name, lang, args.split)
                continue
            generate(variant, prompts, out_path=generations_path(spec.name, set_name, lang, args.split),
                     batch_size=args.batch_size)
    return 0


# --- lift judge ---------------------------------------------------------------------------------------


def _judge(args: argparse.Namespace) -> int:
    _logging()
    from lift.judge.pipeline import judge_dir, judge_file, make_judge, read_keys_file
    from lift.models.generate import generations_path, read_generations
    from lift.paths import generations_dir

    if (args.sample is not None or args.keys) and args.judge != "second":
        raise SystemExit("--sample and --keys select a subset, which only the second judge may judge; "
                         "the primary judge covers every generation")
    if args.sample is not None and args.keys:
        raise SystemExit("pass either --sample or --keys, not both")
    sets = _names(args.sets, ALL_SETS, SAFETY_SETS)
    langs = _names(args.langs, LANGS, LANGS)
    paths = {(s, lang): generations_path(args.variant, s, lang, args.split) for s in sets for lang in langs}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} generations files are missing, e.g. {missing[:3]}; "
                         "run lift generate first")
    keys = read_keys_file(Path(args.keys)) if args.keys else None
    local = judge_dir(args.judge) == "qwen3guard"
    if local:
        require_pinned(_judge_entry(args.judge))
    if keys is not None:
        # A listed key that is in none of this variant's generations files for the split was regenerated
        # under a new key (a fixed prompt) or is a typo: its item would silently lose its label.
        files = (generations_dir() / args.variant).glob(f"*_{args.split}.jsonl")
        known = {g.key for path in files for g in read_generations(path)}
        unknown = sorted(set(keys) - known)
        if unknown and not args.allow_missing:
            raise SystemExit(f"{len(unknown)} listed keys are in none of {args.variant}'s {args.split} "
                             f"generations files, e.g. {[k[:12] for k in unknown[:5]]}; regenerate the key "
                             "list (lift validate second-judge-sample) or pass --allow-missing")
        if unknown:
            log.warning("%d listed keys are in no %s %s generations file", len(unknown), args.variant,
                        args.split)
    judge = make_judge(args.judge, device=_judge_device(args.device) if local else None,
                       dtype=_dtype(args.dtype) if local else None)
    refused: list[str] = []
    for (set_name, lang), path in paths.items():
        judge_file(path, judge=judge, sample=args.sample, keys=keys)
        if args.judge == "primary":
            _record(args.variant, set_name, lang, args.split, dtype=_dtype(args.dtype), refused=refused)
    return _refused_exit(refused)


def _judge_entry(role: str) -> str:
    return {"primary": "judges.safety_primary", "second": "judges.safety_second"}[role]


# --- lift run-dev -------------------------------------------------------------------------------------


def run_cells(holder: dict[str, Any], make_judge: Callable[[], Any], cells: Sequence[tuple[str, str]], *,
              split: str = "dev", concurrent: bool, interleave: bool = True, limit: int | None = None,
              batch_size: int | None = None, lid: Any = None,
              record: Callable[[str, str], None] | None = None) -> list[Path]:
    """Generate then judge every (set, lang) cell; returns the judgements paths.

    ``holder["variant"]`` is the only reference to the generator, so it can really be freed.
    ``concurrent``: the judge runs in its own thread on its own GPU and judges each file as soon as
    it is written. Otherwise both share one device: with ``interleave`` each file is judged right
    after it is generated (both models loaded); without, every file is generated first, the
    generator is dropped and its memory released, then the judge is loaded and judges them all.
    """
    from lift.judge.pipeline import judge_file
    from lift.models.generate import generate, generations_path
    from lift.prompts import load

    name = holder["variant"].spec.name
    done: list[Path] = []

    def gen(set_name: str, lang: str) -> Path | None:
        prompts = load(set_name, lang, split)[:limit] if limit else load(set_name, lang, split)
        if not prompts:
            log.warning("%s_%s has no %s prompts", set_name, lang, split)
            return None
        return generate(holder["variant"], prompts, out_path=generations_path(name, set_name, lang, split),
                        batch_size=batch_size)

    def jud(judge: Any, path: Path, set_name: str, lang: str) -> None:
        done.append(judge_file(path, judge=judge, lid=lid))
        if record is not None:
            record(set_name, lang)

    if concurrent:
        work: queue.Queue = queue.Queue()
        failure: list[BaseException] = []

        def worker() -> None:
            try:
                judge = make_judge()
                while (item := work.get()) is not None:
                    jud(judge, *item)
            except BaseException as exc:  # logged here, re-raised (or chained) by the main thread
                log.exception("the judge thread failed")
                failure.append(exc)

        thread = threading.Thread(target=worker, name="lift-judge", daemon=True)
        thread.start()
        try:
            for set_name, lang in cells:
                if failure:
                    break
                path = gen(set_name, lang)
                if path is not None:
                    work.put((path, set_name, lang))
        except BaseException as exc:
            work.put(None)
            thread.join()
            if failure:                    # both sides failed: neither cause may be lost
                raise exc from failure[0]
            raise
        work.put(None)
        thread.join()
        if failure:
            raise failure[0]
        return done

    if interleave:
        judge = make_judge()
        for set_name, lang in cells:
            path = gen(set_name, lang)
            if path is not None:
                jud(judge, path, set_name, lang)
        return done

    written = [(gen(s, lang), s, lang) for s, lang in cells]
    holder.clear()
    _release_memory()
    judge = make_judge()
    for path, set_name, lang in written:
        if path is not None:
            jud(judge, path, set_name, lang)
    return done


def _release_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _free_gb(device: str) -> float:
    import torch

    free, _total = torch.cuda.mem_get_info(torch.device(device))
    return free / 1e9


def _run_dev(args: argparse.Namespace) -> int:
    _logging()
    import torch

    from lift.judge.pipeline import make_judge
    from lift.models.variants import build_variant, get_spec

    require_pinned("models.global", "judges.safety_primary")
    langs = _names(args.langs, LANGS, LANGS)
    sets = _names(args.sets, SAFETY_SETS, SAFETY_SETS)
    dtype = _dtype(args.dtype)
    gen_device, judge_device = args.device, _judge_device(args.judge_device)
    # The holder is the only reference to the generator, so the one-GPU path can free it.
    holder = {"variant": build_variant(get_spec(args.variant), device=gen_device, dtype=dtype)}
    concurrent = gen_device != judge_device
    interleave = True
    if not concurrent and gen_device.startswith("cuda") and torch.cuda.is_available():
        free = _free_gb(gen_device)
        interleave = free >= JUDGE_GB + JUDGE_KV_HEADROOM_GB
        log.info("one GPU: %.1f GB free after loading the generator; %s", free,
                 "judge loaded alongside" if interleave else "judging after all generation, generator freed")

    cells = [(s, lang) for s in sets for lang in langs]
    refused: list[str] = []

    def record(set_name: str, lang: str) -> None:
        _record(args.variant, set_name, lang, "dev", dtype=dtype, refused=refused)

    run_cells(holder, lambda: make_judge("primary", device=judge_device, dtype=dtype), cells,
              concurrent=concurrent, interleave=interleave, limit=args.limit, batch_size=args.batch_size,
              record=None if args.limit else record)
    return _refused_exit(refused)


# --- registration -------------------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("smoke", help="Gate 0: precision, throughput, dry run, preamble, merge parity (GPU)")
    p.add_argument("--device", default="cuda:0", help="generator device (default cuda:0)")
    p.add_argument("--judge-device", default="auto",
                   help="judge device; auto = cuda:1 if two GPUs, else cuda:0")
    p.add_argument("--compare-dtype", choices=["fp16"], default=None,
                   help="run in fp16 (a T4) and compare with the stored bf16 run from the runs repo")
    p.add_argument("--skip", nargs="+", default=None, metavar="CHECK",
                   help="checks to skip: preamble precision fp16 throughput dry_run parity")
    p.add_argument("--n-precision", type=int, default=200, help="prompts in the precision check (200)")
    p.add_argument("--n-dry", type=int, default=20, help="prompts per language and set in the dry run (20)")
    p.add_argument("--batch-sizes", nargs="+", default=["8", "16", "32"], help="throughput batch sizes")
    p.set_defaults(func=_smoke)

    p = sub.add_parser("generate", help="generate one variant's outputs for prompt sets and languages")
    p.add_argument("--variant", required=True, help="a name from configs/variants.yaml")
    p.add_argument("--sets", nargs="+", default=None, help=f"default: {' '.join(SAFETY_SETS)}")
    p.add_argument("--langs", nargs="+", default=None, help="default: all six")
    p.add_argument("--split", required=True, choices=["dev", "final"])
    p.add_argument("--limit", type=int, default=None, help="first N prompts per set and language (by pid)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None,
                   help="default: generation.dtype")
    p.add_argument("--batch-size", type=int, default=None, help="default: chosen from free memory")
    p.set_defaults(func=_generate)

    p = sub.add_parser("judge", help="judge one variant's generations (every one, or a second-judge subset)")
    p.add_argument("--variant", required=True)
    p.add_argument("--sets", nargs="+", default=None, help=f"default: {' '.join(SAFETY_SETS)}")
    p.add_argument("--langs", nargs="+", default=None, help="default: all six")
    p.add_argument("--split", required=True, choices=["dev", "final"])
    p.add_argument("--judge", choices=["primary", "second"], default="primary")
    p.add_argument("--sample", type=float, default=None,
                   help="second judge: a seeded random fraction (e.g. 0.10) plus every validated item")
    p.add_argument("--keys", default=None,
                   help="second judge: judge exactly the generation keys listed in this file")
    p.add_argument("--allow-missing", action="store_true",
                   help="with --keys: go on when listed keys are in none of the variant's generations files")
    p.add_argument("--device", default="auto", help="judge device; auto = cuda:1 if two GPUs, else cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None,
                   help="default: generation.dtype")
    p.set_defaults(func=_judge)

    p = sub.add_parser("run-dev", help="generate, then judge, every dev safety set in all six languages")
    p.add_argument("--variant", required=True)
    p.add_argument("--langs", nargs="+", default=None, help="default: all six")
    p.add_argument("--sets", nargs="+", default=None, help=f"default: {' '.join(SAFETY_SETS)}")
    p.add_argument("--limit", type=int, default=None, help="first N prompts per set and language (no record)")
    p.add_argument("--device", default="cuda:0", help="generator device")
    p.add_argument("--judge-device", default="auto", help="auto = cuda:1 if two GPUs, else cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default=None, help="default: generation.dtype")
    p.add_argument("--batch-size", type=int, default=None)
    p.set_defaults(func=_run_dev)
