"""``lift smoke|generate|judge|run-dev|restart``: Gate 0 and the generate -> judge harness (GPU runtimes).

Torch and transformers are imported inside the handlers, so ``lift --help`` works without them, and
only after the checks that need neither (pinned revisions, names), so a laptop gets their messages.
Devices: the generator takes ``--device`` (cuda:0). The judge takes ``--judge-device``; "auto" means
cuda:1 when there are two GPUs (Kaggle T4 x2: two separate 16 GB devices) and cuda:0 otherwise
(Colab A100: both models share the card, about 7 GB for the generator and 9 GB for the judge in bf16).

A cell refused as stale (StaleOutputError: begun under another stack, or made from other inputs) is
logged and skipped, and the other cells go on; the command then exits 1 listing each such cell with
the ``lift restart`` line that starts it over, a job line right after ``lift sync pull``.
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
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
    could sit on different commits). Gate 0 resolves and prints the shas to pin. The GPU job reads the
    configs/project.yaml of the ref it runs, so a pin counts there only once it is committed and pushed."""
    import re

    from lift import config

    loose = [f"{k}.revision = {config.get(k + '.revision')!r}" for k in keys
             if not re.fullmatch(r"[0-9a-f]{40}", str(config.get(k + ".revision")))]
    if loose:
        raise SystemExit("unpinned revisions in configs/project.yaml: " + "; ".join(loose) + ". Pin each to "
                         "the commit sha Gate 0 printed (outputs/metrics/gate0.json, 'revisions') with a "
                         "dated docs/decisions.md entry, then commit the pin and push it to the ref the GPU "
                         "job runs (lift remote --ref, Kaggle BRANCH), before generating or judging: the "
                         "job reads that ref's configs/project.yaml, not this working tree.")


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
    from lift.models.generate import StaleOutputError

    if not _complete(variant, set_name, lang, split):
        log.info("%s %s_%s_%s: generations do not cover every prompt yet; no AuditKIT record",
                 variant, set_name, lang, split)
        return
    try:
        log.info("AuditKIT record: %s", record_safety(variant, set_name, lang, split, dtype=dtype))
    except (ValueError, StaleOutputError) as exc:
        log.error("%s %s_%s_%s: no AuditKIT record: %s", variant, set_name, lang, split, exc)
        refused.append(f"{set_name}_{lang}_{split}")


def _stale(stale: dict[str, str], variant: str, set_name: str, lang: str, split: str, exc: Exception) -> None:
    """Log a cell refused as stale and keep the ``lift restart`` line its error names (``command``),
    so the command goes on with the other cells and lists it at the end."""
    log.error("%s %s_%s_%s: stale, skipped: %s", variant, set_name, lang, split, exc)
    stale[f"{set_name}_{lang}_{split}"] = (getattr(exc, "command", None)
                                           or "no lift restart line covers it (see its error above)")


def _refused_exit(refused: list[str], stale: Mapping[str, str] | None = None) -> int:
    """1 when an AuditKIT record was refused or a cell was skipped as stale, else 0. The stale cells are
    listed with the line that starts each over."""
    if refused:
        log.error("%d AuditKIT records were refused (see the errors above): %s", len(refused), refused)
    if stale:
        log.error("%d cells were skipped as stale (see the errors above). To start one over, run its line "
                  "once as a job line right after `lift sync pull`, followed by the line that failed:\n%s",
                  len(stale), "".join(f"  {cell}: {line}\n" for cell, line in stale.items()))
    return 1 if refused or stale else 0


# --- lift smoke ---------------------------------------------------------------------------------------


def _smoke(args: argparse.Namespace) -> int:
    _logging()
    import torch

    if not torch.cuda.is_available():
        print("lift smoke is Gate 0 and needs a CUDA GPU", file=sys.stderr)
        return 2
    from lift.gate0 import CHECKS, check_status, pin_hint, report_name, run_gate0, verdict, write_report
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
        why = c.get("error") or c.get("reason") or c.get("failure")
        print(f"{check_status(c)}  {check}" + (f"  ({why})" if why else ""))
    print(f"\n{verdict(result)}: {result['decision']}")
    print(f"\n{pin_hint(result)}\n\nwrote {json_path} and {md_path}")
    return 0 if result["pass"] else 1


# --- lift generate ------------------------------------------------------------------------------------


def _generate(args: argparse.Namespace) -> int:
    _logging()
    require_pinned("models.global")
    sets = _names(args.sets, ALL_SETS, SAFETY_SETS)
    langs = _names(args.langs, LANGS, LANGS)
    from lift.models.generate import StaleOutputError, generate, generations_path
    from lift.models.variants import build_variant, get_spec
    from lift.prompts import load

    spec = get_spec(args.variant)
    variant = build_variant(spec, device=args.device, dtype=_dtype(args.dtype))
    stale: dict[str, str] = {}
    for set_name in sets:
        for lang in langs:
            prompts = load(set_name, lang, args.split)
            if args.limit:
                prompts = prompts[: args.limit]
            if not prompts:
                log.warning("%s_%s has no %s prompts", set_name, lang, args.split)
                continue
            try:
                generate(variant, prompts, out_path=generations_path(spec.name, set_name, lang, args.split),
                         batch_size=args.batch_size)
            except StaleOutputError as exc:
                _stale(stale, spec.name, set_name, lang, args.split, exc)
    return _refused_exit([], stale)


# --- lift judge ---------------------------------------------------------------------------------------


def _judge(args: argparse.Namespace) -> int:
    _logging()
    from lift.judge.pipeline import judge_dir, judge_file, make_judge, read_keys_file

    if (args.sample is not None or args.keys) and args.judge != "second":
        raise SystemExit("--sample and --keys select a subset, which only the second judge may judge; "
                         "the primary judge covers every generation")
    if args.sample is not None and args.keys:
        raise SystemExit("pass either --sample or --keys, not both")
    sets = _names(args.sets, ALL_SETS, SAFETY_SETS)
    langs = _names(args.langs, LANGS, LANGS)
    local = judge_dir(args.judge) == "qwen3guard"
    if local:
        require_pinned(_judge_entry(args.judge))
    from lift.models.generate import StaleOutputError, generations_path

    keys = read_keys_file(Path(args.keys)) if args.keys else None
    if keys is None:
        cells = [(s, lang) for s in sets for lang in langs]
    else:
        cells = _key_cells(args.variant, args.split, keys, sets=sets if args.sets else None,
                           langs=langs if args.langs else None, allow_missing=args.allow_missing)
    paths = {(s, lang): generations_path(args.variant, s, lang, args.split) for s, lang in cells}
    missing = [str(p) for p in paths.values() if not p.is_file() or not p.stat().st_size]
    if missing:
        raise SystemExit(f"{len(missing)} generations files are missing or empty, e.g. {missing[:3]}; "
                         "run lift generate first (lift restart leaves a cell's files empty)")
    judge = make_judge(args.judge, device=_judge_device(args.device) if local else None,
                       dtype=_dtype(args.dtype) if local else None)
    refused: list[str] = []
    stale: dict[str, str] = {}
    for (set_name, lang), path in paths.items():
        try:
            judge_file(path, judge=judge, sample=args.sample, keys=keys)
        except StaleOutputError as exc:
            _stale(stale, args.variant, set_name, lang, args.split, exc)
            continue
        if args.judge == "primary":
            _record(args.variant, set_name, lang, args.split, dtype=_dtype(args.dtype), refused=refused)
    return _refused_exit(refused, stale)


def _key_cells(variant: str, split: str, keys: Sequence[str], *, sets: Collection[str] | None,
               langs: Collection[str] | None, allow_missing: bool) -> list[tuple[str, str]]:
    """The (set, lang) generations files a --keys run judges: every file of ``variant``'s ``split`` that
    holds a listed key, within --sets/--langs when they were given. A key in no file at all was
    regenerated under a new key (a fixed prompt) or is a typo, and stops the command unless
    ``allow_missing``: its item would silently lose its label. Keys in files outside an explicit
    --sets/--langs are expected (P04 prints one line per group over one list) and are logged by
    "{set}_{lang}" of the file, counts only."""
    from lift.models.generate import read_generations
    from lift.paths import generations_dir

    wanted = set(keys)
    holder: dict[str, tuple[str, str]] = {}
    for path in sorted((generations_dir() / variant).glob(f"*_{split}.jsonl")):
        set_name, lang, _ = path.stem.rsplit("_", 2)
        holder.update({g.key: (set_name, lang) for g in read_generations(path) if g.key in wanted})
    unknown = sorted(wanted - set(holder))
    if unknown and not allow_missing:
        raise SystemExit(f"{len(unknown)} listed keys are in none of {variant}'s {split} generations files, "
                         f"e.g. {[k[:12] for k in unknown[:5]]}; regenerate the key list "
                         "(lift validate second-judge-sample) or pass --allow-missing")
    if unknown:
        log.warning("%d listed keys are in no %s %s generations file", len(unknown), variant, split)
    held = Counter(holder.values())
    chosen = [c for c in held if (sets is None or c[0] in sets) and (langs is None or c[1] in langs)]
    outside = {f"{s}_{lang}": n for (s, lang), n in sorted(held.items()) if (s, lang) not in chosen}
    if outside:
        log.warning("%d listed keys are in %s %s generations files outside --sets/--langs; this run does "
                    "not judge them (count by set_lang): %s", sum(outside.values()), variant, split, outside)
    return sorted(chosen, key=lambda c: (_rank(c[0], ALL_SETS), _rank(c[1], LANGS)))


def _rank(name: str, order: Sequence[str]) -> tuple[int, str]:
    return (order.index(name) if name in order else len(order), name)


def _judge_entry(role: str) -> str:
    return {"primary": "judges.safety_primary", "second": "judges.safety_second"}[role]


# --- lift run-dev -------------------------------------------------------------------------------------


def run_cells(holder: dict[str, Any], make_judge: Callable[[], Any], cells: Sequence[tuple[str, str]], *,
              split: str = "dev", concurrent: bool, interleave: bool = True, limit: int | None = None,
              batch_size: int | None = None, lid: Any = None,
              record: Callable[[str, str], None] | None = None,
              stale: dict[str, str] | None = None) -> list[Path]:
    """Generate then judge every (set, lang) cell; returns the judgements paths.

    ``holder["variant"]`` is the only reference to the generator, so it can really be freed.
    ``concurrent``: the judge runs in its own thread on its own GPU and judges each file as soon as
    it is written. Otherwise both share one device: with ``interleave`` each file is judged right
    after it is generated (both models loaded); without, every file is generated first, the
    generator is dropped and its memory released, then the judge is loaded and judges them all.
    A cell whose generations or judgements file is refused as stale (StaleOutputError) is logged and
    skipped, and the other cells go on: a cut-short session must not block the rest. Its ``lift restart``
    line goes into ``stale`` under "{set}_{lang}_{split}"; without ``stale``, one StaleOutputError
    listing them is raised once every other cell is done.
    """
    from lift.judge.pipeline import judge_file
    from lift.models.generate import StaleOutputError, generate, generations_path
    from lift.prompts import load

    name = holder["variant"].spec.name
    done: list[Path] = []
    skipped: dict[str, str] = {} if stale is None else stale

    def finished() -> list[Path]:
        if stale is None and skipped:
            raise StaleOutputError(f"{len(skipped)} cells were skipped as stale: "
                                   + "; ".join(f"{cell}: {line}" for cell, line in skipped.items()))
        return done

    def gen(set_name: str, lang: str) -> Path | None:
        prompts = load(set_name, lang, split)[:limit] if limit else load(set_name, lang, split)
        if not prompts:
            log.warning("%s_%s has no %s prompts", set_name, lang, split)
            return None
        out_path = generations_path(name, set_name, lang, split)
        try:
            return generate(holder["variant"], prompts, out_path=out_path, batch_size=batch_size)
        except StaleOutputError as exc:
            _stale(skipped, name, set_name, lang, split, exc)
            return None

    def jud(judge: Any, path: Path, set_name: str, lang: str) -> None:
        try:
            done.append(judge_file(path, judge=judge, lid=lid))
        except StaleOutputError as exc:
            _stale(skipped, name, set_name, lang, split, exc)
            return
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
        return finished()

    if interleave:
        judge = make_judge()
        for set_name, lang in cells:
            path = gen(set_name, lang)
            if path is not None:
                jud(judge, path, set_name, lang)
        return finished()

    written = [(gen(s, lang), s, lang) for s, lang in cells]
    holder.clear()
    _release_memory()
    judge = make_judge()
    for path, set_name, lang in written:
        if path is not None:
            jud(judge, path, set_name, lang)
    return finished()


def _release_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _on_cuda(device: str) -> bool:
    if not device.startswith("cuda"):
        return False
    import torch

    return torch.cuda.is_available()


def _free_gb(device: str) -> float:
    import torch

    free, _total = torch.cuda.mem_get_info(torch.device(device))
    return free / 1e9


def _run_dev(args: argparse.Namespace) -> int:
    _logging()
    from lift.judge.pipeline import judge_dir, make_judge

    # As in _judge, only a local judge has a revision to pin: once the judge decision makes the frontier
    # judge primary, the judges.safety_primary block names a provider and model instead.
    local = judge_dir("primary") == "qwen3guard"
    require_pinned("models.global", *([_judge_entry("primary")] if local else []))
    langs = _names(args.langs, LANGS, LANGS)
    sets = _names(args.sets, SAFETY_SETS, SAFETY_SETS)
    dtype = _dtype(args.dtype)
    from lift.models.variants import build_variant, get_spec

    spec = get_spec(args.variant)
    gen_device, judge_device = args.device, _judge_device(args.judge_device)
    # The holder is the only reference to the generator, so the one-GPU path can free it.
    holder = {"variant": build_variant(spec, device=gen_device, dtype=dtype)}
    concurrent = gen_device != judge_device
    interleave = True
    if not concurrent and _on_cuda(gen_device):
        free = _free_gb(gen_device)
        interleave = free >= JUDGE_GB + JUDGE_KV_HEADROOM_GB
        log.info("one GPU: %.1f GB free after loading the generator; %s", free,
                 "judge loaded alongside" if interleave else "judging after all generation, generator freed")

    cells = [(s, lang) for s in sets for lang in langs]
    refused: list[str] = []
    stale: dict[str, str] = {}

    def record(set_name: str, lang: str) -> None:
        _record(args.variant, set_name, lang, "dev", dtype=dtype, refused=refused)

    run_cells(holder, lambda: make_judge("primary", device=judge_device, dtype=dtype), cells,
              concurrent=concurrent, interleave=interleave, limit=args.limit, batch_size=args.batch_size,
              record=None if args.limit else record, stale=stale)
    return _refused_exit(refused, stale)


# --- lift restart -------------------------------------------------------------------------------------


RESTART_HELP = """\
Start cells over that lift generate, judge or run-dev skipped as stale (begun under another stack, or
made from other inputs; the error names the exact line). For each --sets x --langs cell, the
generations file and every judgements file of the cell (with --judge, only that judge's judgements
file) is copied to outputs/.aside/<UTC stamp>/, which is never synced, and left EMPTY: the next
lift sync push replaces the runs repo's copies, and generate, judge or run-dev fill the cell again.

Run it once, as a job line right after `lift sync pull` and before the line that failed, then leave
it out of later jobs (each run would start the cell over again). Moving a file aside by hand does not
work: sync never deletes, so the next pull brings the runs repo's copy back. A file that is not here
is logged and skipped.

example job (lift remote --job, or the Kaggle runner's JOB):
  lift sync pull
  lift restart --variant stock --split dev --sets harmful --langs hi
  lift run-dev --variant stock
"""


def _restart(args: argparse.Namespace) -> int:
    _logging()
    from lift.models.generate import aside_root, restart_cell
    from lift.models.variants import get_spec

    try:
        get_spec(args.variant)
    except KeyError as exc:
        raise SystemExit(exc.args[0]) from None
    sets = _names(args.sets, ALL_SETS, ())
    langs = _names(args.langs, LANGS, ())
    judge = None
    if args.judge:
        from lift.judge.pipeline import judge_dir

        judge = judge_dir(args.judge)
    aside = aside_root()
    emptied = 0
    for set_name in sets:
        for lang in langs:
            for path, n, copy in restart_cell(args.variant, set_name, lang, args.split, aside=aside,
                                              judge_dir=judge):
                if n is None:
                    log.warning("%s: not here, skipped (lift restart runs right after lift sync pull: a file "
                                "only the runs repo holds is left as it is there)", path)
                elif copy is None:
                    log.info("%s: already empty", path)
                else:
                    emptied += 1
                    log.info("%s: %d record(s) set aside in %s; the file is left empty", path, n, aside)
    if emptied:
        log.info("%d file(s) emptied: the next lift sync push replaces the runs repo's copies, and lift "
                 "generate, judge or run-dev fill them again", emptied)
    return 0


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
                   help="second judge: judge exactly the generation keys listed in this file, in every "
                        "generations file that holds one (within --sets/--langs when given)")
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

    p = sub.add_parser("restart", help="start stale cells over: set their files aside, leave them empty "
                                       "(a job line right after lift sync pull)",
                       description=RESTART_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, help="a name from configs/variants.yaml")
    p.add_argument("--split", required=True, choices=["dev", "final"])
    p.add_argument("--sets", nargs="+", required=True, help=f"one or more of: {' '.join(ALL_SETS)}")
    p.add_argument("--langs", nargs="+", required=True, help=f"one or more of: {' '.join(LANGS)}")
    p.add_argument("--judge", choices=["primary", "second"], default=None,
                   help="only this judge's judgements file (default: the generations file and every "
                        "judgements file of each cell)")
    p.set_defaults(func=_restart)
