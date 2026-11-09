"""``lift sweep``: the seed-1 dev sweep of the repair settings (P05). The rules are here; the CLI is
lift.commands.repair.

GPU half (``lift sweep --adapter A``). The settings of the chosen families (lift.repair.settings, cost
order) are registered in this checkout's configs/variants.yaml first (deterministic entries; the laptop
writes the same ones with ``lift register-repairs``). Then each setting runs three stages, each through
the existing command, in this process, so every number comes from the one harness (rule 2):

    safety    lift run-dev --variant V                       harmful, easy and hard benign, 6 languages
    utility   lift utility --variant V --split dev --generate-only        100 Hindi pairs vs stock
    belebele  lift belebele --variant V --split dev --langs hi

Each command builds the variant afresh (``build_variant``: a weight-space repair starts from a freshly
merged adapter every time) and resumes by cache key. A stage that exits 0 is recorded in the sweep state
(outputs/metrics/sweep/{adapter}.json) with the setting's intervention digest and the dtype, so a rerun
after a crash goes straight to the first stage not done (``--recheck`` runs every stage again: each one
then finds its cells done and costs a model load). A stage that exits 3 (cells skipped, each listed with
its remedy by the command) or fails is not recorded as done; the sweep goes on with the next stage and
setting, and exits 3 (some stage partial) or 1 (some stage failed) at the end.

Costs. Each setting gets a run manifest (stage "sweep_setting": the wall time of each stage this run, the
judge device, and ``devices``: 2 when run-dev's local judge sits on another GPU than the generator, as on
Kaggle T4 x2, else 1) and its cost summary, recomputed from its records whenever it runs: the GPU seconds the
generation records attribute to each prompt, the tokens generated, seconds per generated token, and for
CAST the gate's prefill pass (``gate_seconds`` per record, lift.repair.cast) apart from the rest, with
its share; Belebele's scoring seconds likewise. The adapter's own summary is printed beside them, so the
gate's overhead is read against the unrepaired model on the same prompts.

Laptop half (``lift sweep --adapter A --judge-only``): ``lift utility --variant V --split dev
--judge-only`` for every setting (pass ``--no-wait`` to submit every Message Batch first and collect
them with a later run).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lift import config
from lift.paths import metrics_dir
from lift.repair import settings as rs

log = logging.getLogger("lift.sweep")

STAGES = ("safety", "utility", "belebele")
SAFETY = ("harmful", "easy_benign", "hard_benign")


def state_path(adapter: str) -> Path:
    return metrics_dir() / "sweep" / f"{adapter}.json"


def read_state(adapter: str) -> dict[str, Any]:
    path = state_path(adapter)
    if not path.is_file() or not path.stat().st_size:
        return {"adapter": adapter, "settings": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, doc: Any) -> None:
    """``doc`` as indented JSON at ``path``, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def write_state(adapter: str, state: dict[str, Any]) -> None:
    write_json(state_path(adapter), state)


def stage_argv(stage: str, name: str, *, device: str, judge_device: str, dtype: str | None,
               batch_size: int | None) -> list[str]:
    """The lift command line of one stage of one setting."""
    tail = ([f"--dtype={dtype}"] if dtype else []) + ([f"--batch-size={batch_size}"] if batch_size else [])
    if stage == "safety":
        return ["run-dev", f"--variant={name}", f"--device={device}", f"--judge-device={judge_device}", *tail]
    if stage == "utility":
        return ["utility", f"--variant={name}", "--split=dev", "--generate-only", f"--device={device}", *tail]
    if stage == "belebele":
        return ["belebele", f"--variant={name}", "--split=dev", "--langs=hi", f"--device={device}", *tail]
    raise ValueError(f"unknown stage {stage!r}")


def _call(argv: Sequence[str]) -> int:
    """Run one lift command in this process: its exit code (SystemExit's, or 1 for an error, logged)."""
    from lift.cli import main

    try:
        return int(main(list(argv)) or 0)
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        log.error("lift %s stopped: %s", argv[0], code)
        return 1
    except KeyboardInterrupt:
        raise
    except Exception:
        log.exception("lift %s failed", " ".join(argv))
        return 1
    finally:
        _release()


def _release() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _done(entry: dict[str, Any], stage: str, digest: str, dtype: str) -> bool:
    got = entry.get("stages", {}).get(stage) or {}
    return got.get("rc") == 0 and got.get("digest") == digest and got.get("dtype") == dtype


def cost_summary(name: str, dtype: str) -> dict[str, Any]:
    """GPU seconds and tokens of one variant's dev records in ``dtype`` (module docstring), from its
    generations (safety cells, utility) and Belebele-hi score files."""
    from lift.models.generate import generations_path, read_generations
    from lift.models.loglik import belebele_scores_path, read_scores
    from lift.schemas import LANGS

    gen = {"n": 0, "seconds": 0.0, "n_generated": 0, "gate_seconds": 0.0, "gate_fired": 0, "gated": 0}
    cells = [(s, lang) for s in SAFETY for lang in LANGS] + [("utility", "hi")]
    for set_name, lang in cells:
        path = generations_path(name, set_name, lang, "dev")
        if not path.is_file():
            continue
        for r in read_generations(path):
            if r.meta.get("dtype") != dtype:
                continue
            gen["n"] += 1
            gen["seconds"] += float(r.seconds)
            gen["n_generated"] += int(r.meta.get("n_generated", r.n_new_tokens))
            if "gate_seconds" in r.meta:
                gen["gated"] += 1
                gen["gate_seconds"] += float(r.meta["gate_seconds"])
                gen["gate_fired"] += int(bool(r.meta.get("gate_fired")))
    out: dict[str, Any] = {"generation": gen}
    if gen["n_generated"]:
        gen["seconds_per_generated_token"] = gen["seconds"] / gen["n_generated"]
    if gen["seconds"] and gen["gated"]:
        gen["gate_share"] = gen["gate_seconds"] / gen["seconds"]
        gen["gate_fire_rate"] = gen["gate_fired"] / gen["gated"]
    bpath = belebele_scores_path(name, "hi", "dev", dtype=dtype)
    if bpath.is_file():
        recs = [r for r in read_scores(bpath, "belebele") if r["meta"].get("dtype") == dtype]
        out["belebele"] = {"n": len(recs), "seconds": sum(float(r["seconds"]) for r in recs),
                           "gate_seconds": sum(float(r["meta"].get("gate_seconds", 0.0)) for r in recs)}
    return out


def gpu(adapter: str, *, families: Sequence[str], device: str, judge_device: str, dtype: str | None,
        batch_size: int | None, recheck: bool = False, call: Callable[[Sequence[str]], int] = _call) -> int:
    """The GPU half: register, then every stage of every setting (module docstring). Exit 0, 3 or 1."""
    from lift.cachekey import intervention_digest
    from lift.commands.run import _judge_device
    from lift.judge.pipeline import judge_dir
    from lift.manifest import finish_run, start_run

    run_dtype = dtype or config.get("generation.dtype")
    judge_device = _judge_device(judge_device)
    # The devices the stages hold: run-dev's judge takes its own GPU only when it is local and elsewhere.
    devices = len({device, judge_device}) if judge_dir("primary") == "qwen3guard" else 1
    chosen = rs.settings(families)
    specs = rs.specs_for(adapter, chosen)
    outcome = rs.register(specs)
    log.info("registry: %s", ", ".join(f"{k} {v}" for k, v in outcome.items()))
    state = read_state(adapter)
    results: dict[str, dict[str, int]] = {}
    for spec in specs:
        digest = intervention_digest(spec)
        entry = state["settings"].setdefault(spec.name, {"stages": {}})
        run = start_run("sweep_setting", variant=spec.name, adapter=adapter, method=spec.method,
                        params=spec.params, dtype=run_dtype, device=device, judge_device=judge_device,
                        devices=devices)
        stage_seconds: dict[str, float] = {}
        results[spec.name] = {}
        for stage in STAGES:
            if not recheck and _done(entry, stage, digest, run_dtype):
                log.info("%s %s: done in an earlier run", spec.name, stage)
                results[spec.name][stage] = 0
                continue
            argv = stage_argv(stage, spec.name, device=device, judge_device=judge_device, dtype=dtype,
                              batch_size=batch_size)
            log.info("%s %s: lift %s", spec.name, stage, " ".join(argv))
            t0 = datetime.now(UTC)
            rc = call(argv)
            stage_seconds[stage] = round((datetime.now(UTC) - t0).total_seconds(), 3)
            results[spec.name][stage] = rc
            entry["stages"][stage] = {"rc": rc, "digest": digest, "dtype": run_dtype,
                                      "finished": datetime.now(UTC).isoformat(),
                                      "seconds": stage_seconds[stage]}
            write_state(adapter, state)
        cost = cost_summary(spec.name, run_dtype)
        entry["cost"] = cost
        write_state(adapter, state)
        finish_run(run, status="ok" if all(v == 0 for v in results[spec.name].values()) else "partial",
                   stages=results[spec.name], stage_seconds=stage_seconds, cost=cost)
    state["adapter_cost"] = cost_summary(adapter, run_dtype)
    write_state(adapter, state)
    print(summary(adapter, results, state))
    codes = {rc for r in results.values() for rc in r.values()}
    if codes - {0, 3}:
        return 1
    return 3 if 3 in codes else 0


def judge_only(adapter: str, *, families: Sequence[str], no_wait: bool = False, sync: bool = False,
               call: Callable[[Sequence[str]], int] = _call) -> int:
    """The laptop half: every setting's utility pairs judged (``lift utility --judge-only``)."""
    codes = {}
    for setting in rs.settings(families):
        name = setting.name(adapter)
        argv = ["utility", f"--variant={name}", "--split=dev", "--judge-only"]
        argv += ["--no-wait"] if no_wait else []
        argv += ["--sync"] if sync else []
        codes[name] = call(argv)
    for name, rc in codes.items():
        print(f"{name}: lift utility --judge-only exit {rc}")
    values = set(codes.values())
    if values - {0, 3}:
        return 1
    return 3 if 3 in values else 0


def summary(adapter: str, results: dict[str, dict[str, int]], state: dict[str, Any]) -> str:
    lines = [f"sweep {adapter}: stage exit codes (0 done, 3 cells listed above, other: failed)"]
    for name, stages in results.items():
        cost = state["settings"].get(name, {}).get("cost", {}).get("generation", {})
        extra = ""
        if cost.get("seconds_per_generated_token"):
            extra = f"; {1000 * cost['seconds_per_generated_token']:.2f} ms GPU per generated token"
        if cost.get("gate_share") is not None:
            extra += (f", gate prefill {100 * cost['gate_share']:.1f}% of it, fires on "
                      f"{100 * cost['gate_fire_rate']:.1f}%")
        lines.append(f"  {name}: " + ", ".join(f"{s} {rc}" for s, rc in stages.items()) + extra)
    base = state.get("adapter_cost", {}).get("generation", {})
    if base.get("seconds_per_generated_token"):
        lines.append(f"  {adapter} (unrepaired): {1000 * base['seconds_per_generated_token']:.2f} ms GPU per "
                     "generated token")
    return "\n".join(lines)
