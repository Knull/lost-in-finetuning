"""The cost section (P07), from the run manifests and the final generation records. Laptop, no torch.

- GPU-hours of the final evaluation per variant: the sum over its ``final_variant`` manifests (lift final;
  each records ``gpu_hours`` = its stages' wall time x the devices it held, so a resumed run adds its own
  share); ``gpu_hours_per_variant_final`` is their mean over the variants with a manifest.
- The dev sweep's GPU-hours: the ``sweep_setting`` manifests (wall time x devices) of the selection's
  adapter in the reported precision only (not a stress sweep, not another dtype), and the CAST fit of that
  adapter (``fit_cast`` manifests, same filter), which the check-up needs before its CAST settings run.
  Final manifests count in the reported precision only.
- A check-up is the whole acceptance test of one adapter (decisions.md 73): the seed-1 dev sweep (with its
  CAST fit) plus the final battery of stock, the seed-1 adapter and the seed-1 frozen winner (``checkup``).
  ``gpu_hours_per_checkup`` is their sum, None (not run) while any of them has no manifest.
- Latency: GPU seconds per generated token of each variant's final generations (safety cells and
  utility answers), as the harness attributes them per record (``seconds``, ``meta.n_generated``), and
  for CAST the gate's prefill pass apart (``meta.gate_seconds``): its share of the time and how often it
  fired. Only records in the reported precision count.

Commercial ROI is never claimed (Tiny Aya derivatives are non-commercial).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lift import config
from lift.paths import manifests_dir
from lift.schemas import LANGS

SPLIT = "final"


def manifests(folder: Path | None = None) -> list[dict[str, Any]]:
    folder = folder or manifests_dir()
    out = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict):
            out.append(doc)
    return out


def _hours(doc: Mapping[str, Any]) -> float:
    if doc.get("gpu_hours") is not None:
        return float(doc["gpu_hours"])
    return float(doc.get("wall_seconds") or 0.0) * int(doc.get("devices") or 1) / 3600


def _mine(doc: Mapping[str, Any], dtype: str | None, adapter: str | None = None) -> bool:
    return (dtype is None or doc.get("dtype") == dtype) and (adapter is None or doc.get("adapter") == adapter)


def final_hours(docs: Iterable[Mapping[str, Any]], dtype: str | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    for doc in docs:
        if doc.get("stage") == "final_variant" and doc.get("variant") and _mine(doc, dtype):
            out[doc["variant"]] = out.get(doc["variant"], 0.0) + _hours(doc)
    return out


def stage_hours(
    docs: Iterable[Mapping[str, Any]], stage: str, *, dtype: str | None = None, adapter: str | None = None
) -> float | None:
    """GPU-hours of every ``stage`` manifest of ``adapter`` in ``dtype`` (None: any); None when none."""
    hours = [_hours(d) for d in docs if d.get("stage") == stage and _mine(d, dtype, adapter)]
    return sum(hours) if hours else None


def sweep_hours(
    docs: Iterable[Mapping[str, Any]], *, dtype: str | None = None, adapter: str | None = None
) -> float | None:
    return stage_hours(docs, "sweep_setting", dtype=dtype, adapter=adapter)


def latency(variant: str, dtype: str | None = None) -> dict[str, Any] | None:
    """GPU seconds per generated token over ``variant``'s final generations in ``dtype``, with CAST's gate
    apart; None when there are none."""
    from lift.models.generate import generations_path, read_generations

    dtype = dtype or str(config.get("generation.dtype"))
    cells = [(s, lang) for s in ("harmful", "easy_benign", "hard_benign") for lang in LANGS]
    cells += [("redteam", "hi"), ("utility", "hi")]
    n = tokens = gated = fired = 0
    seconds = gate_seconds = 0.0
    for set_name, lang in cells:
        path = generations_path(variant, set_name, lang, SPLIT)
        if not path.is_file():
            continue
        for r in read_generations(path):
            if r.meta.get("dtype") != dtype:
                continue
            n += 1
            seconds += float(r.seconds)
            tokens += int(r.meta.get("n_generated", r.n_new_tokens))
            if "gate_seconds" in r.meta:
                gated += 1
                gate_seconds += float(r.meta["gate_seconds"])
                fired += int(bool(r.meta.get("gate_fired")))
    if not n or not tokens:
        return None
    return {
        "n": n,
        "seconds": seconds,
        "n_generated": tokens,
        "ms_per_token": 1000 * seconds / tokens,
        "gate_share": gate_seconds / seconds if gated and seconds else 0.0,
        "gate_fire_rate": fired / gated if gated else None,
    }


def section(
    names: Sequence[tuple[str, str]],
    *,
    docs: Sequence[Mapping[str, Any]] | None = None,
    dtype: str | None = None,
    checkup: Sequence[str] = (),
    adapter: str | None = None,
) -> dict[str, Any]:
    """{"rows": [...], "gpu_hours_per_checkup", "gpu_hours_per_variant_final", "sweep_gpu_hours", ...} for
    ``names`` [(variant, role)]; ``checkup``: the final variants a check-up runs; ``adapter``: the
    selection's (seed-1) adapter, whose sweep and CAST fit a check-up counts (module docstring)."""
    docs = manifests() if docs is None else docs
    dtype = dtype or str(config.get("generation.dtype"))
    hours = final_hours(docs, dtype)
    rows = []
    for name, role in names:
        lat = latency(name, dtype)
        rows.append(
            {
                "variant": name,
                "role": role,
                "gpu_hours": hours.get(name),
                "ms_per_token": lat["ms_per_token"] if lat else None,
                "gate_share": lat["gate_share"] if lat else None,
                "gate_fire_rate": lat["gate_fire_rate"] if lat else None,
            }
        )
    measured = [h for name, _ in names if (h := hours.get(name)) is not None]
    sweep = sweep_hours(docs, dtype=dtype, adapter=adapter)
    fit = stage_hours(docs, "fit_cast", dtype=dtype, adapter=adapter)
    parts = [hours.get(name) for name in checkup]
    complete = bool(checkup) and sweep is not None and all(h is not None for h in parts)
    total = sweep + (fit or 0.0) + sum(h for h in parts if h is not None) if complete else None
    return {
        "rows": rows,
        "gpu_hours_per_checkup": total,
        "checkup": {"sweep": sweep, "fit_cast": fit, **{name: hours.get(name) for name in checkup}},
        "gpu_hours_per_variant_final": sum(measured) / len(measured) if measured else None,
        "gpu_hours_final_total": sum(measured) if measured else None,
        "sweep_gpu_hours": sweep,
        "fit_cast_gpu_hours": fit,
        "adapter": adapter,
    }
