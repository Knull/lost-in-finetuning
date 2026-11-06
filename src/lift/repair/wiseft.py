"""Adapter attenuation (WiSE-FT), the baseline every repair must beat. Implemented in P05.

Our parameterisation: model = Global + lambda * (adapter - Global), lambda in {0.25, 0.5, 0.75}.
SafeTune 0.1.12 ``safetune.interventions.recover.wise_ft.apply_wise_ft(model, aligned, alpha)``
computes ``alpha * aligned + (1 - alpha) * model`` (alpha TOWARD the aligned model), so call it
with ``model=<merged adapter>, aligned=<stock Global>, alpha=1 - lambda``. Getting this backwards
silently swaps lambda 0.25 and 0.75. It mutates ``model`` in place.

With a LoRA adapter this equals merging the LoRA delta scaled by lambda; a unit test checks the
two agree on a tiny model, which catches the sign convention.

Memory (one T4, 16 GB): the merged adapter is on the GPU in the run's dtype, stock Global stays on the
CPU, and apply_wise_ft moves one stock tensor at a time to the adapter's device (computing in fp32).
It holds five fp32 copies of a tensor at once (stock, adapter, both scaled, their sum): for Tiny Aya's
262k x 2048 tied embedding that is 10 GiB beside the 6.2 GiB fp16 model, more than a T4's 15 GiB. So a
parameter of more than ``LARGE_NUMEL`` elements is moved to the CPU for the call and back after (the same
elementwise fp32 arithmetic, the same result; a tied weight stays tied).
The caller always passes a freshly merged adapter (lift.repair.build): applying twice compounds.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

LARGE_NUMEL = 1 << 27         # 512 MiB in fp32: on Tiny Aya only the embedding (2 GiB) is larger


def alpha_for(lam: float) -> float:
    """SafeTune's alpha (weight on stock Global) for our lambda (weight on the adapter's delta)."""
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"WiSE-FT lambda must be in [0, 1], got {lam}")
    return 1.0 - lam


def make(adapter_model: Any, stock_model: Any, lam: float) -> Any:
    """``adapter_model`` (the merged adapter, mutated in place and returned) moved to
    Global + lam * (adapter - Global)."""
    from safetune.interventions.recover.wise_ft import apply_wise_ft

    _check_same_parameters(adapter_model, stock_model)
    with _large_on_cpu(adapter_model):
        apply_wise_ft(adapter_model, stock_model, alpha=alpha_for(lam))
    return adapter_model


def large_parameters(model: Any) -> list[str]:
    """The parameters (each tied group once) with more than ``LARGE_NUMEL`` elements."""
    return [n for n, p in model.named_parameters() if p.numel() > LARGE_NUMEL]


@contextmanager
def _large_on_cpu(model: Any) -> Iterator[None]:
    """Inside the block the large parameters' data is on the CPU; after it, back on its own device."""
    large = set(large_parameters(model))
    moved = []
    for name, p in model.named_parameters():
        if name in large:
            moved.append((p, p.device))
            p.data = p.data.to("cpu")
    try:
        yield
    finally:
        for p, device in moved:
            p.data = p.data.to(device)


def _check_same_parameters(model: Any, reference: Any) -> None:
    """apply_wise_ft skips a parameter missing from the reference and says so only in a log line; here a
    missing name or another shape is an error, so a repaired model is never silently half-repaired."""
    ref = {k: tuple(v.shape) for k, v in reference.state_dict().items()}
    missing = [n for n, p in model.named_parameters() if n not in ref]
    shapes = [n for n, p in model.named_parameters() if n in ref and tuple(p.shape) != ref[n]]
    if missing or shapes:
        raise ValueError(f"stock Global does not match the adapter model: missing {missing[:5]}, "
                         f"other shapes {shapes[:5]}")
