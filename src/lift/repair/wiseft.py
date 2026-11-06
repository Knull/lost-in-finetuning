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
The caller always passes a freshly merged adapter (lift.repair.build): applying twice compounds.
"""

from __future__ import annotations

from typing import Any


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
    return apply_wise_ft(adapter_model, stock_model, alpha=alpha_for(lam))


def _check_same_parameters(model: Any, reference: Any) -> None:
    """apply_wise_ft skips a parameter missing from the reference and says so only in a log line; here a
    missing name or another shape is an error, so a repaired model is never silently half-repaired."""
    ref = {k: tuple(v.shape) for k, v in reference.state_dict().items()}
    missing = [n for n, p in model.named_parameters() if n not in ref]
    shapes = [n for n, p in model.named_parameters() if n in ref and tuple(p.shape) != ref[n]]
    if missing or shapes:
        raise ValueError(f"stock Global does not match the adapter model: missing {missing[:5]}, "
                         f"other shapes {shapes[:5]}")
