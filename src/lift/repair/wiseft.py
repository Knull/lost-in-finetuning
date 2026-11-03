"""Adapter attenuation (WiSE-FT), the baseline every repair must beat. Implemented in P04.

Our parameterisation: model = Global + lambda * (adapter - Global), lambda in {0.25, 0.5, 0.75}.
SafeTune 0.1.12 ``safetune.interventions.recover.wise_ft.apply_wise_ft(model, aligned, alpha)``
computes ``alpha * aligned + (1 - alpha) * model`` (alpha TOWARD the aligned model), so call it
with ``model=<merged adapter>, aligned=<stock Global>, alpha=1 - lambda``. Getting this backwards
silently swaps lambda 0.25 and 0.75. It mutates ``model`` in place.

With a LoRA adapter this equals merging the LoRA delta scaled by lambda; a unit test checks the
two agree on a tiny model, which catches the sign convention.
"""

from __future__ import annotations

from typing import Any


def make(adapter_model: Any, stock_model: Any, lam: float) -> Any:
    raise NotImplementedError("P04")
