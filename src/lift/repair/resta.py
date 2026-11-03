"""ReSta-A: add alpha * (Global - Base), the full post-training delta. Implemented in P05.

SafeTune 0.1.12: ``apply_resta(finetuned, *, base, aligned, alpha, dare=False, device="cpu")``
computes ``finetuned + alpha * (aligned - base)`` one tensor at a time and mutates ``finetuned``.
Always pass ``dare=False`` explicitly (it defaults to False in 0.1.12 but earlier releases and
ReStaTrainer turned DARE on). Keep Base and Global on CPU (``device="cpu"``). Reload pristine
adapter weights before each alpha: applying in place accumulates. alpha 1 breaks Tiny Aya.

The delta carries instruction following and style as well as safety; the write-up says so.
"""

from __future__ import annotations

from typing import Any


def make(adapter_model: Any, base_cpu: Any, stock_cpu: Any, alpha: float) -> Any:
    raise NotImplementedError("P05")
