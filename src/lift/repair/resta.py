"""ReSta-A: add alpha * (Global - Base), the full post-training delta. Implemented in P05.

SafeTune 0.1.12: ``apply_resta(finetuned, *, base, aligned, alpha, dare=False, device="cpu")``
computes ``finetuned + alpha * (aligned - base)`` one tensor at a time and mutates ``finetuned``.
Always pass ``dare=False`` explicitly (it defaults to False in 0.1.12 but earlier releases and
ReStaTrainer turned DARE on). Keep Base and Global on CPU (``device="cpu"``). Reload pristine
adapter weights before each alpha: applying in place accumulates. alpha 1 breaks Tiny Aya.

The delta carries instruction following and style as well as safety; the write-up says so.

Before applying, Base and Global must hold the same parameter names and shapes, and every parameter of
the adapter model must be among them: apply_resta silently skips names it cannot pair, which would leave
part of the model unrepaired.
"""

from __future__ import annotations

from typing import Any

DARE = False          # configs/repairs.yaml resta.dare; checked by make


def make(adapter_model: Any, base_cpu: Any, stock_cpu: Any, alpha: float) -> Any:
    """``adapter_model`` (mutated in place and returned) + alpha * (stock_cpu - base_cpu), DARE off."""
    from safetune.interventions.recover.resta import apply_resta

    from lift import config

    if config.get("resta.dare", name="repairs") is not False:
        raise ValueError("configs/repairs.yaml resta.dare must be false: DARE is never enabled "
                         "(locked Oct 28)")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"ReSta alpha must be in (0, 1) (alpha 1 breaks Tiny Aya), got {alpha}")
    _check_references(adapter_model, base_cpu, stock_cpu)
    return apply_resta(adapter_model, base=base_cpu, aligned=stock_cpu, alpha=alpha, dare=DARE, device="cpu")


def _check_references(model: Any, base: Any, aligned: Any) -> None:
    b = {k: tuple(v.shape) for k, v in base.state_dict().items()}
    a = {k: tuple(v.shape) for k, v in aligned.state_dict().items()}
    if b != a:
        only = sorted(set(a) ^ set(b))[:5]
        shapes = sorted(k for k in set(a) & set(b) if a[k] != b[k])[:5]
        raise ValueError(f"Base and Global do not hold the same parameters (names in one only {only}, other "
                         f"shapes {shapes}): the ReSta delta Global - Base is undefined")
    missing = [n for n, p in model.named_parameters() if n not in a or tuple(p.shape) != a[n]]
    if missing:
        raise ValueError("the adapter model has parameters Global/Base lack or shape differently: "
                         f"{missing[:5]}")
