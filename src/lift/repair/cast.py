"""CAST: conditional activation steering with one global threshold. Implemented in P05.

SafeTune 0.1.12 (all badged Faithful):
- ``fit_cast_condition(model, harmful_prompts, benign_prompts, tokenizer, candidate_layers=None,
  pool="last", chat_template=None, device=...)`` grid-searches (layer, threshold, comparator) by
  held-out F1 and returns a CASTCondition. ``chat_template=None`` means True (fit on the same
  chat-formatted prompts the model sees at inference). Do not set ``legacy_cast_gate``.
- ``extract_caa_vectors(model, tokenizer, positive_prompts, negative_prompts, config)`` gives the
  per-layer steering vectors (difference of means).
- ``CASTModel(model, steering_vectors, condition=cond, alpha=strength)``; ``cast.generate(...)``
  gates each prompt of a batch on its own (per_prompt_gate defaults to True). ``with cast:``
  installs the vectors unconditionally: that is the gate-always-on setting.

Decisions (docs/decisions.md): the condition and the vectors are fitted per seed on the adapter
being repaired, from train/fit data only (PolyRefuse harmful train vs harmless train + the 100
OR-Bench fit prompts, all 6 languages pooled). Strengths {0.5, 1.0, 2.0} plus always-on at 1.0.
Artifacts (condition vector, layer, threshold, comparator, steering vectors) are saved with
torch.save and their sha256 goes into the VariantSpec, so they reach the cache key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def fit(model: Any, tokenizer: Any, harmful: list[str], benign: list[str], *, out_dir: Path) -> Path:
    """Fit the condition and the CAA vectors; save them; return the artifact path."""
    raise NotImplementedError("P05")


def make(model: Any, artifact_path: Path, strength: float, *, gate: bool = True) -> Any:
    """A CASTModel (gate on) or an always-on wrapper with the same ``generate`` interface."""
    raise NotImplementedError("P05")
