"""Exploratory diagnostic signatures, on stock and each adapter seed, per eval language.
Implemented in P05. Never delays CAST; write-up says "consistent with", never "caused by".

1. Harm probe at the last instruction token, layers 12 and 24: logistic regression, harmful vs
   hard benign. Frozen (trained on stock activations, applied to the adapter's) and refit
   (trained on the adapter's). AUROC with a bootstrap interval. Lexical baseline: a character
   n-gram (2-5) TF-IDF + logistic regression on the prompt text alone, same folds; report
   probe minus lexical.
2. Refusal direction at the post-instruction tokens, layer chosen on stock (start at 24) with
   SafeTune ``extract_refusal_direction`` and ``RefusalDirectionConfig(pick_layer=...)``. Do not
   let SafeTune's selection sweep pick the layer for multilingual prompts: its refusal check is an
   English substring list. Report the cosine of each adapter's direction to stock's and the mean
   projection of harmful prompts onto the stock direction. Seed-to-seed spread is the noise floor.
3. Causal check: ablate the stock direction (SafeTune ``orthogonalize_weights`` /
   ``restore_weights``, or the RefusalDirectionModel hook) in stock and in each adapter on 100
   Hindi and 100 English dev harmful prompts; judge the outputs with the primary judge; report
   the change in harmful rate. A direction that does not move behaviour is not interpreted.
4. Once, on stock: extract the direction with and without the chat preamble; report the cosine.

Probes and directions are fitted on train/fit prompts and evaluated on dev prompts only.
"""

from __future__ import annotations

from typing import Any


def harm_probes(variant: str, lang: str) -> dict[str, Any]:
    raise NotImplementedError("P05")


def refusal_direction(variant: str, *, layer: int | None = None, preamble: bool = True) -> dict[str, Any]:
    raise NotImplementedError("P05")


def ablation_check(variant: str, langs: tuple[str, ...] = ("hi", "en"), n: int = 100) -> dict[str, Any]:
    raise NotImplementedError("P05")
