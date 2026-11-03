"""Residual-stream activations at fixed token positions. Implemented in P05.

Positions (on the chat-formatted prompt, preamble on unless stated):
- ``last_instruction``: the last token of the user's text, before the end-of-turn token.
  Used by the harm probes (layers 12 and 24).
- ``post_instruction``: the template tokens after the user's text (end-of-turn, start-of-turn,
  chatbot token, start-of-response). Used for the refusal direction (Arditi et al.).
Locate these by tokenising the template around a sentinel, not by guessing offsets, and test the
positions on 5 prompts per language (the 262k vocabulary splits strings differently).
Activations are collected in fp32 from hooks on ``model.model.layers[i]`` outputs and saved as
.pt files under outputs/artifacts/activations/{variant}/.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def collect(model: Any, tokenizer: Any, prompts: Sequence[str], layers: Sequence[int], *,
            position: str = "last_instruction", preamble: bool = True,
            batch_size: int = 8) -> dict[int, Any]:
    """{layer: tensor (n_prompts, d_model)} in fp32 on CPU."""
    raise NotImplementedError("P05")
