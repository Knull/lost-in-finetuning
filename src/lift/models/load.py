"""Loading Tiny Aya and formatting prompts. Implemented in P02.

Tiny Aya facts that matter here (CohereLabs/tiny-aya-global, gated, CC-BY-NC-4.0):
- cohere2 / Cohere2ForCausalLM, 36 layers at ``model.model.layers``, d_model 2048, 262k vocab,
  tied embeddings, parallel attention+MLP blocks, 3:1 sliding-window:global attention.
- Weights are bf16 (~6.7 GB). Reported runs use bf16 on a Colab A100/L4. A T4 has no bf16, so it
  loads fp16, which may overflow on a bf16-trained model: Gate 0 compares T4 fp16 with bf16 on 200
  prompts (finite activations, same judge labels) before any T4 number is reported.
- The chat template always injects a default system preamble (~366 tokens, "contextual safety
  mode"). We keep it on for training and evaluation alike. Never strip it except in the explicit
  preamble on/off diagnosis.
- tiny-aya-base has no chat template; it is only ever loaded as the ReSta reference.
- Always resolve ``revision`` to a commit sha and record it; "main" is not a revision.
"""

from __future__ import annotations

from typing import Any


def load_tokenizer(model_id: str, revision: str) -> Any:
    """Tokenizer with ``padding_side="left"`` for batched generation."""
    raise NotImplementedError("P02")


def load_model(model_id: str, revision: str, *, dtype: str = "fp16", device: str = "cuda:0") -> Any:
    raise NotImplementedError("P02")


def resolved_revision(model_id: str, revision: str) -> str:
    """Commit sha for ``revision`` (huggingface_hub.model_info(...).sha)."""
    raise NotImplementedError("P02")


def chat_text(tokenizer: Any, user_text: str) -> str:
    """The exact string the model sees: chat template, default preamble, generation prompt."""
    raise NotImplementedError("P02")


def template_hash(tokenizer: Any) -> str:
    """sha256 over the chat template source and the rendering of a fixed probe message, so a
    changed preamble or template changes every cache key."""
    raise NotImplementedError("P02")
