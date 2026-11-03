"""Merge a LoRA adapter into Global and check parity. Implemented in P03.

Parity check (Gate 0 and every new adapter): greedy outputs of merged vs unmerged on 20 dev
prompts must match token for token in fp32; in fp16 report the share that match and the first
divergence position. The adapter hash (sha256 of adapter_model.safetensors) goes into every cache
key for variants built on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def merged_model(adapter_dir: Path, *, device: str = "cuda:0", dtype: str = "fp16") -> Any:
    raise NotImplementedError("P03")


def adapter_hash(adapter_dir: Path) -> str:
    raise NotImplementedError("P03")


def parity_check(adapter_dir: Path, n_prompts: int = 20) -> dict:
    raise NotImplementedError("P03")
