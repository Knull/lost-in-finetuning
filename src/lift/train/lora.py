"""Hindi LoRA SFT on Tiny Aya Global. Implemented in P03.

Recipe in configs/train.yaml (standard and stress). Notes:
- TRL SFTTrainer + PEFT LoRA on q,k,v,o,gate,up,down (check the cohere2 module names), with the
  chat template ON so the preamble stays in distribution. Loss on assistant tokens only.
- Colab A100 (primary): bf16, gradient checkpointing, sequence 1024; raise the per-device batch
  and lower gradient accumulation so the effective batch stays 16.
- T4 (backup): fp16 base weights, LoRA weights in fp32, AMP, batch 1, gradient accumulation 16.
  The 262k vocabulary makes fp32 logits about 1 GB per 1,024-token sequence, so batch 4 runs out
  of memory on a T4.
- All three seeds of a recipe train on the same hardware and precision.
- Log the number of supervised (assistant) tokens per batch; fail if any batch has zero
  (the preamble crowding out the target).
- Seeds 1, 2, 3 set torch, numpy, python and data order.
- Save the adapter, the trainer log, and a manifest with the data file hash and the adapter hash.
- On Kaggle/Colab, ``pip uninstall -y torchao`` before importing SafeTune or TRL if imports fail.
"""

from __future__ import annotations

from pathlib import Path


def train_lora(candidate: str, recipe: str, seed: int, *, out_dir: Path | None = None) -> Path:
    """Return the adapter directory: outputs/adapters/adapter-{candidate}-{recipe}-s{seed}/."""
    raise NotImplementedError("P03")
