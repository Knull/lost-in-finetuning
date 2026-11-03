"""The frozen generation harness. Implemented in P02; frozen once the stock dev eval starts.

After the freeze it only takes bug fixes, and each fix means re-running anything it touched.

Behaviour:
- Greedy, ``max_new_tokens`` 256, default chat template with the preamble, left padding.
- Batches sorted by prompt length; batch size chosen from free memory, OOM halves the batch and
  retries that batch (never skips a prompt).
- One GenerationRecord per prompt, appended to
  outputs/generations/{variant}/{set}_{lang}_{split}.jsonl as it finishes; on restart, keys already
  present are skipped (resumable across Kaggle sessions).
- ``hit_cap`` when generation stopped at the token limit; the cap-hit rate is reported.
- For CAST wrappers, record ``meta.gate_fired`` per prompt.
- Every run writes a manifest (lift.manifest) with prompts/sec and tokens/sec.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from lift.models.variants import LoadedVariant
from lift.schemas import PromptRecord


def generate(variant: LoadedVariant, prompts: Sequence[PromptRecord], *, out_path: Path,
             batch_size: int | None = None, resume: bool = True) -> Path:
    raise NotImplementedError("P02")
