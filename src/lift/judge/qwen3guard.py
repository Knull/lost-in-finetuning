"""Qwen3Guard-Gen-4B response moderation, the primary safety judge. Implemented in P02.

Runs on the native-language prompt and response (no translation). The model's response
moderation output has the form

    Safety: Safe|Unsafe|Controversial
    Categories: ...
    Refusal: Yes|No

(check the exact format and the chat template against the model card before parsing; fail loudly
on an unparseable output instead of defaulting). Map with lift.judge.labels.map_qwen3guard,
strict for the primary analysis; store Safety, Categories and Refusal in ``native`` so the loose
mapping can be recomputed without re-judging. Put the judge on cuda:1 when the generator has
cuda:0 (Kaggle T4x2: two separate 16 GB devices).
"""

from __future__ import annotations

from collections.abc import Sequence


class Qwen3GuardJudge:
    def __init__(self, model_id: str, revision: str, *, device: str = "cuda:1",
                 dtype: str = "fp16") -> None:
        raise NotImplementedError("P02")

    @property
    def judge_id(self) -> str:
        """``{model_id}@{commit sha}``."""
        raise NotImplementedError("P02")

    def judge(self, prompts: Sequence[str], responses: Sequence[str]) -> list[dict]:
        """Per pair: {"safety", "categories", "refusal", "raw"}."""
        raise NotImplementedError("P02")
