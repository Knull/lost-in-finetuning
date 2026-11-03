"""Frontier LLM judges over an API: the second safety judge and the utility judge. P02 (safety),
P03 (utility).

- Prompts come verbatim from the frozen rubrics (configs/rubrics/*.md); temperature 0; JSON
  output parsed strictly, one retry with a reminder on a parse failure, then the item is recorded
  as a judge failure (never guessed).
- Every call is cached on disk by sha256(rubric version, model, prompt, response) so reruns are
  free and the API spend is bounded. Log tokens and cost per call into the run manifest.
- Concurrency with backoff on rate limits; a hard spend cap from an env var
  (LIFT_JUDGE_MAX_USD) stops the run before it is exceeded.
- The provider and model are set in configs/project.yaml (judges.safety_second / judges.utility).
- If the API refuses to judge an item, record that as a judge failure with the reason.
"""

from __future__ import annotations

from collections.abc import Sequence


class FrontierJudge:
    def __init__(self, provider: str, model: str, rubric_path: str) -> None:
        raise NotImplementedError("P02")

    @property
    def judge_id(self) -> str:
        raise NotImplementedError("P02")

    def safety(self, prompts: Sequence[str], responses: Sequence[str],
               prompt_langs: Sequence[str]) -> list[dict]:
        """Per pair: {"label", "reason", "raw"} with label in HARMFUL/REFUSAL/BENIGN/UNCLEAR."""
        raise NotImplementedError("P02")

    def pairwise(self, instructions: Sequence[str], a: Sequence[str], b: Sequence[str]) -> list[dict]:
        """Per pair: {"winner": "A"|"B"|"TIE", "reason", "raw"}."""
        raise NotImplementedError("P03")
