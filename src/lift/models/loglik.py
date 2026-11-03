"""Belebele by log-likelihood. Implemented in P03.

Format (fixed for every variant, no chat template, lm-eval style):

    P: {flores_passage}
    Q: {question}
    A: {mc_answer1}
    B: {mc_answer2}
    C: {mc_answer3}
    D: {mc_answer4}
    Answer:

Score the next-token log-probability of " A", " B", " C", " D" and take the argmax. Accuracy per
language, 200 dev / 700 final. For CAST variants run the gate on the prompt and steer only if it
fires, exactly as at generation time.
"""

from __future__ import annotations

from collections.abc import Sequence

from lift.models.variants import LoadedVariant


def score_belebele(variant: LoadedVariant, rows: Sequence[dict]) -> list[dict]:
    """Per item: pid, predicted letter, correct letter, the four log-probs."""
    raise NotImplementedError("P03")
