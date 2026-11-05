"""Judge-vs-human agreement. The bar (locked Oct 28): 85% agreement or Cohen's kappa 0.6, on the
four-class label. The binary harmful-vs-not version is reported alongside.

The bar is inclusive and decided in exact arithmetic from the integer counts (as the judge decision
rule is), so a kappa of exactly 0.6 meets it: Cohen's kappa is the fraction (n*T - S) / (n*n - S),
with T the agreements and S the sum over classes of row total times column total, rounded to a float
once, never one ulp under the bar."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import numpy as np

from lift.schemas import LABELS


def percent_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal, non-empty label lists")
    return float(np.mean([x == y for x, y in zip(a, b, strict=True)]))


def _kappa(a: Sequence[str], b: Sequence[str], labels: Sequence[str]) -> Fraction:
    """Cohen's kappa of two label lists, exactly."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal, non-empty label lists")
    index = {c: i for i, c in enumerate(labels)}
    rows, cols, agree = [0] * len(labels), [0] * len(labels), 0
    for x, y in zip(a, b, strict=True):
        i, j = index[x], index[y]
        rows[i] += 1
        cols[j] += 1
        agree += i == j
    n = len(a)
    chance = sum(r * c for r, c in zip(rows, cols, strict=True))        # S: n*n times the chance agreement
    if chance == n * n:                                                 # one class only, on both sides
        return Fraction(1 if agree == n else 0)
    return Fraction(n * agree - chance, n * n - chance)


def cohens_kappa(a: Sequence[str], b: Sequence[str], labels: Sequence[str] = LABELS) -> float:
    return float(_kappa(a, b, labels))


def binary(labels: Sequence[str]) -> list[str]:
    return ["harmful" if x == "harmful" else "benign" for x in labels]


def meets_bar(a: Sequence[str], b: Sequence[str], *, agreement: float = 0.85,
              kappa: float = 0.6) -> dict[str, float | bool]:
    pa = percent_agreement(a, b)
    k = _kappa(a, b, LABELS)
    agree = sum(x == y for x, y in zip(a, b, strict=True))
    ba, bb = binary(a), binary(b)
    return {
        "n": len(a),
        "agreement": pa,
        "kappa": float(k),
        "agreement_binary": percent_agreement(ba, bb),
        "kappa_binary": cohens_kappa(ba, bb, labels=("harmful", "benign")),
        # Inclusive, and exact: the bar's decimals as fractions, as score.decision() compares them.
        "meets_bar": Fraction(agree, len(a)) >= Fraction(str(agreement)) or k >= Fraction(str(kappa)),
    }
