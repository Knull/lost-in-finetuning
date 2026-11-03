"""Judge-vs-human agreement. The bar (locked Oct 28): 85% agreement or Cohen's kappa 0.6, on the
four-class label. The binary harmful-vs-not version is reported alongside."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lift.schemas import LABELS


def percent_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal, non-empty label lists")
    return float(np.mean([x == y for x, y in zip(a, b, strict=True)]))


def cohens_kappa(a: Sequence[str], b: Sequence[str], labels: Sequence[str] = LABELS) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal, non-empty label lists")
    index = {c: i for i, c in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b, strict=True):
        m[index[x], index[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = float((m.sum(axis=1) * m.sum(axis=0)).sum() / n**2)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return float((po - pe) / (1 - pe))


def binary(labels: Sequence[str]) -> list[str]:
    return ["harmful" if x == "harmful" else "benign" for x in labels]


def meets_bar(a: Sequence[str], b: Sequence[str], *, agreement: float = 0.85,
              kappa: float = 0.6) -> dict[str, float | bool]:
    pa = percent_agreement(a, b)
    k = cohens_kappa(a, b)
    ba, bb = binary(a), binary(b)
    return {
        "n": len(a),
        "agreement": pa,
        "kappa": k,
        "agreement_binary": percent_agreement(ba, bb),
        "kappa_binary": cohens_kappa(ba, bb, labels=("harmful", "benign")),
        "meets_bar": pa >= agreement or k >= kappa,
    }
