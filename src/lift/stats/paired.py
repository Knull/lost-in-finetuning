"""Paired statistics for every reported comparison. Used instead of AuditKIT's paired_bootstrap,
which resamples uncentred differences and returns no intervals (verified Oct 28).

Conventions: inputs are per-row outcomes for two variants on the same rows (row i of ``x`` and
row i of ``y`` are the same prompt in the same language and seed). ``clusters`` groups rows that
must be resampled together; pass the pid so one prompt seen in several languages or seeds counts
once. Rates are fractions in [0, 1]; differences are x minus y. The multiplicity corrections (``holm``,
``benjamini_hochberg``) take a family of p-values in [0, 1] and refuse anything else: a test with no
p-value is listed as not run and left out of its family before the correction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest


@dataclass(frozen=True)
class McNemarResult:
    n: int
    x_only: int          # rows where x is 1 and y is 0
    y_only: int          # rows where y is 1 and x is 0
    diff: float          # mean(x) - mean(y)
    p_value: float       # exact two-sided binomial test on the discordant pairs


@dataclass(frozen=True)
class BootstrapResult:
    diff: float
    lo: float
    hi: float
    p_value: float       # two-sided, from the bootstrap distribution centred on the estimate
    n_rows: int
    n_clusters: int
    n_boot: int


def _paired_arrays(x: Sequence, y: Sequence) -> tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.shape != ya.shape or xa.ndim != 1:
        raise ValueError("x and y must be 1-D and the same length (paired rows)")
    if xa.size == 0:
        raise ValueError("no rows")
    if np.isnan(xa).any() or np.isnan(ya).any():
        raise ValueError("NaN in outcomes; every prompt must stay in the denominator")
    return xa, ya


def mcnemar_exact(x: Sequence[bool], y: Sequence[bool]) -> McNemarResult:
    xa, ya = _paired_arrays(x, y)
    if not set(np.unique(xa)) <= {0.0, 1.0} or not set(np.unique(ya)) <= {0.0, 1.0}:
        raise ValueError("McNemar needs binary outcomes")
    x_only = int(((xa == 1) & (ya == 0)).sum())
    y_only = int(((xa == 0) & (ya == 1)).sum())
    discordant = x_only + y_only
    p = 1.0 if discordant == 0 else float(binomtest(x_only, discordant, 0.5).pvalue)
    return McNemarResult(n=int(xa.size), x_only=x_only, y_only=y_only,
                         diff=float(xa.mean() - ya.mean()), p_value=p)


def paired_bootstrap(
    x: Sequence[float],
    y: Sequence[float],
    clusters: Sequence[str] | None = None,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
    chunk: int = 1_000,
) -> BootstrapResult:
    """Cluster bootstrap of mean(x) - mean(y) with a percentile interval and a centred p-value.

    Clusters are resampled with replacement; the statistic is the ratio of resampled difference
    sums to resampled row counts, so clusters of different sizes are weighted by their rows.
    p = (1 + #{|d* - d| >= |d|}) / (B + 1).
    """
    xa, ya = _paired_arrays(x, y)
    d = xa - ya
    if clusters is None:
        cluster_ids = np.arange(d.size)
    else:
        if len(clusters) != d.size:
            raise ValueError("clusters must have one entry per row")
        _, cluster_ids = np.unique(np.asarray(clusters, dtype=object).astype(str), return_inverse=True)
    k = int(cluster_ids.max()) + 1
    sums = np.bincount(cluster_ids, weights=d, minlength=k)
    counts = np.bincount(cluster_ids, minlength=k).astype(float)
    est = float(sums.sum() / counts.sum())

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, k, size=(b, k))
        boots[start : start + b] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)

    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    extreme = np.abs(boots - est) >= abs(est) - 1e-12
    p = float((1 + extreme.sum()) / (n_boot + 1))
    return BootstrapResult(diff=est, lo=float(lo), hi=float(hi), p_value=min(p, 1.0),
                           n_rows=int(d.size), n_clusters=k, n_boot=n_boot)


def _p_values(p_values: Sequence[float]) -> np.ndarray:
    """A family of p-values as a 1-D float array. A value that is not a number in [0, 1] is a ValueError:
    NaN (or None) is a test that gave no p-value, which a correction must neither make one up for nor let
    erase the others'."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p-values must be a 1-D sequence, one per test of the family")
    for i, v in enumerate(p.tolist()):
        if not (math.isfinite(v) and 0.0 <= v <= 1.0):
            raise ValueError(f"p-value {i} of the family is {v!r}, not a number in [0, 1]: a test with no "
                             "p-value (an arm not run, a degenerate test) is listed as not run and left out "
                             "of its family before the correction, never adjusted")
    return p


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order. Every p-value must be a number in [0, 1]
    (``_p_values``): a family holding a test with none is a ValueError."""
    p = _p_values(p_values)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """BH-FDR adjusted p-values (q-values), in the input order. Every p-value must be a number in [0, 1]
    (``_p_values``): a family holding a test with none is a ValueError."""
    p = _p_values(p_values)
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted.tolist()


def rates(labels: Sequence[str]) -> dict[str, float | int]:
    """Class rates with every prompt in the denominator. Harmful upper bound is (H + U) / N."""
    n = len(labels)
    if n == 0:
        raise ValueError("no labels")
    counts = {c: 0 for c in ("harmful", "refusal", "benign", "unclear")}
    for label in labels:
        if label not in counts:
            raise ValueError(f"unknown label {label!r}")
        counts[label] += 1
    return {
        "n": n,
        **{f"n_{c}": v for c, v in counts.items()},
        "harmful_rate": counts["harmful"] / n,
        "harmful_upper": (counts["harmful"] + counts["unclear"]) / n,
        "refusal_rate": counts["refusal"] / n,
        "unclear_rate": counts["unclear"] / n,
    }


def win_rate(scores: Sequence[float], clusters: Sequence[str] | None = None, *,
             n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """Mean pairwise score against stock (1 win, 0.5 tie, 0 loss, averaged over both orders),
    with a cluster bootstrap interval. Gain over parity is ``rate - 0.5``."""
    s = np.asarray(scores, dtype=float)
    res = paired_bootstrap(s, np.full_like(s, 0.5), clusters, n_boot=n_boot, seed=seed, alpha=alpha)
    return {"win_rate": float(s.mean()), "lo": res.lo + 0.5, "hi": res.hi + 0.5,
            "p_vs_parity": res.p_value, "n": int(s.size)}
