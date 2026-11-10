"""Effects for the final report (P07), built on lift.stats.paired.

- ``paired_effect``: one paired comparison of binary outcomes (x minus y): the clustered bootstrap
  interval and p-value, and McNemar's exact p beside it.
- ``pooled_effect``: the README's primary endpoint shape: McNemar per seed (with each seed's bootstrap
  interval) and one clustered bootstrap pooled over seeds, clusters = pid, so a prompt seen in several
  seeds (or languages) is resampled as one.
- ``ratio_bootstrap``: a ratio of two paired sums, resampled jointly by cluster (the share of the
  adapter's utility gain a repair keeps: sum(repair - 0.5) / sum(adapter - 0.5) over the same pids).
- ``adjust``: Holm or BH applied over one family of claims, in place, as ``p_adj``. A claim whose p is
  None (an arm not run) stays out of the family and keeps ``p_adj`` None: it is never estimated.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import numpy as np

from lift.stats.paired import benjamini_hochberg, holm, mcnemar_exact, paired_bootstrap

N_BOOT = 10_000


def paired_effect(
    x: Sequence[float], y: Sequence[float], clusters: Sequence[str], *, n_boot: int = N_BOOT, seed: int = 0
) -> dict[str, Any]:
    """{est, lo, hi, p, p_mcnemar, n, n_clusters, x_rate, y_rate}; est = mean(x) - mean(y)."""
    boot = paired_bootstrap(x, y, clusters, n_boot=n_boot, seed=seed)
    binary = set(np.unique(np.asarray(x, dtype=float))) | set(np.unique(np.asarray(y, dtype=float)))
    mc = mcnemar_exact(x, y) if binary <= {0.0, 1.0} else None
    return {
        "est": boot.diff,
        "lo": boot.lo,
        "hi": boot.hi,
        "p": boot.p_value,
        "p_mcnemar": mc.p_value if mc else None,
        "n": boot.n_rows,
        "n_clusters": boot.n_clusters,
        "x_rate": float(np.mean(x)),
        "y_rate": float(np.mean(y)),
    }


def pooled_effect(
    per_seed: Mapping[int, tuple[Sequence[float], Sequence[float], Sequence[str]]],
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    """{"pooled": {est, lo, hi, p, n, n_clusters, seeds}, "per_seed": [{seed, est, lo, hi, p, p_mcnemar,
    p_bootstrap, n}]}. Per seed, ``p`` is McNemar's exact p (the README's test for a single paired
    comparison) and the interval is that seed's bootstrap; pooled, ``p`` is the clustered bootstrap's.
    No seeds: pooled None and per_seed empty (not run)."""
    rows = []
    xs: list[float] = []
    ys: list[float] = []
    cl: list[str] = []
    for s in sorted(per_seed):
        x, y, clusters = per_seed[s]
        e = paired_effect(x, y, clusters, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "seed": s,
                "est": e["est"],
                "lo": e["lo"],
                "hi": e["hi"],
                "p": e["p_mcnemar"] if e["p_mcnemar"] is not None else e["p"],
                "p_mcnemar": e["p_mcnemar"],
                "p_bootstrap": e["p"],
                "n": e["n"],
                "x_rate": e["x_rate"],
                "y_rate": e["y_rate"],
            }
        )
        xs += list(x)
        ys += list(y)
        cl += [str(c) for c in clusters]
    if not rows:
        return {"pooled": None, "per_seed": []}
    pooled = paired_effect(xs, ys, cl, n_boot=n_boot, seed=seed)
    return {
        "pooled": {k: pooled[k] for k in ("est", "lo", "hi", "p", "n", "n_clusters", "x_rate", "y_rate")}
        | {"seeds": [r["seed"] for r in rows]},
        "per_seed": rows,
    }


def ratio_bootstrap(
    num: Sequence[float],
    den: Sequence[float],
    clusters: Sequence[str],
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any] | None:
    """sum(num) / sum(den) with a percentile interval from resampling clusters (rows of one cluster move
    together). None when the denominator's sum is not positive (no gain to keep: the share is undefined).
    Resamples whose denominator is not positive are dropped from the interval and counted."""
    n = np.asarray(num, dtype=float)
    d = np.asarray(den, dtype=float)
    if n.shape != d.shape or n.ndim != 1 or not n.size or len(clusters) != n.size:
        raise ValueError("num, den and clusters must be 1-D and the same length")
    if d.sum() <= 0:
        return None
    _, ids = np.unique(np.asarray(clusters, dtype=object).astype(str), return_inverse=True)
    k = int(ids.max()) + 1
    ns = np.bincount(ids, weights=n, minlength=k)
    ds = np.bincount(ids, weights=d, minlength=k)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    bn, bd = ns[idx].sum(axis=1), ds[idx].sum(axis=1)
    ok = bd > 0
    ratios = bn[ok] / bd[ok]
    lo, hi = np.quantile(ratios, [alpha / 2, 1 - alpha / 2]) if ratios.size else (np.nan, np.nan)
    return {
        "est": float(n.sum() / d.sum()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(n.size),
        "n_clusters": k,
        "n_boot": n_boot,
        "n_undefined": int((~ok).sum()),
    }


def adjust(claims: Sequence[MutableMapping[str, Any]], method: str) -> None:
    """Holm ("holm") or BH-FDR ("bh") over the claims that have a p, written as ``p_adj``."""
    fn = {"holm": holm, "bh": benjamini_hochberg}[method]
    present = [c for c in claims if c.get("p") is not None]
    for c in claims:
        c["p_adj"] = None
        c["adjustment"] = method
    if present:
        for c, q in zip(present, fn([float(c["p"]) for c in present]), strict=True):
            c["p_adj"] = q
