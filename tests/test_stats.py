import numpy as np
import pytest

from lift.stats import (
    benjamini_hochberg,
    cohens_kappa,
    holm,
    mcnemar_exact,
    meets_bar,
    paired_bootstrap,
    percent_agreement,
    rates,
    win_rate,
)


def test_bootstrap_all_positive_differences_is_significant():
    # AuditKIT's uncentred bootstrap returns p ~ 0.5 here; the centred one must not.
    x = np.ones(172)
    y = np.zeros(172)
    res = paired_bootstrap(x, y, n_boot=2000, seed=1)
    assert res.diff == 1.0
    assert res.p_value < 0.001
    assert res.lo == res.hi == 1.0


def test_bootstrap_identical_is_null():
    x = np.array([1, 0, 1, 1, 0] * 20, dtype=float)
    res = paired_bootstrap(x, x.copy(), n_boot=2000, seed=1)
    assert res.diff == 0.0
    assert res.p_value == 1.0


def test_bootstrap_interval_covers_estimate_and_clusters_count_once():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.binomial(1, 0.35, n).astype(float)
    y = rng.binomial(1, 0.25, n).astype(float)
    res = paired_bootstrap(x, y, n_boot=3000, seed=2)
    assert res.lo <= res.diff <= res.hi
    # Same prompts repeated in 3 "languages": clustering by pid keeps 400 clusters.
    xs, ys = np.tile(x, 3), np.tile(y, 3)
    pids = [f"p{i}" for i in range(n)] * 3
    res_c = paired_bootstrap(xs, ys, pids, n_boot=3000, seed=2)
    assert res_c.n_clusters == n and res_c.n_rows == 3 * n
    assert res_c.diff == pytest.approx(res.diff)
    # Unclustered triplicates would pretend to have 3x the data and shrink the interval.
    res_u = paired_bootstrap(xs, ys, n_boot=3000, seed=2)
    assert (res_u.hi - res_u.lo) < (res_c.hi - res_c.lo)


def test_bootstrap_rejects_nan_and_shape_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, float("nan")], [0.0, 0.0])
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 0.0], [0.0])


def test_mcnemar_counts_and_p():
    x = [1] * 20 + [0] * 80
    y = [0] * 20 + [0] * 80
    res = mcnemar_exact(x, y)
    assert (res.x_only, res.y_only) == (20, 0)
    assert res.diff == pytest.approx(0.2)
    assert res.p_value < 1e-5
    assert mcnemar_exact([1, 0], [1, 0]).p_value == 1.0


def test_holm_and_bh_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    assert holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_rates_keep_every_prompt_in_denominator():
    r = rates(["harmful", "refusal", "unclear", "benign"])
    assert r["n"] == 4
    assert r["harmful_rate"] == 0.25
    assert r["harmful_upper"] == 0.5
    with pytest.raises(ValueError):
        rates(["harmful", "maybe"])


def test_win_rate_interval():
    scores = [1.0] * 60 + [0.5] * 20 + [0.0] * 20
    out = win_rate(scores, n_boot=2000)
    assert out["win_rate"] == pytest.approx(0.7)
    assert out["lo"] < 0.7 < out["hi"]


def test_kappa_and_bar():
    a = ["harmful", "refusal", "benign", "benign", "refusal", "harmful"]
    assert percent_agreement(a, a) == 1.0
    assert cohens_kappa(a, a) == pytest.approx(1.0)
    b = ["harmful", "refusal", "benign", "refusal", "refusal", "benign"]
    out = meets_bar(a, b)
    assert out["agreement"] == pytest.approx(4 / 6)
    assert -1.0 <= out["kappa"] <= 1.0
    assert out["meets_bar"] is (out["agreement"] >= 0.85 or out["kappa"] >= 0.6)
