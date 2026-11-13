# `paired_bootstrap` tests the wrong null hypothesis: p is about 0.5 or 1.0 even for a large, consistent improvement

<!-- Not filed: auditkit 1.1.4 already centres the differences (checked Nov 13 against the 1.1.4
and 1.1.11 wheels' source; neither returns an interval). Kept as the reproduction against 1.1.3
(`src/auditkit/_bootstrap.py`, unchanged at commit 71316da), the version we pin and the one the
article describes (docs/decisions.md item 102). -->

**Version:** auditkit 1.1.3 (PyPI), Python 3.12.

## What happens

`auditkit._bootstrap.paired_bootstrap` resamples the paired differences as they are and counts
how often the resampled mean is at least as large in absolute value as the observed mean:

```python
diffs = [c - b for b, c in pairs]
observed = sum(diffs) / n
for _ in range(n_resamples):
    resample = [rng.choice(diffs) for _ in range(n)]
    if abs(sum(resample) / n) >= abs(observed):
        count += 1
p = (count + 1) / (n_resamples + 1)
```

Resampling the uncentred differences draws from a distribution centred on the observed mean,
not on zero, so about half the resamples land above it whatever the effect size. The test
therefore returns p near 0.5 for any effect whose differences vary, and p = 1.0 when every
difference is equal (every resample equals the observed mean). It never returns a small p, and
it returns no confidence interval.

`RunComparison.significance()`, `Experiment` contrasts and `model_compare` all call it, so every
"significant" flag they report is affected.

## Minimal reproduction

```python
from auditkit._bootstrap import paired_bootstrap

# The candidate beats the baseline on all 50 samples, by the same amount.
print(paired_bootstrap([(0.0, 1.0)] * 50))
# {'n': 50, 'mean_baseline': 0.0, 'mean_candidate': 1.0, 'delta': 1.0, 'p_value': 1.0, 'significant': False}

# 30 of 50 samples improve and none get worse.
print(paired_bootstrap([(0.0, 1.0)] * 30 + [(0.0, 0.0)] * 20))
# {'n': 50, ..., 'delta': 0.6, 'p_value': 0.544456, 'significant': False}
```

An exact sign test on the second case gives p well below 0.001 (30 improvements, 0 losses).

## Expected

A paired bootstrap test resamples the differences centred on zero (`d_i - mean(d)`) to build
the null distribution, or reports a percentile interval of the resampled means and calls the
effect significant when the interval excludes zero. Either way it should also return the
interval, since a p-value alone hides the size of the effect.

## Suggested fix

```python
centred = [d - observed for d in diffs]
means = []
for _ in range(n_resamples):
    means.append(sum(rng.choice(diffs) for _ in range(n)) / n)
    null = sum(rng.choice(centred) for _ in range(n)) / n
    if abs(null) >= abs(observed):
        count += 1
means.sort()
lo, hi = means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples) - 1]
p = (count + 1) / (n_resamples + 1)
# return lo and hi with the rest
```

When samples are not independent (one prompt asked in several languages, or several seeds), the
resampling should also be clustered by prompt; a `clusters` argument would cover it.

## Context

We found this while using AuditKIT as the evaluation record for a hackathon project; we use our
own clustered paired bootstrap for every reported interval and p-value instead.
