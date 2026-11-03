"""AuditKIT as the evaluation record. Implemented in P02 (safety) and P03 (utility).

AuditKIT 1.1.3 runs ``Dataset -> Adapter -> Model -> Metrics -> RunResult`` via
``auditkit.evaluate(model=..., prompts=..., metrics=...)``; any ``list[str] -> list[str]``
callable works as the model. We wrap our cached generations as that callable (it looks up the
output by prompt) and attach metrics that read our final labels, so AuditKIT records exactly the
numbers we report, per variant, language and set. The utility win rate also goes through
AuditKIT's pairwise/preference metrics. Save each RunResult under outputs/metrics/auditkit/.

Do NOT use auditkit's paired_bootstrap (src/auditkit/_bootstrap.py): it resamples uncentred
differences (p near 0.5 when every difference is positive) and returns no intervals. Our paired
statistics are lift.stats. File the issue upstream (submission checklist).
"""

from __future__ import annotations

from pathlib import Path


def record_safety(variant: str, set_name: str, lang: str, split: str) -> Path:
    raise NotImplementedError("P02")


def record_utility(variant: str, split: str) -> Path:
    raise NotImplementedError("P03")
