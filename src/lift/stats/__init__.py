"""Paired statistics and agreement. Implemented and tested; see tests/test_stats.py."""

from lift.stats.agreement import cohens_kappa, meets_bar, percent_agreement
from lift.stats.paired import (
    benjamini_hochberg,
    holm,
    mcnemar_exact,
    paired_bootstrap,
    rates,
    win_rate,
)

__all__ = [
    "benjamini_hochberg",
    "cohens_kappa",
    "holm",
    "mcnemar_exact",
    "meets_bar",
    "paired_bootstrap",
    "percent_agreement",
    "rates",
    "win_rate",
]
