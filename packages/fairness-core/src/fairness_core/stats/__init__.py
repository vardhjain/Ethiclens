"""Statistical rigour for fairness estimates.

The original spec reported bare point estimates and flagged a group the instant
DI dipped below 0.80 — so 0.79 vs 0.81 (often pure subgroup noise) flipped the
verdict. This package adds the uncertainty quantification a defensible audit
needs: bootstrap confidence intervals, a two-proportion significance test, and a
minimum-subgroup policy.
"""

from __future__ import annotations

from fairness_core.stats.bootstrap import (
    MIN_SUBGROUP_FOR_ERROR_METRICS,
    MIN_SUBGROUP_FOR_RATES,
    bootstrap_ci,
    disparate_impact_ci,
    equalized_odds_ci,
    has_sufficient_data,
    two_proportion_ztest,
)

__all__ = [
    "MIN_SUBGROUP_FOR_ERROR_METRICS",
    "MIN_SUBGROUP_FOR_RATES",
    "bootstrap_ci",
    "disparate_impact_ci",
    "equalized_odds_ci",
    "has_sufficient_data",
    "two_proportion_ztest",
]
