"""Error-based fairness metrics: Equalized Odds, Equal Opportunity, etc.

These metrics compare *error rates* (true-positive / false-positive rates)
across groups and therefore **require ground-truth labels** ``y_true``. This is
the crux of the methodological fix EthicLens makes over its original spec:
on a label-free synthetic cohort these are mathematically uncomputable, and the
engine must return :attr:`Classification.INSUFFICIENT_DATA` rather than
fabricate a value.

Definitions (privileged group ``p``, unprivileged group ``u``):

* Equalized Odds difference = ``max(|TPR_p - TPR_u|, |FPR_p - FPR_u|)``
  (matches Fairlearn's ``equalized_odds_difference`` with ``agg='worst_case'``).
* Equal Opportunity difference = ``|TPR_p - TPR_u|``.
"""

from __future__ import annotations

from fairness_core.types import EO_THRESHOLD, Classification

__all__ = [
    "average_odds_difference",
    "calculate_eo",
    "classify_equalized_odds",
    "equal_opportunity_difference",
]


def classify_equalized_odds(eo: float, threshold: float = EO_THRESHOLD) -> str:
    """``Flagged`` if the Equalized-Odds gap exceeds ``threshold`` else ``Acceptable``."""
    return Classification.FLAGGED.value if eo > threshold else Classification.ACCEPTABLE.value


def calculate_eo(
    tpr_privileged: float,
    fpr_privileged: float,
    tpr_unprivileged: float,
    fpr_unprivileged: float,
) -> float:
    """Equalized Odds difference: the larger of the TPR gap and the FPR gap.

    Examples:
        >>> calculate_eo(0.90, 0.20, 0.75, 0.10)
        0.15
    """
    tpr_gap = abs(tpr_privileged - tpr_unprivileged)
    fpr_gap = abs(fpr_privileged - fpr_unprivileged)
    return max(tpr_gap, fpr_gap)


def equal_opportunity_difference(tpr_privileged: float, tpr_unprivileged: float) -> float:
    """Absolute true-positive-rate gap between groups."""
    return abs(tpr_privileged - tpr_unprivileged)


def average_odds_difference(
    tpr_privileged: float,
    fpr_privileged: float,
    tpr_unprivileged: float,
    fpr_unprivileged: float,
) -> float:
    """AIF360-style average odds difference: mean of the signed TPR and FPR gaps.

    Exposed alongside :func:`calculate_eo` but **never conflated** with it —
    EthicLens labels each metric by its precise definition.
    """
    return 0.5 * ((fpr_unprivileged - fpr_privileged) + (tpr_unprivileged - tpr_privileged))
