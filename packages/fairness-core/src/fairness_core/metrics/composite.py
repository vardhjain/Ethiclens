"""Composite Bias Score.

The composite folds Disparate Impact, Statistical Parity Difference and
Equalized Odds into a single 0-1 *fairness-goodness* score (higher = fairer),
using configurable weights (defaults: DI 0.40, SPD 0.35, EO 0.25).

The original STP formula was incoherent for DI values above 1.0 (a *favoured*
flip). This implementation fixes that with a symmetric DI goodness term while
**preserving the exact signature, default weights, and golden value** asserted
in STP test script ``TS-UNIT-004``::

    compute_composite_bias_score(di=0.6, spd=-0.25, eo=0.15) == 0.7150

Note the naming subtlety: despite "bias" in the name, a *higher* score is
*better*. The composite is a triage convenience, **not** a legal standard — the
raw metrics and their confidence intervals are always reported alongside it.
"""

from __future__ import annotations

from collections.abc import Mapping

from fairness_core.types import DEFAULT_COMPOSITE_WEIGHTS, RiskBand

__all__ = [
    "DEFAULT_COMPOSITE_WEIGHTS",
    "classify_composite_score",
    "compute_composite_bias_score",
]


def _di_goodness(di: float) -> float:
    """Symmetric DI sub-score in [0, 1]; 1.0 at parity (DI == 1).

    ``min(di, 1/di)`` collapses both under- and over-selection to the same
    goodness, so a favoured group (DI > 1) is scored identically to its mirror.
    """
    if di <= 0:
        return 0.0
    return min(di, 1.0 / di)


def compute_composite_bias_score(
    di: float,
    spd: float,
    eo: float,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Return the weighted composite fairness-goodness score in [0, 1].

    Args:
        di: Disparate Impact ratio.
        spd: Statistical Parity Difference.
        eo: Equalized Odds difference.
        weights: Optional override of the ``{"di", "spd", "eo"}`` weights.

    Examples:
        >>> round(compute_composite_bias_score(0.6, -0.25, 0.15), 4)
        0.715
    """
    w = weights if weights is not None else DEFAULT_COMPOSITE_WEIGHTS
    g_di = _di_goodness(di)
    g_spd = 1.0 - min(abs(spd), 1.0)
    g_eo = 1.0 - min(abs(eo), 1.0)
    score = w["di"] * g_di + w["spd"] * g_spd + w["eo"] * g_eo
    return max(0.0, min(1.0, score))


def classify_composite_score(score: float) -> str:
    """Bucket a composite score into a risk band.

    ``< 0.60`` High Risk, ``< 0.80`` Medium Risk, ``>= 0.80`` Low Risk.

    Examples:
        >>> classify_composite_score(0.55)
        'High Risk'
        >>> classify_composite_score(0.85)
        'Low Risk'
    """
    if score < 0.60:
        return RiskBand.HIGH.value
    if score < 0.80:
        return RiskBand.MEDIUM.value
    return RiskBand.LOW.value
