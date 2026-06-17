"""Fairness metrics — implemented from scratch, validated against Fairlearn.

Two layers:

* **Scalar primitives** whose signatures match the EthicLens STP exactly
  (:func:`calculate_disparate_impact`, :func:`calculate_spd`,
  :func:`compute_composite_bias_score`, and the ``classify_*`` helpers).
* **Array-level helpers** (:mod:`fairness_core.metrics.group`) that turn raw
  prediction arrays into the rates those primitives consume.
"""

from __future__ import annotations

from fairness_core.metrics.composite import (
    DEFAULT_COMPOSITE_WEIGHTS,
    classify_composite_score,
    compute_composite_bias_score,
)
from fairness_core.metrics.disparate_impact import (
    DI_THRESHOLD,
    calculate_disparate_impact,
    classify_disparate_impact,
)
from fairness_core.metrics.equalized_odds import (
    average_odds_difference,
    calculate_eo,
    equal_opportunity_difference,
)
from fairness_core.metrics.extra import (
    expected_calibration_error,
    fpr_balance_difference,
    predictive_parity_difference,
    theil_index,
)
from fairness_core.metrics.group import (
    GroupRates,
    compute_group_rates,
    confusion_rates,
    selection_rate,
)
from fairness_core.metrics.spd import SPD_ACCEPTABLE, calculate_spd, classify_spd

__all__ = [
    # Disparate Impact
    "calculate_disparate_impact",
    "classify_disparate_impact",
    "DI_THRESHOLD",
    # SPD
    "calculate_spd",
    "classify_spd",
    "SPD_ACCEPTABLE",
    # Equalized Odds family
    "calculate_eo",
    "equal_opportunity_difference",
    "average_odds_difference",
    # Composite
    "compute_composite_bias_score",
    "classify_composite_score",
    "DEFAULT_COMPOSITE_WEIGHTS",
    # Extra / depth
    "predictive_parity_difference",
    "fpr_balance_difference",
    "expected_calibration_error",
    "theil_index",
    # Array helpers
    "GroupRates",
    "compute_group_rates",
    "confusion_rates",
    "selection_rate",
]
