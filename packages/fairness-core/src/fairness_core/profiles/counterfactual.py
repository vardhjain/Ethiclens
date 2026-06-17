"""Counterfactual fairness probing.

This is what "flip the persona" *should* have meant. Instead of auditing on
fabricated people, we take **real** records, flip only the protected attribute
(e.g. Black -> White), hold everything else fixed, re-score, and measure how
often and how much the model's output changes. A model that changes its
decision purely because a protected attribute changed is individually unfair
(Dwork et al. 2012; Kusner et al. 2017 counterfactual fairness).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["CounterfactualResult", "counterfactual_probe"]


@dataclass(frozen=True)
class CounterfactualResult:
    attribute: str
    from_value: object
    to_value: object
    n: int
    #: Fraction of records whose hard decision flips when the attribute flips.
    flip_rate: float
    #: Mean absolute change in predicted score (if scores are available).
    mean_score_gap: float | None
    max_score_gap: float | None


def counterfactual_probe(
    predict_score: Callable[[pd.DataFrame], np.ndarray],
    data: pd.DataFrame,
    attribute: str,
    from_value: object,
    to_value: object,
    *,
    threshold: float = 0.5,
    scores_are_probabilities: bool = True,
) -> CounterfactualResult:
    """Flip ``attribute`` from ``from_value`` to ``to_value`` on matching rows.

    Args:
        predict_score: Callable returning a 1-D score (probability) per row.
        data: Real records.
        attribute: Protected attribute column to flip.
        from_value / to_value: The flip.
        threshold: Decision threshold for the hard label.
    """
    mask = data[attribute] == from_value
    subset = data[mask]
    n = len(subset)
    if n == 0:
        return CounterfactualResult(attribute, from_value, to_value, 0, float("nan"), None, None)

    flipped = subset.copy()
    # Broadcast the counterfactual value across the column; pandas-stubs is
    # over-strict about scalar assignment of an ``object``-typed value.
    flipped[attribute] = to_value  # type: ignore[call-overload]

    s_orig = np.asarray(predict_score(subset), dtype=float)
    s_cf = np.asarray(predict_score(flipped), dtype=float)

    d_orig = (s_orig >= threshold).astype(int)
    d_cf = (s_cf >= threshold).astype(int)
    flip_rate = float(np.mean(d_orig != d_cf))

    if scores_are_probabilities:
        gaps = np.abs(s_orig - s_cf)
        return CounterfactualResult(
            attribute,
            from_value,
            to_value,
            n,
            flip_rate,
            float(np.mean(gaps)),
            float(np.max(gaps)),
        )
    return CounterfactualResult(attribute, from_value, to_value, n, flip_rate, None, None)
