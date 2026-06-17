"""Additional fairness & inequality metrics (depth beyond the STP core).

These power the responsible-AI "depth" story:

* **Predictive parity / FPR balance** — surfaced together on COMPAS to make the
  *impossibility theorem* visible (you cannot equalise predictive parity and
  the error rates simultaneously when base rates differ; Chouldechova 2017,
  Kleinberg et al. 2016).
* **Calibration / Expected Calibration Error (ECE)** — per-group reliability.
* **Theil index (a generalised-entropy index)** — an individual-level
  inequality measure of the benefit distribution (Speicher et al. 2018).
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "expected_calibration_error",
    "fpr_balance_difference",
    "predictive_parity_difference",
    "theil_index",
]


def predictive_parity_difference(ppv_privileged: float, ppv_unprivileged: float) -> float:
    """Absolute gap in positive predictive value (precision) between groups."""
    return abs(ppv_privileged - ppv_unprivileged)


def fpr_balance_difference(fpr_privileged: float, fpr_unprivileged: float) -> float:
    """Absolute gap in false-positive rate between groups."""
    return abs(fpr_privileged - fpr_unprivileged)


def expected_calibration_error(y_true: ArrayLike, y_score: ArrayLike, n_bins: int = 10) -> float:
    """Expected Calibration Error over equal-width score bins.

    ECE = sum over bins of ``(|bin| / N) * |accuracy(bin) - confidence(bin)|``.
    """
    yt = np.asarray(y_true, dtype=float)
    ys = np.asarray(y_score, dtype=float)
    if ys.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = ys.size
    for lo, hi in itertools.pairwise(edges):
        in_bin = (ys > lo) & (ys <= hi) if lo > 0 else (ys >= lo) & (ys <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        confidence = float(np.mean(ys[in_bin]))
        accuracy = float(np.mean(yt[in_bin]))
        ece += (count / n) * abs(accuracy - confidence)
    return ece


def theil_index(benefits: ArrayLike) -> float:
    """Theil index (generalised entropy index, alpha -> 1) of a benefit vector.

    ``benefits`` are typically ``y_pred - y_true + 1`` so that values are
    non-negative. Returns 0 for perfect equality.
    """
    b = np.asarray(benefits, dtype=float)
    mu = float(np.mean(b))
    if mu <= 0:
        return float("nan")
    ratio = b / mu
    # Use x*ln(x) with the convention 0*ln(0) = 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(ratio > 0, ratio * np.log(ratio), 0.0)
    return float(np.mean(terms))
