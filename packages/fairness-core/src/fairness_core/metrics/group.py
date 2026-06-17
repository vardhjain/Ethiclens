"""Array-level group statistics.

These helpers turn raw prediction arrays into the group rates that the scalar
metric primitives consume. Keeping the array math here (and the scalar formulas
in their own modules) is what lets us unit-test the formulas against the STP's
hard-coded values *and* validate the array path against Fairlearn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["GroupRates", "compute_group_rates", "confusion_rates", "selection_rate"]


def _binary(arr: ArrayLike) -> NDArray[np.int_]:
    a = np.asarray(arr)
    return (a == 1).astype(int) if a.dtype != bool else a.astype(int)


def selection_rate(y_pred: ArrayLike, mask: NDArray[np.bool_] | None = None) -> float:
    """Positive-prediction rate ``P(Y_hat = 1)`` over an optional subgroup mask."""
    yp = _binary(y_pred)
    if mask is not None:
        yp = yp[mask]
    if yp.size == 0:
        return float("nan")
    return float(np.mean(yp))


@dataclass(frozen=True)
class GroupRates:
    """Rates for a single subgroup. Error rates are ``None`` without labels."""

    n: int
    selection_rate: float
    tpr: float | None = None  # recall / sensitivity
    fpr: float | None = None
    ppv: float | None = None  # precision
    base_rate: float | None = None  # P(Y = 1)

    @property
    def has_error_rates(self) -> bool:
        return self.tpr is not None and self.fpr is not None


def confusion_rates(
    y_true: ArrayLike, y_pred: ArrayLike, mask: NDArray[np.bool_] | None = None
) -> tuple[float | None, float | None, float | None]:
    """Return ``(tpr, fpr, ppv)`` for a subgroup, or ``None`` where undefined."""
    yt, yp = _binary(y_true), _binary(y_pred)
    if mask is not None:
        yt, yp = yt[mask], yp[mask]
    pos, neg = yt == 1, yt == 0
    tpr = float(np.mean(yp[pos])) if pos.any() else None
    fpr = float(np.mean(yp[neg])) if neg.any() else None
    pred_pos = yp == 1
    ppv = float(np.mean(yt[pred_pos])) if pred_pos.any() else None
    return tpr, fpr, ppv


def compute_group_rates(
    sensitive: ArrayLike,
    y_pred: ArrayLike,
    group_value: object,
    y_true: ArrayLike | None = None,
) -> GroupRates:
    """Compute :class:`GroupRates` for ``sensitive == group_value``."""
    s = np.asarray(sensitive)
    mask = s == group_value
    n = int(mask.sum())
    sr = selection_rate(y_pred, mask)
    if y_true is None:
        return GroupRates(n=n, selection_rate=sr)
    tpr, fpr, ppv = confusion_rates(y_true, y_pred, mask)
    base = float(np.mean(_binary(y_true)[mask])) if n else None
    return GroupRates(n=n, selection_rate=sr, tpr=tpr, fpr=fpr, ppv=ppv, base_rate=base)
