"""Bootstrap confidence intervals and significance tests.

Uses :class:`statistics.NormalDist` from the standard library for the normal
CDF / inverse-CDF, so there is **no SciPy dependency**. The default interval is
bias-corrected and accelerated (BCa); acceleration uses a jackknife, which is
skipped for very large subgroups (falling back to bias-corrected percentile) to
keep audits fast — this choice is reported in the returned interval's ``method``.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fairness_core.seeds import DEFAULT_SEED
from fairness_core.types import ConfidenceInterval

__all__ = [
    "MIN_SUBGROUP_FOR_ERROR_METRICS",
    "MIN_SUBGROUP_FOR_RATES",
    "bootstrap_ci",
    "disparate_impact_ci",
    "equalized_odds_ci",
    "has_sufficient_data",
    "two_proportion_ztest",
]

#: A group below this many members yields unstable selection-rate estimates.
MIN_SUBGROUP_FOR_RATES: int = 100
#: Error-based metrics (TPR/FPR) need at least this many positives *and* negatives.
MIN_SUBGROUP_FOR_ERROR_METRICS: int = 30

_NORMAL = NormalDist()


def has_sufficient_data(
    n: int,
    n_positive: int | None = None,
    n_negative: int | None = None,
    *,
    error_metric: bool = False,
) -> bool:
    """Whether a subgroup is large enough for a stable estimate."""
    if error_metric:
        if n_positive is None or n_negative is None:
            return False
        return (
            n_positive >= MIN_SUBGROUP_FOR_ERROR_METRICS
            and n_negative >= MIN_SUBGROUP_FOR_ERROR_METRICS
        )
    return n >= MIN_SUBGROUP_FOR_RATES


def two_proportion_ztest(count1: int, n1: int, count2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z-test. Returns ``(z, two_sided_p_value)``."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = count1 / n1, count2 / n2
    p_pool = (count1 + count2) / (n1 + n2)
    se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p = 2.0 * (1.0 - _NORMAL.cdf(abs(z)))
    return float(z), float(p)


def _clip_prob(p: float, n_boot: int) -> float:
    eps = 1.0 / (n_boot + 1)
    return min(max(p, eps), 1.0 - eps)


def bootstrap_ci(
    statistic: Callable[..., float],
    *arrays: ArrayLike,
    n_boot: int = 2000,
    level: float = 0.95,
    method: str = "bca",
    seed: int = DEFAULT_SEED,
    jackknife_max: int = 2000,
) -> ConfidenceInterval:
    """Confidence interval for ``statistic`` via a paired bootstrap.

    All input ``arrays`` are resampled with a *shared* index set (paired
    bootstrap), so row-aligned arrays such as ``(sensitive, y_pred, y_true)``
    stay aligned. Non-finite bootstrap replicates (e.g. a resample with an empty
    privileged group) are discarded.
    """
    cols: list[NDArray] = [np.asarray(a) for a in arrays]
    n = len(cols[0])
    rng = np.random.default_rng(seed)
    theta_hat = statistic(*cols)

    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = statistic(*[c[idx] for c in cols])
    boot = boot[np.isfinite(boot)]
    if boot.size < 2 or not np.isfinite(theta_hat):
        return ConfidenceInterval(
            low=float("nan"), high=float("nan"), level=level, method="bootstrap-degenerate"
        )

    alpha = 1.0 - level
    used = method

    if method == "bca":
        prop_less = float(np.mean(boot < theta_hat))
        try:
            z0 = _NORMAL.inv_cdf(_clip_prob(prop_less, boot.size))
        except Exception:
            z0 = 0.0

        a = 0.0
        if n <= jackknife_max:
            jack = np.empty(n, dtype=float)
            base = np.arange(n)
            for i in range(n):
                m = base != i
                jack[i] = statistic(*[c[m] for c in cols])
            jack = jack[np.isfinite(jack)]
            jbar = jack.mean()
            num = float(np.sum((jbar - jack) ** 3))
            den = 6.0 * float(np.sum((jbar - jack) ** 2)) ** 1.5
            a = num / den if den != 0 else 0.0
        else:
            used = "bootstrap-bc"  # bias-corrected only (no acceleration)

        def adjusted(z_a: float) -> float:
            denom = 1.0 - a * (z0 + z_a)
            if denom == 0:
                denom = 1e-12
            return _NORMAL.cdf(z0 + (z0 + z_a) / denom)

        p_low = _clip_prob(adjusted(_NORMAL.inv_cdf(alpha / 2.0)), boot.size)
        p_high = _clip_prob(adjusted(_NORMAL.inv_cdf(1.0 - alpha / 2.0)), boot.size)
        method_name = f"bootstrap-{'bca' if used == 'bca' else 'bc'}"
    else:
        p_low, p_high = alpha / 2.0, 1.0 - alpha / 2.0
        method_name = "bootstrap-percentile"

    lo = float(np.quantile(boot, p_low))
    hi = float(np.quantile(boot, p_high))
    if lo > hi:
        lo, hi = hi, lo
    return ConfidenceInterval(low=lo, high=hi, level=level, method=method_name)


def _di_statistic(sensitive: NDArray, y_pred: NDArray, priv: object, unpriv: object) -> float:
    p_mask, u_mask = sensitive == priv, sensitive == unpriv
    if not p_mask.any() or not u_mask.any():
        return float("nan")
    p_rate = float(np.mean(y_pred[p_mask]))
    if p_rate == 0:
        return float("nan")
    return float(np.mean(y_pred[u_mask])) / p_rate


def disparate_impact_ci(
    sensitive: ArrayLike,
    y_pred: ArrayLike,
    privileged_value: object,
    unprivileged_value: object,
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> ConfidenceInterval:
    """Bootstrap CI for the Disparate Impact ratio of one group vs the privileged group."""
    s = np.asarray(sensitive)
    yp = (np.asarray(y_pred) == 1).astype(int)

    def stat(sens: NDArray, pred: NDArray) -> float:
        return _di_statistic(sens, pred, privileged_value, unprivileged_value)

    return bootstrap_ci(stat, s, yp, n_boot=n_boot, level=level, seed=seed)


def _eo_statistic(
    sensitive: NDArray, y_pred: NDArray, y_true: NDArray, priv: object, unpriv: object
) -> float:
    p_mask, u_mask = sensitive == priv, sensitive == unpriv
    if not p_mask.any() or not u_mask.any():
        return float("nan")

    def _rates(mask: NDArray) -> tuple[float | None, float | None]:
        yt, yp = y_true[mask], y_pred[mask]
        pos, neg = yt == 1, yt == 0
        if not pos.any() or not neg.any():
            return None, None
        return float(np.mean(yp[pos])), float(np.mean(yp[neg]))

    tpr_p, fpr_p = _rates(p_mask)
    tpr_u, fpr_u = _rates(u_mask)
    if None in (tpr_p, fpr_p, tpr_u, fpr_u):
        return float("nan")
    return max(abs(tpr_p - tpr_u), abs(fpr_p - fpr_u))  # type: ignore[operator]


def equalized_odds_ci(
    sensitive: ArrayLike,
    y_pred: ArrayLike,
    y_true: ArrayLike,
    privileged_value: object,
    unprivileged_value: object,
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> ConfidenceInterval:
    """Bootstrap CI for the Equalized-Odds gap (max TPR/FPR difference) of one group."""
    s = np.asarray(sensitive)
    yp = (np.asarray(y_pred) == 1).astype(int)
    yt = (np.asarray(y_true) == 1).astype(int)

    def stat(sens: NDArray, pred: NDArray, true: NDArray) -> float:
        return _eo_statistic(sens, pred, true, privileged_value, unprivileged_value)

    return bootstrap_ci(stat, s, yp, yt, n_boot=n_boot, level=level, seed=seed)
