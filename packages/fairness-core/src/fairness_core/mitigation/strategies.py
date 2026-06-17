"""Real mitigation strategies, measured on a held-out split.

The reported Disparate-Impact change is always the difference between the
*original* model and the *mitigated* predictor **on data the mitigation was not
fitted on**. Anything else would be leakage — and a fairness number you cannot
trust is worse than none.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fairness_core.metrics import calculate_disparate_impact, compute_group_rates
from fairness_core.seeds import DEFAULT_SEED
from fairness_core.stats import disparate_impact_ci
from fairness_core.types import DI_THRESHOLD, ConfidenceInterval

__all__ = [
    "reweighing_sample_weights",
    "MitigationResult",
    "mitigate_and_reaudit",
    "AVAILABLE_STRATEGIES",
]


def reweighing_sample_weights(y: ArrayLike, sensitive: ArrayLike) -> np.ndarray:
    """Kamiran & Calders (2012) reweighing weights (implemented from scratch).

    Each sample gets weight ``P(S=s)·P(Y=y) / P(S=s, Y=y)``, so that protected
    group and outcome become statistically independent in the *weighted* data.
    """
    y_arr = np.asarray(y)
    s_arr = np.asarray(sensitive)
    n = len(y_arr)
    weights = np.ones(n, dtype=float)
    for sv in np.unique(s_arr):
        for yv in np.unique(y_arr):
            mask = (s_arr == sv) & (y_arr == yv)
            n_sy = int(mask.sum())
            if n_sy == 0:
                continue
            expected = (s_arr == sv).sum() * (y_arr == yv).sum() / n
            weights[mask] = expected / n_sy
    return weights


def _group_di(sensitive: np.ndarray, y_pred: np.ndarray, priv: object, unpriv: object) -> float:
    p = compute_group_rates(sensitive, y_pred, priv)
    u = compute_group_rates(sensitive, y_pred, unpriv)
    if p.selection_rate == 0:
        return float("nan")
    return calculate_disparate_impact(p.selection_rate, u.selection_rate)


# --- Strategy implementations ---------------------------------------------
# Each returns a callable ``predict(X, sensitive) -> y_pred`` fitted on TRAIN.

#: A fitted, mitigated predictor: ``predict(X, sensitive_features) -> y_pred``.
PredictFn = Callable[[pd.DataFrame, np.ndarray], np.ndarray]
#: A strategy: ``fit(model, X_train, y_train, sensitive_train, seed) -> PredictFn``.
FitFn = Callable[[object, pd.DataFrame, np.ndarray, np.ndarray, int], PredictFn]


def _fit_reweighing(
    model: object, x_tr: pd.DataFrame, y_tr: np.ndarray, s_tr: np.ndarray, seed: int
) -> Callable[[pd.DataFrame, np.ndarray], np.ndarray]:
    weights = reweighing_sample_weights(y_tr, s_tr)
    refit = clone(model).fit(x_tr, y_tr, sample_weight=weights)
    return lambda x, _s: np.asarray(refit.predict(x))


def _fit_threshold_optimizer(
    model: object, x_tr: pd.DataFrame, y_tr: np.ndarray, s_tr: np.ndarray, seed: int
) -> Callable[[pd.DataFrame, np.ndarray], np.ndarray]:
    from fairlearn.postprocessing import ThresholdOptimizer

    opt = ThresholdOptimizer(
        estimator=model,
        constraints="demographic_parity",
        predict_method="predict_proba",
        prefit=True,
    )
    opt.fit(x_tr, y_tr, sensitive_features=s_tr)
    rng = np.random.RandomState(seed)
    return lambda x, s: np.asarray(opt.predict(x, sensitive_features=s, random_state=rng))


#: ``key -> (fit_fn, stage, human name)``
AVAILABLE_STRATEGIES: dict[str, tuple[FitFn, str, str]] = {
    "threshold_optimizer": (_fit_threshold_optimizer, "post", "Group-specific thresholds"),
    "reweighing": (_fit_reweighing, "pre", "Reweighing (Kamiran-Calders)"),
}


@dataclass
class MitigationResult:
    """Measured outcome of applying one strategy, evaluated on held-out data."""

    strategy: str
    stage: str
    group_label: str
    di_before: float
    di_after: float
    di_after_ci: ConfidenceInterval | None
    accuracy_before: float
    accuracy_after: float
    n_test: int

    @property
    def di_improvement(self) -> float:
        return self.di_after - self.di_before

    @property
    def accuracy_cost(self) -> float:
        return self.accuracy_before - self.accuracy_after

    @property
    def crossed_threshold(self) -> bool:
        return self.di_after >= DI_THRESHOLD


def mitigate_and_reaudit(
    model: object,
    data: pd.DataFrame,
    attribute: str,
    privileged_value: object,
    unprivileged_value: object,
    *,
    target: str,
    feature_columns: list[str],
    strategy: str = "threshold_optimizer",
    test_size: float = 0.3,
    n_boot: int = 1000,
    seed: int = DEFAULT_SEED,
) -> MitigationResult:
    """Fit ``strategy`` on a train split and measure its effect on a held-out split."""
    if strategy not in AVAILABLE_STRATEGIES:
        raise KeyError(f"Unknown strategy '{strategy}'. Available: {list(AVAILABLE_STRATEGIES)}")
    fit_fn, stage, _name = AVAILABLE_STRATEGIES[strategy]

    x = data[feature_columns]
    y = data[target].to_numpy()
    s = data[attribute].to_numpy()
    x_tr, x_te, y_tr, y_te, s_tr, s_te = train_test_split(
        x, y, s, test_size=test_size, random_state=seed, stratify=s
    )

    # Baseline (original model) on the held-out split.
    base_pred = np.asarray(model.predict(x_te))  # type: ignore[attr-defined]
    di_before = _group_di(s_te, base_pred, privileged_value, unprivileged_value)
    acc_before = float(accuracy_score(y_te, base_pred))

    # Mitigated predictor, fitted on TRAIN, evaluated on the same held-out split.
    predict = fit_fn(model, x_tr, y_tr, s_tr, seed)
    mit_pred = np.asarray(predict(x_te, s_te))
    di_after = _group_di(s_te, mit_pred, privileged_value, unprivileged_value)
    acc_after = float(accuracy_score(y_te, mit_pred))

    di_ci = disparate_impact_ci(
        s_te, mit_pred, privileged_value, unprivileged_value, n_boot=n_boot, seed=seed
    )
    return MitigationResult(
        strategy=strategy,
        stage=stage,
        group_label=f"{attribute}:{unprivileged_value}",
        di_before=di_before,
        di_after=di_after,
        di_after_ci=di_ci,
        accuracy_before=acc_before,
        accuracy_after=acc_after,
        n_test=len(y_te),
    )
