"""Accuracy-vs-fairness Pareto frontier.

Sweeps a constrained reduction (Fairlearn ``GridSearch`` with a demographic-parity
constraint) to produce a family of models trading accuracy against fairness, each
evaluated on a **held-out split** with a bootstrap CI on its Disparate Impact.
The Pareto-optimal subset is the honest menu of options to put in front of a
decision-maker — there is rarely a single "best" point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fairness_core.mitigation.strategies import _group_di
from fairness_core.seeds import DEFAULT_SEED
from fairness_core.stats import disparate_impact_ci
from fairness_core.types import ConfidenceInterval

__all__ = ["ParetoPoint", "pareto_frontier"]


@dataclass
class ParetoPoint:
    label: str
    accuracy: float
    di: float
    di_ci: ConfidenceInterval | None
    fairness_violation: float  # |1 - DI|; 0 == perfect parity
    pareto_optimal: bool = False


def _mark_pareto(points: list[ParetoPoint]) -> None:
    """Flag points that are not dominated (higher accuracy *and* lower violation)."""
    for p in points:
        dominated = any(
            q is not p
            and q.accuracy >= p.accuracy
            and q.fairness_violation <= p.fairness_violation
            and (q.accuracy > p.accuracy or q.fairness_violation < p.fairness_violation)
            for q in points
        )
        p.pareto_optimal = not dominated


def pareto_frontier(
    estimator_factory: Callable[[], object],
    data: pd.DataFrame,
    attribute: str,
    privileged_value: object,
    unprivileged_value: object,
    *,
    target: str,
    feature_columns: list[str],
    grid_size: int = 8,
    test_size: float = 0.3,
    n_boot: int = 400,
    seed: int = DEFAULT_SEED,
) -> list[ParetoPoint]:
    """Trace the accuracy/fairness trade-off for one protected group."""
    from fairlearn.reductions import DemographicParity, GridSearch

    x = data[feature_columns]
    y = data[target].to_numpy()
    s = data[attribute].to_numpy()
    x_tr, x_te, y_tr, y_te, s_tr, s_te = train_test_split(
        x, y, s, test_size=test_size, random_state=seed, stratify=s
    )

    def evaluate(label: str, y_pred: np.ndarray) -> ParetoPoint:
        di = _group_di(s_te, y_pred, privileged_value, unprivileged_value)
        ci = disparate_impact_ci(
            s_te, y_pred, privileged_value, unprivileged_value, n_boot=n_boot, seed=seed
        )
        return ParetoPoint(
            label=label,
            accuracy=float(accuracy_score(y_te, y_pred)),
            di=di,
            di_ci=ci,
            fairness_violation=abs(1.0 - di) if np.isfinite(di) else float("inf"),
        )

    points: list[ParetoPoint] = []

    # Unconstrained baseline.
    base = clone(estimator_factory()).fit(x_tr, y_tr)
    points.append(evaluate("baseline", np.asarray(base.predict(x_te))))

    # Constrained family.
    sweep = GridSearch(estimator_factory(), constraints=DemographicParity(), grid_size=grid_size)
    sweep.fit(x_tr, y_tr, sensitive_features=s_tr)
    for i, predictor in enumerate(sweep.predictors_):
        y_pred = np.asarray(predictor.predict(x_te))
        if len(np.unique(y_pred)) < 2:  # skip degenerate all-one-class models
            continue
        points.append(evaluate(f"grid-{i}", y_pred))

    points = [p for p in points if np.isfinite(p.di)]
    _mark_pareto(points)
    points.sort(key=lambda p: p.accuracy, reverse=True)
    return points
