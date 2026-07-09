"""Stage 3 — execute a confirmed :class:`AuditPlan` via the existing fairness-core engine.

No LLM involved, and no changes to ``fairness_core`` itself: this module only adapts an
agent-uploaded predictions CSV (which already contains the model's decisions — there is
no model object to call) into the ``run_audit`` call signature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from ethiclens_api.agent.schemas import AuditPlan, OutcomeDirection
from fairness_core import AttributeSpec, AuditResult, run_audit
from fairness_core.mitigation import MitigationResult, mitigate_and_reaudit

#: fairness_core's disparate-impact/SPD semantics assume 1 == the favorable outcome.
#: A pass-through "model" that just returns an already-computed prediction column,
#: reorienting it to that convention when the outcome is adverse (e.g. COMPAS risk flags).


class _PassthroughModel(BaseEstimator, ClassifierMixin):
    """Exposes the CSV's own prediction (and optional score) column as a live model's output.

    Indexes into ``x``'s own index rather than assuming full-length input, so this works
    both when called on the full dataframe (Stage 3's ``run_audit``) and on a train/test
    split subset (measured mitigation, see :func:`measure_top_mitigation`).

    Subclasses sklearn's ``BaseEstimator``/``ClassifierMixin`` purely so Fairlearn's
    ``ThresholdOptimizer`` (even with ``prefit=True``) accepts it as a "real" fitted
    estimator — sklearn's ``check_is_fitted`` now requires ``__sklearn_tags__`` support.
    """

    def __init__(
        self,
        predictions: pd.Series | None = None,
        direction: OutcomeDirection | None = None,
        scores: pd.Series | None = None,
    ) -> None:
        self.predictions = predictions
        self.direction = direction
        self.scores = scores
        self._flip = direction is not None and not direction.positive_is_favorable
        # There is nothing to actually fit: predictions/scores are already fixed. This
        # attribute only exists so sklearn's check_is_fitted() sees a "fitted" estimator.
        self.fitted_ = True
        # fairness_core.predict_labels() prefers predict_proba when present (hasattr check),
        # so only expose it on instances that actually have a score column — otherwise it
        # would shadow predict() with a method that always raises.
        if scores is not None:
            self.predict_proba = self._predict_proba

    def fit(self, *_args: object, **_kwargs: object) -> _PassthroughModel:
        """No-op: predictions/scores are already fixed."""
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        assert self.predictions is not None
        values = self.predictions.loc[x.index].astype(int)
        return (1 - values).to_numpy() if self._flip else values.to_numpy()

    def _predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """[P(adverse), P(favorable)] per row, for Fairlearn's ThresholdOptimizer."""
        assert self.scores is not None
        favorable = self.scores.loc[x.index].astype(float).to_numpy()
        if self._flip:
            favorable = 1.0 - favorable
        favorable = np.clip(favorable, 0.0, 1.0)
        return np.column_stack([1.0 - favorable, favorable])


def execute_plan(plan: AuditPlan, data: pd.DataFrame, direction: OutcomeDirection) -> AuditResult:
    """Run the confirmed plan through ``fairness_core.run_audit``.

    ``direction`` is passed separately (rather than re-derived) because reorienting the
    outcome polarity is a semantic decision the human already confirmed in Stage 1 — this
    stage only applies it mechanically.
    """
    working = data.copy()
    model = _PassthroughModel(working[plan.outcome_column], direction)

    target_column = None
    if plan.true_label_column is not None:
        target_column = plan.true_label_column
        if not direction.positive_is_favorable:
            working[target_column] = 1 - working[target_column].astype(int)

    specs = [
        AttributeSpec(
            name=attr,
            unprivileged_values=_surviving_values(plan, attr, working),
        )
        for attr in plan.protected_attribute_columns
    ]

    return run_audit(
        model,
        working,
        specs,
        target=target_column,
        feature_columns=plan.feature_columns,
        compute_ci=True,
    )


def measure_top_mitigation(
    plan: AuditPlan, data: pd.DataFrame, direction: OutcomeDirection, result: AuditResult
) -> MitigationResult | None:
    """A real, held-out-measured before/after for the most-flagged group.

    Only "group-specific decision thresholds" (Fairlearn's ThresholdOptimizer) is usable
    here: it's the one strategy that needs no retrainable model, only an existing
    continuous score. Reweighing/retraining strategies need a real classifier to fit,
    which a predictions-only CSV never provides.

    Returns ``None`` (not an error) whenever a precondition isn't met — a continuous
    score column and true labels are both required, plus at least one flagged group —
    or if Fairlearn's optimizer fails on the data (e.g. a degenerate split). Callers
    should treat ``None`` as "measured mitigation unavailable," falling back to the
    projected-only recommendations already computed elsewhere.
    """
    if plan.score_column is None or plan.true_label_column is None:
        return None
    flagged = [g for g in result.flagged_groups if g.metric("disparate_impact") is not None]
    if not flagged:
        return None
    group = min(flagged, key=lambda g: g.metric("disparate_impact").value or 1.0)

    working = data.copy()
    target_column = plan.true_label_column
    if not direction.positive_is_favorable:
        working[target_column] = 1 - working[target_column].astype(int)

    values = working[group.attribute].dropna().unique()
    privileged_value = next((v for v in values if str(v) == group.privileged_value), None)
    unprivileged_value = next((v for v in values if str(v) == group.unprivileged_value), None)
    if privileged_value is None or unprivileged_value is None:
        return None

    # Fairlearn's ThresholdOptimizer requires every sensitive-feature value present in
    # the data to have both label classes ("non-degenerate"). A real-world protected
    # attribute often has more than two categories (e.g. COMPAS's six race values); pass
    # only the privileged/unprivileged pair being compared, not the whole column, or a
    # tiny unrelated category (e.g. n=3) can fail the whole optimizer.
    working = working[working[group.attribute].isin([privileged_value, unprivileged_value])]

    model = _PassthroughModel(
        working[plan.outcome_column], direction, scores=working[plan.score_column]
    )
    try:
        return mitigate_and_reaudit(
            model,
            working,
            group.attribute,
            privileged_value,
            unprivileged_value,
            target=target_column,
            feature_columns=plan.feature_columns,
            strategy="threshold_optimizer",
        )
    except Exception:
        return None


def _surviving_values(plan: AuditPlan, attribute: str, data: pd.DataFrame) -> list[object] | None:
    """Values of ``attribute`` not excluded by Stage 2's min-subgroup-size rule.

    Returns ``None`` (meaning "all other values") when nothing was excluded, so
    small datasets with no exclusions aren't needlessly constrained.
    """
    excluded_values = {e.group_value for e in plan.excluded_subgroups if e.attribute == attribute}
    if not excluded_values:
        return None
    return [v for v in data[attribute].dropna().unique() if str(v) not in excluded_values]
