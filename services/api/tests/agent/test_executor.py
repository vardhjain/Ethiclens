from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ethiclens_api.agent.executor import _PassthroughModel, execute_plan, measure_top_mitigation
from ethiclens_api.agent.schemas import AuditPlan, OutcomeDirection


def _dataset() -> pd.DataFrame:
    # Race A (60 rows): 10% flagged high-risk (flag=1). Race B (60 rows): 50% flagged.
    a_flags = [1] * 6 + [0] * 54
    b_flags = [1] * 30 + [0] * 30
    return pd.DataFrame(
        {
            "race": ["A"] * 60 + ["B"] * 60,
            "flag": a_flags + b_flags,
            "score": list(range(120)),
        }
    )


def test_passthrough_model_flips_adverse_outcome():
    predictions = pd.Series([1, 0, 1, 0])
    x = pd.DataFrame(index=predictions.index)
    adverse = OutcomeDirection(
        positive_label_meaning="flagged high-risk", positive_is_favorable=False
    )
    model = _PassthroughModel(predictions, adverse)
    assert list(model.predict(x)) == [0, 1, 0, 1]


def test_passthrough_model_keeps_favorable_outcome_as_is():
    predictions = pd.Series([1, 0, 1, 0])
    x = pd.DataFrame(index=predictions.index)
    favorable = OutcomeDirection(positive_label_meaning="approved", positive_is_favorable=True)
    model = _PassthroughModel(predictions, favorable)
    assert list(model.predict(x)) == [1, 0, 1, 0]


def test_passthrough_model_has_no_predict_proba_without_scores():
    predictions = pd.Series([1, 0])
    favorable = OutcomeDirection(positive_label_meaning="approved", positive_is_favorable=True)
    model = _PassthroughModel(predictions, favorable)
    assert not hasattr(model, "predict_proba")


def test_passthrough_model_predict_proba_reorients_for_adverse():
    predictions = pd.Series([1, 0])
    scores = pd.Series([0.9, 0.1])  # score = P(raw positive) = P(adverse) here
    x = pd.DataFrame(index=predictions.index)
    adverse = OutcomeDirection(
        positive_label_meaning="flagged high-risk", positive_is_favorable=False
    )
    model = _PassthroughModel(predictions, adverse, scores=scores)
    proba = model.predict_proba(x)
    # Favorable = 1 - score: row 0 -> 0.1 favorable, row 1 -> 0.9 favorable.
    assert proba[0, 1] == pytest.approx(0.1)
    assert proba[1, 1] == pytest.approx(0.9)


def test_execute_plan_flags_the_adverse_outcome_group_correctly():
    # flag=1 means "flagged high-risk" (adverse), so group B (higher flag rate) should end
    # up with a LOWER favorable-outcome rate than group A once direction is reoriented.
    plan = AuditPlan(
        include_equalized_odds=False,
        metrics_reason="no labels",
        protected_attribute_columns=["race"],
        outcome_column="flag",
        true_label_column=None,
        feature_columns=["score"],
        excluded_subgroups=[],
    )
    direction = OutcomeDirection(
        positive_label_meaning="flagged high-risk", positive_is_favorable=False
    )

    result = execute_plan(plan, _dataset(), direction)

    assert len(result.groups) == 1
    group = result.groups[0]
    # Group B (50% flagged -> 50% favorable) vs group A (10% flagged -> 90% favorable):
    # disparate impact ratio should be well below the 0.80 threshold.
    di = group.metric("disparate_impact")
    assert di is not None
    assert di.value is not None
    assert di.value < 0.80


def _labeled_dataset() -> pd.DataFrame:
    """200 rows/group with a continuous score, a thresholded decision, and noisy labels."""
    rng = np.random.RandomState(42)
    n_per_group = 100
    race = ["A"] * n_per_group + ["B"] * n_per_group
    # Group B's scores are shifted down -> lower selection rate -> flagged disparate impact.
    score_a = rng.uniform(0.3, 0.9, n_per_group)
    score_b = rng.uniform(0.1, 0.6, n_per_group)
    score = np.concatenate([score_a, score_b])
    flag = (score >= 0.5).astype(int)
    label = (score + rng.normal(0, 0.15, n_per_group * 2) >= 0.5).astype(int)
    feature = rng.uniform(0, 1, n_per_group * 2)
    return pd.DataFrame(
        {"race": race, "flag": flag, "risk_score": score, "actual": label, "feature": feature}
    )


def _labeled_plan(*, score_column: str | None = "risk_score") -> AuditPlan:
    return AuditPlan(
        include_equalized_odds=True,
        metrics_reason="labels present",
        protected_attribute_columns=["race"],
        outcome_column="flag",
        true_label_column="actual",
        score_column=score_column,
        feature_columns=["feature"],
        excluded_subgroups=[],
    )


def test_measure_top_mitigation_returns_measured_result_when_preconditions_met():
    plan = _labeled_plan()
    direction = OutcomeDirection(positive_label_meaning="approved", positive_is_favorable=True)
    data = _labeled_dataset()
    result = execute_plan(plan, data, direction)

    mitigation = measure_top_mitigation(plan, data, direction, result)

    assert mitigation is not None
    assert mitigation.strategy == "threshold_optimizer"
    assert 0.0 <= mitigation.accuracy_before <= 1.0
    assert 0.0 <= mitigation.accuracy_after <= 1.0
    assert mitigation.n_test > 0


def test_measure_top_mitigation_none_without_score_column():
    plan = _labeled_plan(score_column=None)
    direction = OutcomeDirection(positive_label_meaning="approved", positive_is_favorable=True)
    data = _labeled_dataset()
    result = execute_plan(plan, data, direction)
    assert measure_top_mitigation(plan, data, direction, result) is None


def test_measure_top_mitigation_none_without_true_labels():
    plan = _labeled_plan()
    plan = plan.model_copy(update={"true_label_column": None})
    direction = OutcomeDirection(positive_label_meaning="approved", positive_is_favorable=True)
    data = _labeled_dataset()
    result = execute_plan(plan, data, direction)
    assert measure_top_mitigation(plan, data, direction, result) is None


def test_measure_top_mitigation_none_without_flagged_groups():
    plan = _labeled_plan()
    direction = OutcomeDirection(positive_label_meaning="approved", positive_is_favorable=True)
    # Identical scores across groups -> nothing should be flagged.
    data = _labeled_dataset()
    data["race"] = "A"
    result = execute_plan(plan, data, direction)
    assert measure_top_mitigation(plan, data, direction, result) is None
