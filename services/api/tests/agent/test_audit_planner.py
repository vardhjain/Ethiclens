from __future__ import annotations

import pandas as pd

from ethiclens_api.agent.audit_planner import MIN_SUBGROUP_SIZE, plan_audit
from ethiclens_api.agent.schemas import OutcomeDirection, SchemaInferenceProposal


def _proposal(**overrides) -> SchemaInferenceProposal:
    defaults = {
        "protected_attribute_columns": ["race"],
        "outcome_column": "approved",
        "outcome_direction": OutcomeDirection(
            positive_label_meaning="loan approved", positive_is_favorable=True
        ),
        "true_label_column": None,
        "feature_columns": ["income", "credit_score"],
        "reasoning": "test",
    }
    defaults.update(overrides)
    return SchemaInferenceProposal(**defaults)


def test_labels_present_includes_equalized_odds():
    proposal = _proposal(true_label_column="actual_default")
    data = pd.DataFrame({"race": ["A"] * 50 + ["B"] * 50})
    plan = plan_audit(proposal, data)
    assert plan.include_equalized_odds is True
    assert "Equalized Odds" in plan.metrics_reason


def test_labels_absent_excludes_equalized_odds_and_states_limitation():
    proposal = _proposal(true_label_column=None)
    data = pd.DataFrame({"race": ["A"] * 50 + ["B"] * 50})
    plan = plan_audit(proposal, data)
    assert plan.include_equalized_odds is False
    assert "skipped" in plan.metrics_reason


def test_small_subgroup_is_excluded_with_reason():
    proposal = _proposal()
    n_small = MIN_SUBGROUP_SIZE - 1
    data = pd.DataFrame({"race": ["A"] * 100 + ["B"] * n_small})
    plan = plan_audit(proposal, data)
    assert len(plan.excluded_subgroups) == 1
    excluded = plan.excluded_subgroups[0]
    assert excluded.attribute == "race"
    assert excluded.group_value == "B"
    assert excluded.n == n_small
    assert "below the minimum floor" in excluded.reason


def test_large_subgroups_are_not_excluded():
    proposal = _proposal()
    data = pd.DataFrame({"race": ["A"] * 50 + ["B"] * 50})
    plan = plan_audit(proposal, data)
    assert plan.excluded_subgroups == []


def test_continuous_score_column_is_kept():
    proposal = _proposal(score_column="risk_score")
    data = pd.DataFrame(
        {"race": ["A"] * 50 + ["B"] * 50, "risk_score": [i / 100 for i in range(100)]}
    )
    plan = plan_audit(proposal, data)
    assert plan.score_column == "risk_score"


def test_binary_score_column_is_dropped():
    proposal = _proposal(score_column="flag")
    data = pd.DataFrame({"race": ["A"] * 50 + ["B"] * 50, "flag": ([0, 1] * 50)})
    plan = plan_audit(proposal, data)
    assert plan.score_column is None


def test_missing_score_column_is_dropped():
    proposal = _proposal(score_column="does_not_exist")
    data = pd.DataFrame({"race": ["A"] * 50 + ["B"] * 50})
    plan = plan_audit(proposal, data)
    assert plan.score_column is None
