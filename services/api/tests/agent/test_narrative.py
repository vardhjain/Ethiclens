from __future__ import annotations

import json

from ethiclens_api.agent.narrative import build_scorecard, generate_narrative
from ethiclens_api.agent.schemas import AuditPlan
from fairness_core.mitigation import MitigationResult
from fairness_core.types import AuditResult, GroupAuditResult, MetricResult

_SCORECARD = {
    "composite_score": 0.62,
    "composite_band": "Medium Risk",
    "min_disparate_impact": 0.55,
    "has_labels": False,
    "groups": [{"group_label": "race:Black", "flagged": True}],
}


class _StubClient:
    """Returns a fixed sequence of raw responses, one per call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, system: str, prompt: str) -> str:
        return self._responses.pop(0)


def _response(narrative: str, mitigation: str) -> str:
    return json.dumps({"narrative": narrative, "mitigation_summary": mitigation})


def test_grounded_narrative_passes_through_unchanged():
    client = _StubClient(
        [_response("The min disparate impact is 0.55, indicating a disparity.", "See recs.")]
    )
    result = generate_narrative(client, _SCORECARD)
    assert result.grounded is True
    assert result.degraded is False
    assert "0.55" in result.output.narrative


def test_grounded_narrative_tolerates_percentage_formatting():
    client = _StubClient(
        [_response("The min disparate impact is 55%, well below the threshold.", "See recs.")]
    )
    result = generate_narrative(client, _SCORECARD)
    assert result.grounded is True


def test_fabricated_number_triggers_regeneration_then_degrades():
    # Both attempts fabricate a number (42) that isn't anywhere in the scorecard.
    client = _StubClient(
        [
            _response("42% of applicants were affected.", "ok"),
            _response("42% of applicants were affected.", "ok"),
        ]
    )
    result = generate_narrative(client, _SCORECARD)
    assert result.grounded is False
    assert result.degraded is True
    assert "0.62" in result.output.narrative  # falls back to numeric-only summary


def test_regeneration_succeeds_on_second_attempt():
    client = _StubClient(
        [
            _response("42% of applicants were affected.", "ok"),
            _response("The composite score is 0.62 (Medium Risk).", "ok"),
        ]
    )
    result = generate_narrative(client, _SCORECARD)
    assert result.grounded is True
    assert result.degraded is False


def _minimal_plan() -> AuditPlan:
    return AuditPlan(
        include_equalized_odds=False,
        metrics_reason="no labels",
        protected_attribute_columns=["race"],
        outcome_column="flag",
        true_label_column=None,
        feature_columns=["score"],
        excluded_subgroups=[],
    )


def _minimal_result() -> AuditResult:
    group = GroupAuditResult(
        attribute="race",
        group_label="race:B",
        privileged_value="A",
        unprivileged_value="B",
        n_privileged=60,
        n_unprivileged=60,
        metrics={"disparate_impact": MetricResult(name="disparate_impact", value=0.5)},
        flagged=True,
    )
    return AuditResult(
        composite_score=0.6, composite_band="Medium Risk", min_di=0.5, groups=[group]
    )


def test_build_scorecard_measured_mitigation_is_none_when_not_provided():
    scorecard, _ = build_scorecard(_minimal_result(), _minimal_plan())
    assert scorecard["measured_mitigation"] is None


def test_build_scorecard_serializes_measured_mitigation():
    mitigation = MitigationResult(
        strategy="threshold_optimizer",
        stage="post",
        group_label="race:B",
        di_before=0.5,
        di_after=0.85,
        di_after_ci=None,
        accuracy_before=0.9,
        accuracy_after=0.87,
        n_test=40,
    )
    scorecard, _ = build_scorecard(_minimal_result(), _minimal_plan(), mitigation)
    measured = scorecard["measured_mitigation"]
    assert measured["strategy"] == "threshold_optimizer"
    assert measured["di_before"] == 0.5
    assert measured["di_after"] == 0.85
    assert measured["di_improvement"] == 0.35
    assert measured["accuracy_cost"] == mitigation.accuracy_cost
    assert measured["crossed_threshold"] is True
