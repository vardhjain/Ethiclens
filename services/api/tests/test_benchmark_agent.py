"""No-network checks for the portfolio evidence runner."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from scripts.benchmark_agent import (
    _interval_label,
    describe_group_evidence,
    render_markdown,
    validate_coverage,
    validate_evidence,
)

from ethiclens_api.agent.demo_datasets import DEMO_DATASETS


def _scorecard(*, di: float = 0.5, measured: object = None) -> dict:
    return {
        "has_labels": False,
        "groups": [
            {
                "attribute": "race",
                "privileged_value": "A",
                "unprivileged_value": "B",
                "flagged": True,
                "metrics": {
                    "disparate_impact": {
                        "value": di,
                        "classification": "FAIL",
                    },
                    "equalized_odds": {
                        "value": None,
                        "classification": "INSUFFICIENT_DATA",
                    },
                },
            }
        ],
        "measured_mitigation": measured,
    }


def test_independent_csv_selection_rate_cross_check() -> None:
    dataset = DEMO_DATASETS["synthetic_hiring"]
    data = pd.DataFrame({"race": ["A"] * 4 + ["B"] * 4, "decision": [1] * 4 + [1, 1, 0, 0]})
    validate_evidence(dataset, data, _scorecard())

    with pytest.raises(ValueError, match="disagrees"):
        validate_evidence(dataset, data, _scorecard(di=0.8))


def test_refuses_equalized_odds_without_true_labels() -> None:
    dataset = DEMO_DATASETS["synthetic_hiring"]
    data = pd.DataFrame({"race": ["A"] * 4 + ["B"] * 4, "decision": [1] * 4 + [1, 1, 0, 0]})
    scorecard = _scorecard()
    scorecard["groups"][0]["metrics"]["equalized_odds"]["value"] = 0.1
    with pytest.raises(ValueError, match="Equalized Odds"):
        validate_evidence(dataset, data, scorecard)


def test_refuses_fabricated_measured_mitigation() -> None:
    dataset = DEMO_DATASETS["synthetic_hiring"]
    data = pd.DataFrame({"race": ["A"] * 4 + ["B"] * 4, "decision": [1] * 4 + [1, 1, 0, 0]})
    with pytest.raises(ValueError, match="measured-mitigation"):
        validate_evidence(dataset, data, _scorecard(measured={"di_after": 0.9}))


def test_markdown_discloses_unavailable_measurements() -> None:
    report = {
        "environment": {"python": "3.12", "pandas": "2.2", "scikit_learn": "1.5"},
        "repeats": 1,
        "cases": [
            {
                "key": "synthetic_hiring",
                "rows": 200,
                "labels_available": False,
                "positive_is_favorable": True,
                "flagged_cohorts": 1,
                "cohort_comparisons": 1,
                "computable_di_cohorts": 0,
                "di_coverage_reason": "1 of 1 group comparisons have insufficient DI evidence.",
                "group_evidence": [
                    {
                        "group_label": "race:B",
                        "reference_group": "A",
                        "reference_rows": 80,
                        "comparison_rows": 20,
                        "di_supported": False,
                        "di_value": None,
                        "di_ci": None,
                        "di_reason": "Requires 100 rows in each group.",
                        "spd_reason": "Requires 100 rows in each group.",
                        "eo_supported": False,
                        "eo_reason": "No true labels.",
                    }
                ],
                "min_disparate_impact": None,
                "composite_reason": "Ground-truth labels absent.",
                "heldout_mitigation": None,
                "timing_ms": {"total_ms": {"median": 42.0}},
            }
        ],
    }
    text = render_markdown(report)
    assert "not assessed" in text
    assert "not available" in text
    assert "not comparable" in text
    assert json.dumps(report)


def _group(*, di: dict, eo: dict | None = None, spd: dict | None = None) -> dict:
    return {
        "group_label": "race:B",
        "privileged_value": "A",
        "unprivileged_value": "B",
        "n_privileged": 300,
        "n_unprivileged": 300,
        "flagged": False,
        "metrics": {
            "disparate_impact": di,
            "spd": spd or {"value": -0.1, "classification": "FLAGGED"},
            "equalized_odds": eo
            or {
                "value": None,
                "classification": "INSUFFICIENT_DATA",
                "details": {"reason": "Ground-truth labels are unavailable."},
            },
        },
    }


def test_coverage_preserves_raw_underpowered_value_but_hides_supported_di() -> None:
    group = _group(
        di={
            "value": 0.45,
            "classification": "INSUFFICIENT_DATA",
            "details": {"reason": "At least 100 people per group required."},
            "ci": {"low": 0.35, "high": 0.55, "level": 0.95},
        }
    )
    evidence = describe_group_evidence([group])
    assert evidence[0]["di_raw_point_estimate"] == 0.45
    assert evidence[0]["di_value"] is None
    assert evidence[0]["di_ci"] is None
    assert "100 people" in evidence[0]["di_reason"]
    with pytest.raises(ValueError, match="underpowered"):
        validate_coverage(
            {"min_disparate_impact": 0.45, "composite_score": None},
            evidence,
        )
    validate_coverage(
        {
            "min_disparate_impact": None,
            "composite_score": None,
            "composite_reason": "Underpowered DI.",
        },
        evidence,
    )


def test_complete_di_with_incomplete_eo_keeps_only_di_headline() -> None:
    interval = {"low": 0.49, "high": 0.67, "level": 0.95, "method": "bootstrap-bca"}
    group = _group(di={"value": 0.56, "classification": "FAIL", "ci": interval})
    evidence = describe_group_evidence([group])
    assert evidence[0]["di_value"] == 0.56
    assert evidence[0]["di_ci"] == interval
    assert _interval_label(interval) == "[0.490, 0.670] (95%, bootstrap-bca)"
    assert evidence[0]["eo_reason"] == "Ground-truth labels are unavailable."
    validate_coverage(
        {
            "min_disparate_impact": 0.56,
            "composite_score": None,
            "composite_reason": "Missing ground truth.",
        },
        evidence,
    )
    with pytest.raises(ValueError, match="does not cover"):
        validate_coverage(
            {"min_disparate_impact": 0.60, "composite_score": None},
            evidence,
        )


def test_metric_coverage_rejects_false_composite_and_missing_explanation() -> None:
    group = _group(di={"value": 0.8, "classification": "PASS"})
    evidence = describe_group_evidence([group])
    with pytest.raises(ValueError, match="availability"):
        validate_coverage(
            {"min_disparate_impact": 0.8, "composite_score": 0.99},
            evidence,
        )
    with pytest.raises(ValueError, match="explanation"):
        validate_coverage(
            {"min_disparate_impact": 0.8, "composite_score": None},
            evidence,
        )


def test_report_includes_cohort_evidence_reason_and_ci() -> None:
    report = {
        "environment": {"python": "3.12", "pandas": "2.2", "scikit_learn": "1.5"},
        "repeats": 1,
        "cases": [
            {
                "key": "synthetic_hiring",
                "rows": 600,
                "labels_available": False,
                "positive_is_favorable": True,
                "flagged_cohorts": 1,
                "cohort_comparisons": 1,
                "computable_di_cohorts": 1,
                "di_coverage_reason": None,
                "min_disparate_impact": 0.56,
                "composite_reason": "Ground truth unavailable.",
                "heldout_mitigation": None,
                "timing_ms": {"total_ms": {"median": 80.0}},
                "group_evidence": describe_group_evidence(
                    [
                        _group(
                            di={
                                "value": 0.56,
                                "classification": "FAIL",
                                "ci": {
                                    "low": 0.49,
                                    "high": 0.67,
                                    "level": 0.95,
                                    "method": "bootstrap-bca",
                                },
                            }
                        )
                    ]
                ),
            }
        ],
    }
    text = render_markdown(report)
    assert "race:B (ref: A)" in text
    assert "[0.490, 0.670] (95%, bootstrap-bca)" in text
    assert "Ground-truth labels are unavailable" in text
    assert "composite unavailable" in text


def test_excluded_cohort_prevents_selective_aggregate_even_with_eligible_groups() -> None:
    from scripts.benchmark_agent import validate_coverage

    one = {
        "group_label": "race:B",
        "privileged_value": "A",
        "unprivileged_value": "B",
        "n_privileged": 240,
        "n_unprivileged": 240,
        "flagged": False,
        "metrics": {
            "disparate_impact": {"value": 0.8, "classification": "PASS"},
            "spd": {"value": -0.1, "classification": "Acceptable"},
            "equalized_odds": {"value": 0.03, "classification": "Acceptable"},
        },
    }
    from scripts.benchmark_agent import describe_group_evidence

    evidence = describe_group_evidence([one])
    scorecard = {
        "excluded_subgroups": [
            {"attribute": "race", "group_value": "C", "n": 12, "reason": "Too small"}
        ],
        "min_disparate_impact": 0.8,
        "composite_score": 0.9,
    }
    with pytest.raises(ValueError, match="minimum DI must be withheld"):
        validate_coverage(scorecard, evidence)
    scorecard["min_disparate_impact"] = None
    with pytest.raises(ValueError, match="composite must be withheld"):
        validate_coverage(scorecard, evidence)
    scorecard["composite_score"] = None
    scorecard["composite_reason"] = "Subgroup C excluded."
    validate_coverage(scorecard, evidence)
