"""No-network checks for the portfolio evidence runner."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from scripts.benchmark_agent import render_markdown, validate_evidence

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
                "min_disparate_impact": None,
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
