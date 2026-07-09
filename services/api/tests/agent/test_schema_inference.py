from __future__ import annotations

import json

import pandas as pd

from ethiclens_api.agent.schema_inference import infer_schema


class _StubClient:
    def __init__(self, response: dict) -> None:
        self._response = response

    def complete(self, system: str, prompt: str) -> str:
        return json.dumps(self._response)


def _compas_like_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race": ["Black", "White", "Black", "White"],
            "score": [1, 0, 1, 0],
            "actual_recid": [1, 0, 0, 0],
            "priors_count": [3, 1, 2, 0],
        }
    )


def test_infer_schema_returns_confirmed_proposal_shape():
    stub_response = {
        "protected_attribute_columns": ["race"],
        "outcome_column": "score",
        "outcome_direction": {
            "positive_label_meaning": "flagged as high risk of recidivism",
            "positive_is_favorable": False,
        },
        "true_label_column": "actual_recid",
        "feature_columns": ["priors_count"],
        "reasoning": "COMPAS-style risk score; a positive flag is adverse, not favorable.",
    }
    client = _StubClient(stub_response)

    proposal = infer_schema(client, _compas_like_frame())

    assert proposal.protected_attribute_columns == ["race"]
    assert proposal.outcome_column == "score"
    assert proposal.outcome_direction.positive_is_favorable is False
    assert proposal.true_label_column == "actual_recid"
    assert proposal.feature_columns == ["priors_count"]


def test_infer_schema_handles_markdown_fenced_response():
    stub_response = {
        "protected_attribute_columns": ["race"],
        "outcome_column": "score",
        "outcome_direction": {
            "positive_label_meaning": "flagged as high risk",
            "positive_is_favorable": False,
        },
        "true_label_column": None,
        "feature_columns": ["priors_count"],
        "reasoning": "test",
    }

    class _FencedClient(_StubClient):
        def complete(self, system: str, prompt: str) -> str:
            return f"```json\n{json.dumps(self._response)}\n```"

    proposal = infer_schema(_FencedClient(stub_response), _compas_like_frame())
    assert proposal.true_label_column is None
