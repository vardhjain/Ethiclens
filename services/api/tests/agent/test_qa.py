from __future__ import annotations

import json

from ethiclens_api.agent.qa import ask

_SCORECARD = {
    "composite_score": 0.62,
    "min_disparate_impact": 0.55,
    "groups": [
        {"group_label": "race:Black", "flagged": True},
        {"group_label": "race:Hispanic", "flagged": False},
    ],
}


class _StubClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, system: str, prompt: str) -> str:
        return self._responses.pop(0)


def _response(answer: str) -> str:
    return json.dumps({"answer": answer})


def test_grounded_answer_passes_through():
    client = _StubClient([_response("race:Black was flagged; the min DI was 0.55.")])
    result = ask(client, _SCORECARD, "Why was race:Black flagged?")
    assert result.grounded is True
    assert result.degraded is False
    assert "0.55" in result.answer


def test_fabricated_number_degrades_after_two_attempts():
    client = _StubClient([_response("It's 42%."), _response("It's still 42%.")])
    result = ask(client, _SCORECARD, "What fraction was affected?")
    assert result.grounded is False
    assert result.degraded is True
    assert "can't answer" in result.answer.lower()


def test_regeneration_recovers_on_second_attempt():
    client = _StubClient(
        [_response("42% were affected."), _response("The composite score is 0.62.")]
    )
    result = ask(client, _SCORECARD, "What was the composite score?")
    assert result.grounded is True
    assert result.degraded is False
