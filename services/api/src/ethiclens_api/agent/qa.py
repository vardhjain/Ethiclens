"""Stage 5 — grounded Q&A over a stored audit record.

Same hard rule as Stage 4: the LLM may only restate numbers already present in the
stored scorecard JSON. It retrieves from that JSON — it never recomputes a metric or
answers from its own training-time knowledge. Uses the same grounding validator as
Stage 4 (``agent.grounding``), regenerating once on failure and degrading to a
"cannot answer without fabricating a number" response if that also fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ethiclens_api.agent.grounding import extract_source_numbers, is_grounded
from ethiclens_api.agent.llm_client import LLMClient, complete_json

_SYSTEM_PROMPT = (
    "You are a fairness-audit assistant answering questions about a completed audit. You "
    "are given the audit's scorecard JSON as your only source of truth. Answer the "
    "question using only facts present in that JSON. "
    "CRITICAL RULE: every number you write must be copied from the JSON, exactly or "
    "lightly rounded. Never compute, estimate, or invent a number that is not present in "
    "the JSON. If the JSON does not contain enough information to answer, say so plainly "
    "instead of guessing."
)


class QAAnswer(BaseModel):
    answer: str = Field(description="Plain-English answer, grounded only in the scorecard JSON")


@dataclass
class QAResult:
    answer: str
    grounded: bool
    degraded: bool


_UNAVAILABLE_ANSWER = (
    "I can't answer that without risking a fabricated number — the audit record doesn't "
    "contain enough grounded information to answer confidently. Try asking about a "
    "specific group or metric shown in the report."
)


def ask(client: LLMClient, scorecard: dict, question: str) -> QAResult:
    """Answer ``question`` about ``scorecard``, validating every number in the response."""
    scorecard_text = json.dumps(scorecard, default=str)
    source_numbers = extract_source_numbers(scorecard_text)
    prompt = f"Scorecard JSON:\n{scorecard_text}\n\nQuestion: {question}"

    for _attempt in range(2):
        output = complete_json(client, _SYSTEM_PROMPT, prompt, QAAnswer)
        if is_grounded(output.answer, source_numbers):
            return QAResult(answer=output.answer, grounded=True, degraded=False)
        prompt = (
            f"{prompt}\n\nYour previous answer contained a number not present in the "
            "scorecard JSON above. Rewrite it using ONLY numbers copied from that JSON."
        )

    return QAResult(answer=_UNAVAILABLE_ANSWER, grounded=False, degraded=True)
