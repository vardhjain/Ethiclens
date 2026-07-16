"""Stage 4 — LLM narrative generation with a deterministic number-grounding validator.

Hard rule from the build plan: the LLM explains numbers it is handed, it never
generates them. Every numeric token in the LLM's output must be traceable back to the
scorecard JSON (the ``fairness_core`` output), with tolerance for rounding/percentage
formatting. If a generated narrative fails validation, we regenerate once; if that also
fails, we degrade to a numeric-only summary rather than ship an ungrounded claim.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ethiclens_api.agent.grounding import extract_source_numbers, is_grounded
from ethiclens_api.agent.llm_client import LLMClient, complete_json
from ethiclens_api.agent.schemas import AuditPlan
from fairness_core import AuditResult, Recommendation, get_recommendations
from fairness_core.mitigation import MitigationResult

_SYSTEM_PROMPT = (
    "You are a compliance-facing fairness-audit narrator. You are given a JSON scorecard "
    "produced by a deterministic fairness-testing engine, plus ranked mitigation "
    "recommendations. Write a plain-English narrative a compliance officer with no ML "
    "background can understand, and summarize the top recommended mitigation in business "
    "terms (the fairness/accuracy tradeoff, in plain language). If 'measured_mitigation' is "
    "present and non-null, prefer it in your summary and say explicitly that it was measured "
    "on held-out data (not projected) — call out its di_before/di_after and accuracy_cost. "
    "If it is null, make clear the recommendations are projected estimates, not measured. "
    "CRITICAL RULE: every number you write (percentages, ratios, scores, group sizes) must "
    "be copied from the JSON you were given, exactly or lightly rounded. Never compute, "
    "estimate, or invent a number that is not present in the input JSON. Never state a "
    "regulatory conclusion (e.g. 'this fails the 80% rule') beyond what the JSON's "
    "'flagged' fields already indicate — you may restate a flag, never derive a new one."
)


class NarrativeOutput(BaseModel):
    narrative: str = Field(description="Plain-English explanation of the audit findings")
    mitigation_summary: str = Field(
        description="Plain-English summary of the top-ranked mitigation and its tradeoff"
    )


@dataclass
class NarrativeResult:
    output: NarrativeOutput
    grounded: bool
    #: True narrative/mitigation text was replaced with a numeric-only fallback because
    #: validation failed twice in a row.
    degraded: bool


def build_scorecard(
    result: AuditResult,
    plan: AuditPlan,
    measured_mitigation: MitigationResult | None = None,
) -> tuple[dict, dict[str, list[Recommendation]]]:
    """Serialize the audit result + plan into the single source of truth for Stage 4."""
    recommendations = get_recommendations(result)
    scorecard = {
        "composite_score": result.composite_score,
        "composite_band": result.composite_band,
        "min_disparate_impact": result.min_di,
        "has_labels": result.has_labels,
        "metrics_reason": plan.metrics_reason,
        "excluded_subgroups": [e.model_dump() for e in plan.excluded_subgroups],
        "groups": [
            {
                "attribute": g.attribute,
                "group_label": g.group_label,
                "privileged_value": g.privileged_value,
                "unprivileged_value": g.unprivileged_value,
                "n_privileged": g.n_privileged,
                "n_unprivileged": g.n_unprivileged,
                "flagged": g.flagged,
                "metrics": {name: dataclasses.asdict(metric) for name, metric in g.metrics.items()},
            }
            for g in result.groups
        ],
        # Projected (not measured) improvements from fairness_core.get_recommendations. The
        # real Pareto frontier requires retraining an estimator on labeled training data,
        # which this predictions-only agent pipeline never has access to.
        "recommendations": {
            group: [dataclasses.asdict(r) for r in recs] for group, recs in recommendations.items()
        },
        # Real, held-out-measured before/after for one strategy (group-specific decision
        # thresholds) — only available when the CSV had a continuous score column and true
        # labels. None means "unavailable"; the frontend falls back to the projected
        # numbers in `recommendations` above.
        "measured_mitigation": _measured_mitigation_dict(measured_mitigation),
    }
    return scorecard, recommendations


def _measured_mitigation_dict(mitigation: MitigationResult | None) -> dict | None:
    if mitigation is None:
        return None
    return {
        "strategy": mitigation.strategy,
        "stage": mitigation.stage,
        "group_label": mitigation.group_label,
        "di_before": mitigation.di_before,
        "di_after": mitigation.di_after,
        "di_after_ci": (
            dataclasses.asdict(mitigation.di_after_ci) if mitigation.di_after_ci else None
        ),
        "accuracy_before": mitigation.accuracy_before,
        "accuracy_after": mitigation.accuracy_after,
        "di_improvement": mitigation.di_improvement,
        "accuracy_cost": mitigation.accuracy_cost,
        "crossed_threshold": mitigation.crossed_threshold,
        "n_test": mitigation.n_test,
    }


def generate_narrative(client: LLMClient, scorecard: dict) -> NarrativeResult:
    """Generate a grounded narrative for ``scorecard`` (from :func:`build_scorecard`)."""
    prompt = f"Scorecard JSON:\n{json.dumps(scorecard, default=str)}"
    source_numbers = extract_source_numbers(json.dumps(scorecard, default=str))

    for _attempt in range(2):
        output = complete_json(client, _SYSTEM_PROMPT, prompt, NarrativeOutput)
        combined_text = f"{output.narrative} {output.mitigation_summary}"
        if is_grounded(combined_text, source_numbers):
            return NarrativeResult(output=output, grounded=True, degraded=False)
        prompt = (
            f"{prompt}\n\nYour previous narrative contained a number not present in the "
            "scorecard JSON above. Rewrite it using ONLY numbers copied from that JSON."
        )

    return unavailable_result(scorecard)


def unavailable_result(scorecard: dict) -> NarrativeResult:
    """A degraded, numeric-only result for when no LLM provider is configured at all."""
    return NarrativeResult(output=_numeric_only_fallback(scorecard), grounded=False, degraded=True)


def _numeric_only_fallback(scorecard: dict) -> NarrativeOutput:
    band = scorecard.get("composite_band", "unknown")
    score = scorecard.get("composite_score")
    min_di = scorecard.get("min_disparate_impact")
    flagged = [g["group_label"] for g in scorecard.get("groups", []) if g.get("flagged")]
    narrative = (
        f"Composite fairness score: {score} ({band}). Minimum disparate impact ratio: {min_di}. "
        f"Flagged groups: {', '.join(flagged) if flagged else 'none'}. "
        "Narrative generation was unavailable or failed a grounding check; showing numeric "
        "results only."
    )
    return NarrativeOutput(
        narrative=narrative,
        mitigation_summary="Mitigation narrative unavailable; see the recommendations JSON.",
    )
