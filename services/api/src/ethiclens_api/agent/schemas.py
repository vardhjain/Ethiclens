"""Pydantic models shared across the agent stages.

Kept separate from ``ethiclens_api.schemas`` (the HTTP contract) because these are
purely internal to the agent pipeline and are also used as LLM structured-output
schemas (``llm_client.complete_json`` validates against them directly).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class OutcomeDirection(BaseModel):
    """Whether the outcome column's positive value (1/True) is a good or bad thing.

    This is the COMPAS-vs-Adult distinction: for Adult Income, 1 = "approved" (favorable).
    For COMPAS, 1 = "high risk of recidivism" (adverse). Getting this backwards silently
    inverts every disparate-impact and SPD finding, so it is never guessed silently — it
    is always shown to the user for confirmation.
    """

    positive_label_meaning: str = Field(
        description=(
            "Plain-English meaning of outcome==1, e.g. 'loan approved' or 'flagged high-risk'"
        )
    )
    positive_is_favorable: bool = Field(
        description="True if outcome==1 is the favorable/desirable outcome for the subject"
    )


class SchemaInferenceProposal(BaseModel):
    """Stage 1 output: the LLM's proposed reading of an uploaded CSV's columns.

    This is a *proposal only* — callers must route it through a human confirmation
    step before any audit runs on it. Never execute an audit against an unconfirmed
    proposal.
    """

    protected_attribute_columns: list[str] = Field(
        description="Column names that look like protected attributes (race, sex, age, etc.)"
    )
    outcome_column: str = Field(description="Column name holding the model's prediction/decision")
    outcome_direction: OutcomeDirection
    true_label_column: str | None = Field(
        default=None, description="Column name holding ground-truth labels, if present, else null"
    )
    score_column: str | None = Field(
        default=None,
        description=(
            "Column name holding a continuous prediction score/probability (e.g. 0.0-1.0 risk "
            "score), distinct from the binary outcome_column, if present, else null. Only set "
            "this if the value genuinely varies continuously — do not reuse outcome_column here."
        ),
    )
    feature_columns: list[str] = Field(
        description="Column names to use as model input features (excludes protected/outcome/label)"
    )
    reasoning: str = Field(description="One or two sentences explaining the inference")


class ExcludedSubgroup(BaseModel):
    attribute: str
    group_value: str
    n: int
    reason: str


class AskRequest(BaseModel):
    """Stage 5 request body: a free-text question about a stored audit record."""

    question: str = Field(min_length=1, max_length=2000)


class AuditPlan(BaseModel):
    """Stage 2 output: which metrics run and why. Produced by rules, never by the LLM."""

    include_equalized_odds: bool
    metrics_reason: str
    protected_attribute_columns: list[str]
    outcome_column: str
    true_label_column: str | None
    score_column: str | None = None
    feature_columns: list[str]
    excluded_subgroups: list[ExcludedSubgroup] = Field(default_factory=list)
