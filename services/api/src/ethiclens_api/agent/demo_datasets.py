"""Canned predictions CSVs for the agent's one-click demo path.

Per the build plan: "95% of visitors (recruiters) should never need to
upload anything." Each entry pairs a CSV shipped in the package (generated
by ``ml/cli/make_agent_demo_csvs.py``, never regenerated at runtime) with a
hand-verified, human-checked-once ``SchemaInferenceProposal`` — so the demo
path burns zero LLM calls on Stage 1 and can never propose the wrong outcome
direction. Stage 4 (narrative) still calls the real LLM.

The three datasets are deliberately chosen to exercise three different
guardrail paths — see each ``blurb`` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ethiclens_api.agent.schemas import OutcomeDirection, SchemaInferenceProposal

_DATA_DIR = Path(__file__).resolve().parent / "demo_data"


@dataclass(frozen=True)
class DemoDataset:
    key: str
    title: str
    blurb: str
    csv_path: Path
    proposal: SchemaInferenceProposal


DEMO_DATASETS: dict[str, DemoDataset] = {
    "compas": DemoDataset(
        key="compas",
        title="COMPAS recidivism risk (ProPublica)",
        blurb=(
            "The canonical algorithmic-bias case study. Has both true labels and a "
            "continuous risk score, so this run gets the full metric suite plus a "
            "real, held-out-measured before/after mitigation — not a projection."
        ),
        csv_path=_DATA_DIR / "compas.csv",
        proposal=SchemaInferenceProposal(
            protected_attribute_columns=["race"],
            outcome_column="decision",
            outcome_direction=OutcomeDirection(
                positive_label_meaning="flagged as high risk of recidivism",
                positive_is_favorable=False,
            ),
            true_label_column="two_year_recid",
            score_column="risk_score",
            feature_columns=["priors_count"],
            reasoning=(
                "COMPAS decile_score >= 5 is flagged high-risk, an adverse outcome for the "
                "subject — the classic direction-flip case: a naive audit that assumed "
                "positive=favorable would silently invert every finding."
            ),
        ),
    ),
    "adult_income": DemoDataset(
        key="adult_income",
        title="Adult Census Income (>$50k prediction)",
        blurb=(
            "A logistic-regression model predicting high income. Has true labels but no "
            "continuous score, so mitigation falls back to clearly-labeled projected "
            "estimates instead of a measured chart — deliberately contrasts with COMPAS."
        ),
        csv_path=_DATA_DIR / "adult_income.csv",
        proposal=SchemaInferenceProposal(
            protected_attribute_columns=["race"],
            outcome_column="decision",
            outcome_direction=OutcomeDirection(
                positive_label_meaning="predicted income above $50k",
                positive_is_favorable=True,
            ),
            true_label_column="actual_income_above_50k",
            score_column=None,
            feature_columns=[
                "age",
                "education-num",
                "hours-per-week",
                "capital-gain",
                "capital-loss",
            ],
            reasoning=(
                "A positive prediction (income > $50k) is favorable to the subject. True "
                "labels are present but the model only exposes a hard decision, no score."
            ),
        ),
    ),
    "synthetic_hiring": DemoDataset(
        key="synthetic_hiring",
        title="Synthetic hiring screen (no ground truth)",
        blurb=(
            "A synthetic model with no true-label column at all — the most common real-world "
            "case for a predictions-only audit. Shows the report explicitly stating that "
            "Equalized Odds was skipped, and why."
        ),
        csv_path=_DATA_DIR / "synthetic_hiring.csv",
        proposal=SchemaInferenceProposal(
            protected_attribute_columns=["race"],
            outcome_column="decision",
            outcome_direction=OutcomeDirection(
                positive_label_meaning="loan/application approved",
                positive_is_favorable=True,
            ),
            true_label_column=None,
            score_column=None,
            feature_columns=["income", "credit_score", "debt_ratio"],
            reasoning=(
                "A positive decision is favorable to the subject. No ground-truth outcome "
                "column exists in this dataset, so Equalized Odds cannot be computed."
            ),
        ),
    ),
}


def list_demo_datasets() -> list[DemoDataset]:
    return list(DEMO_DATASETS.values())


def get_demo_dataset(key: str) -> DemoDataset | None:
    return DEMO_DATASETS.get(key)
