"""Stage 1 — LLM schema inference from an uploaded CSV, with mandatory human confirmation.

The LLM only *proposes* column roles and outcome direction (the COMPAS-vs-Adult
distinction). Callers must route the result through a confirmation step before any
audit runs — this module never executes an audit itself.
"""

from __future__ import annotations

import pandas as pd

from ethiclens_api.agent.llm_client import LLMClient, complete_json
from ethiclens_api.agent.schemas import SchemaInferenceProposal

_SYSTEM_PROMPT = (
    "You are a data-schema analyst for a fairness-audit tool. Given column names and sample "
    "rows from a predictions CSV, identify: which columns are protected attributes (e.g. race, "
    "sex, age, disability status), which column holds the model's prediction/decision, whether "
    "a ground-truth label column exists, whether a separate continuous prediction-score column "
    "exists (e.g. a 0.0-1.0 risk score or probability the binary decision was thresholded from — "
    "only report one if it is genuinely continuous and distinct from the decision column, else "
    "leave it null), and which columns are ordinary input features. "
    "Critically, determine outcome direction: does a positive value (1/True) mean something "
    "favorable to the subject (e.g. loan approved) or adverse (e.g. flagged as high-risk)? "
    "Getting this backwards silently inverts every fairness finding, so reason about it "
    "explicitly from the column name and sample values. You are proposing an inference for a "
    "human to confirm, not making a final decision — if uncertain, say so in your reasoning."
)

_SAMPLE_ROWS = 5


def infer_schema(client: LLMClient, data: pd.DataFrame) -> SchemaInferenceProposal:
    """Propose column roles for ``data``. The caller must get human confirmation before use."""
    sample = data.head(_SAMPLE_ROWS)
    prompt = (
        f"Columns: {list(data.columns)}\n\n"
        f"Sample rows (first {_SAMPLE_ROWS}):\n{sample.to_csv(index=False)}\n\n"
        f"Total rows: {len(data)}"
    )
    return complete_json(client, _SYSTEM_PROMPT, prompt, SchemaInferenceProposal)
