"""Stage 2 — deterministic audit planning rules engine. No LLM involved.

Takes a *confirmed* :class:`SchemaInferenceProposal` plus the actual dataframe and
decides which metrics apply and which subgroups are too small to trust. These are
regulatory/statistical rules, not model judgment — the LLM in Stage 4 may only
restate what this stage already decided, never re-derive it.
"""

from __future__ import annotations

import pandas as pd

from ethiclens_api.agent.schemas import AuditPlan, ExcludedSubgroup, SchemaInferenceProposal

#: Subgroups smaller than this are excluded from per-group metrics: bootstrap
#: confidence intervals on tiny groups are noise, not signal.
MIN_SUBGROUP_SIZE = 30


def plan_audit(proposal: SchemaInferenceProposal, data: pd.DataFrame) -> AuditPlan:
    """Decide which metrics run and which subgroups are excluded.

    ``proposal`` must already be human-confirmed — this function does not re-validate
    the semantic inference (protected attribute / outcome direction), only applies
    deterministic rules to it.
    """
    has_labels = proposal.true_label_column is not None
    if has_labels:
        metrics_reason = (
            f"True labels present in '{proposal.true_label_column}': Disparate Impact, "
            "Statistical Parity Difference, and Equalized Odds will all be computed."
        )
    else:
        metrics_reason = (
            "No true-label column provided: only Disparate Impact and Statistical Parity "
            "Difference can be computed. Equalized Odds requires ground-truth outcomes and "
            "is skipped; this report will state that limitation explicitly."
        )

    excluded: list[ExcludedSubgroup] = []
    for attr in proposal.protected_attribute_columns:
        if attr not in data.columns:
            continue
        counts = data[attr].value_counts(dropna=True)
        for value, n in counts.items():
            if n < MIN_SUBGROUP_SIZE:
                excluded.append(
                    ExcludedSubgroup(
                        attribute=attr,
                        group_value=str(value),
                        n=int(n),
                        reason=(
                            f"Group has only {n} rows, below the minimum floor of "
                            f"{MIN_SUBGROUP_SIZE}; confidence intervals would be unreliable."
                        ),
                    )
                )

    # A score column only enables the measured-mitigation simulation (Stage 3b) if it's
    # both present and genuinely continuous — a column with <=2 distinct values is really
    # just a re-encoding of the binary decision, not a usable score to threshold-sweep.
    score_column = proposal.score_column
    if score_column is not None and (
        score_column not in data.columns or data[score_column].nunique(dropna=True) <= 2
    ):
        score_column = None

    return AuditPlan(
        include_equalized_odds=has_labels,
        metrics_reason=metrics_reason,
        protected_attribute_columns=proposal.protected_attribute_columns,
        outcome_column=proposal.outcome_column,
        true_label_column=proposal.true_label_column,
        score_column=score_column,
        feature_columns=proposal.feature_columns,
        excluded_subgroups=excluded,
    )
