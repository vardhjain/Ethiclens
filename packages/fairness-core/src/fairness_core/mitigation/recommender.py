"""Ranked mitigation recommendations (FR-005 / FR-006).

``get_recommendations`` consumes an :class:`AuditResult` and returns, for each
**flagged** group, a ranked list of candidate strategies with a **projected**
Disparate-Impact improvement and a plain-language explanation. The projection is
honestly labelled (``measured=False``); the *measured* held-out delta is produced
later by :func:`fairness_core.mitigation.mitigate_and_reaudit`.
"""

from __future__ import annotations

from dataclasses import dataclass

from fairness_core.types import DI_THRESHOLD, AuditResult, MetricName, Recommendation

__all__ = ["STRATEGY_CATALOG", "get_recommendations", "StrategyInfo"]


@dataclass(frozen=True)
class StrategyInfo:
    key: str
    name: str
    stage: str  # pre | in | post
    efficacy: float  # typical fraction of the gap a strategy tends to close
    description: str


#: Ordered by typical efficacy. ``post`` strategies are preferred for opaque,
#: already-trained models (they need only predictions + the protected attribute).
STRATEGY_CATALOG: list[StrategyInfo] = [
    StrategyInfo(
        "threshold_optimizer",
        "Group-specific decision thresholds",
        "post",
        0.92,
        "Calibrate a separate decision threshold per group (Fairlearn ThresholdOptimizer) so "
        "selection rates equalise. Lowest-risk option: it needs only the model's scores and the "
        "protected attribute, never a retrain.",
    ),
    StrategyInfo(
        "exponentiated_gradient",
        "Constrained retraining (reductions)",
        "in",
        0.85,
        "Retrain the model under an explicit demographic-parity constraint "
        "(Fairlearn ExponentiatedGradient). Strong fairness gains, but requires retraining and "
        "access to the training data.",
    ),
    StrategyInfo(
        "reweighing",
        "Reweighing the training data",
        "pre",
        0.70,
        "Reweight training samples (Kamiran-Calders) so protected group and outcome become "
        "independent, then retrain. Model-agnostic and does not distort feature values.",
    ),
    StrategyInfo(
        "correlation_remover",
        "Remove protected-attribute correlation",
        "pre",
        0.55,
        "Project the features to remove their linear correlation with the protected attribute "
        "(Fairlearn CorrelationRemover) before retraining. A lighter-touch pre-processing step.",
    ),
]


def _project_improvement(di: float, efficacy: float) -> float:
    """A conservative projected DI lift; the real value is measured after applying."""
    headroom = max(0.0, (DI_THRESHOLD + 0.10) - di)  # aim a little past the 0.80 line
    projected_post = min(0.98, di + headroom * efficacy)
    return round(projected_post - di, 4)


def get_recommendations(
    audit_result: AuditResult, model_metadata: dict | None = None
) -> dict[str, list[Recommendation]]:
    """Return ``{group_label: [ranked Recommendation, ...]}`` for flagged groups only."""
    out: dict[str, list[Recommendation]] = {}
    for group in audit_result.flagged_groups:
        di_metric = group.metric(MetricName.DISPARATE_IMPACT)
        if di_metric is None or di_metric.value is None:
            continue
        di = di_metric.value
        recs: list[Recommendation] = []
        for rank, strat in enumerate(STRATEGY_CATALOG, start=1):
            improvement = _project_improvement(di, strat.efficacy)
            recs.append(
                Recommendation(
                    rank=rank,
                    strategy=strat.key,
                    strategy_name=strat.name,
                    description=(
                        f"{strat.description} Projected Disparate-Impact lift "
                        f"~+{improvement:.2f} (from {di:.2f}); confirm by applying and re-auditing."
                    ),
                    estimated_di_improvement=improvement,
                    measured=False,
                    stage=strat.stage,
                )
            )
        out[group.group_label] = recs
    return out
