"""Bias mitigation: ranked recommendations, real strategies, Pareto trade-offs.

Two layers, deliberately separated to keep an honest contract (the original spec
blurred them):

* :func:`get_recommendations` — given an :class:`AuditResult`, returns ranked
  recommendations for each *flagged* group with a **projected** improvement.
  Cheap; no model access. Matches the STP ``get_recommendations`` signature.
* :func:`mitigate_and_reaudit` — actually *runs* a mitigation strategy, then
  re-audits on a **held-out split**, so the reported Disparate-Impact change is
  a **measured** number, never an estimate. Powers the API's apply/re-audit flow
  (FR-009).
* :func:`pareto_frontier` — sweeps a constrained reduction to trace the
  accuracy-vs-fairness trade-off, with bootstrap CIs.
"""

from __future__ import annotations

from fairness_core.mitigation.pareto import ParetoPoint, pareto_frontier
from fairness_core.mitigation.recommender import (
    STRATEGY_CATALOG,
    get_recommendations,
)
from fairness_core.mitigation.strategies import (
    AVAILABLE_STRATEGIES,
    MitigationResult,
    mitigate_and_reaudit,
    reweighing_sample_weights,
)

__all__ = [
    "get_recommendations",
    "STRATEGY_CATALOG",
    "mitigate_and_reaudit",
    "MitigationResult",
    "reweighing_sample_weights",
    "AVAILABLE_STRATEGIES",
    "pareto_frontier",
    "ParetoPoint",
]
