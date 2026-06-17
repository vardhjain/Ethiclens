"""fairness-core: the audited, framework-agnostic heart of EthicLens.

Public API (stable):

* Metric primitives — :func:`calculate_disparate_impact`, :func:`calculate_spd`,
  :func:`calculate_eo`, :func:`compute_composite_bias_score`, and the
  ``classify_*`` helpers.
* :func:`run_audit` — the end-to-end audit pipeline (FR-003).
* :func:`generate_profiles` — synthetic demographic personas (FR-002).
* Statistical rigour — :func:`bootstrap_ci`, :func:`disparate_impact_ci`.
"""

from __future__ import annotations

from fairness_core.audit import AttributeSpec, run_audit
from fairness_core.metrics import (
    calculate_disparate_impact,
    calculate_eo,
    calculate_spd,
    classify_composite_score,
    classify_disparate_impact,
    classify_spd,
    compute_composite_bias_score,
)
from fairness_core.mitigation import (
    MitigationResult,
    get_recommendations,
    mitigate_and_reaudit,
    pareto_frontier,
)
from fairness_core.profiles import ProfileConfig, generate_profiles
from fairness_core.seeds import DEFAULT_SEED, set_global_seed
from fairness_core.stats import bootstrap_ci, disparate_impact_ci
from fairness_core.types import (
    AuditResult,
    Classification,
    ConfidenceInterval,
    GroupAuditResult,
    MetricResult,
    Recommendation,
    RiskBand,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # audit
    "run_audit",
    "AttributeSpec",
    # metrics
    "calculate_disparate_impact",
    "classify_disparate_impact",
    "calculate_spd",
    "classify_spd",
    "calculate_eo",
    "compute_composite_bias_score",
    "classify_composite_score",
    # mitigation
    "get_recommendations",
    "mitigate_and_reaudit",
    "MitigationResult",
    "pareto_frontier",
    # profiles
    "generate_profiles",
    "ProfileConfig",
    # stats
    "bootstrap_ci",
    "disparate_impact_ci",
    # seeds
    "set_global_seed",
    "DEFAULT_SEED",
    # types
    "AuditResult",
    "GroupAuditResult",
    "MetricResult",
    "ConfidenceInterval",
    "Recommendation",
    "Classification",
    "RiskBand",
]
