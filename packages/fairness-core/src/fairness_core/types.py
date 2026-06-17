"""Core data types shared across the fairness engine.

These are deliberately framework-agnostic (plain dataclasses / enums) so that
``fairness_core`` carries no web or database dependencies. The API layer maps
these to Pydantic/SQLAlchemy models at its boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# --- Thresholds defined by regulation / the EthicLens spec -----------------

#: Four-fifths (80%) rule threshold (EEOC Uniform Guidelines, 1978).
DI_THRESHOLD: float = 0.80

#: |SPD| below this is considered acceptable (open interval, per the STP glossary).
SPD_ACCEPTABLE: float = 0.10

#: Default composite-score weights (Disparate Impact / SPD / Equalized Odds).
DEFAULT_COMPOSITE_WEIGHTS: dict[str, float] = {"di": 0.40, "spd": 0.35, "eo": 0.25}


class MetricName(StrEnum):
    """Canonical metric identifiers (also used as DB enum values)."""

    DISPARATE_IMPACT = "disparate_impact"
    SPD = "spd"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    PREDICTIVE_PARITY = "predictive_parity"
    FPR_BALANCE = "fpr_balance"
    CALIBRATION = "calibration"
    COMPOSITE = "composite"


class Classification(StrEnum):
    """Pass/fail-style labels emitted by the metric classifiers."""

    PASS = "PASS"
    FAIL = "FAIL"
    ACCEPTABLE = "Acceptable"
    FLAGGED = "Flagged"
    #: Returned when a metric cannot be computed (e.g. Equalized Odds with no
    #: ground-truth labels, or a subgroup below the minimum-sample floor).
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class RiskBand(StrEnum):
    """Composite-score risk bands."""

    LOW = "Low Risk"
    MEDIUM = "Medium Risk"
    HIGH = "High Risk"


@dataclass(frozen=True)
class ConfidenceInterval:
    """A two-sided confidence interval for a metric estimate."""

    low: float
    high: float
    level: float = 0.95
    method: str = "bootstrap-bca"

    def excludes(self, value: float) -> bool:
        """True if ``value`` lies entirely outside the interval.

        Used to decide whether a fairness violation is *statistically* real
        rather than an artefact of a small subgroup.
        """
        return value < self.low or value > self.high

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass
class MetricResult:
    """The result of computing a single fairness metric for one group."""

    name: str
    value: float | None
    group_label: str | None = None
    classification: str | None = None
    ci: ConfidenceInterval | None = None
    p_value: float | None = None
    n: int | None = None
    privileged_rate: float | None = None
    unprivileged_rate: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def computable(self) -> bool:
        return self.value is not None and self.classification != Classification.INSUFFICIENT_DATA


@dataclass
class GroupAuditResult:
    """All metrics for a single (attribute, unprivileged-group) comparison."""

    attribute: str
    group_label: str  # e.g. "Race:Black"
    privileged_value: str
    unprivileged_value: str
    n_privileged: int
    n_unprivileged: int
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    flagged: bool = False

    def metric(self, name: str | MetricName) -> MetricResult | None:
        return self.metrics.get(str(getattr(name, "value", name)))


@dataclass
class AuditResult:
    """The full output of :func:`fairness_core.audit.run_audit`."""

    composite_score: float | None
    composite_band: str | None
    min_di: float | None
    groups: list[GroupAuditResult] = field(default_factory=list)
    has_labels: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def flagged_groups(self) -> list[GroupAuditResult]:
        return [g for g in self.groups if g.flagged]


@dataclass
class Recommendation:
    """A single ranked mitigation recommendation (FR-005/FR-006)."""

    rank: int
    strategy: str
    strategy_name: str
    description: str
    #: At recommend-time this is a *projection*; after the mitigation is applied
    #: and re-audited it is replaced by a measured held-out delta.
    estimated_di_improvement: float
    estimated_ci: ConfidenceInterval | None = None
    measured: bool = False
    stage: str = "post"  # pre | in | post
    accuracy_cost: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
