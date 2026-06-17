"""Fairness Scorecard PDF, Model Card, and Datasheet generation (FR-007)."""

from __future__ import annotations

from fairness_core.reporting import (
    generate_datasheet,
    generate_model_card,
    generate_scorecard_pdf,
)
from fairness_core.reporting.datasheet import DatasheetInfo
from fairness_core.reporting.scorecard import ScorecardMeta
from fairness_core.types import (
    AuditResult,
    ConfidenceInterval,
    GroupAuditResult,
    MetricName,
    MetricResult,
)


def _audit() -> AuditResult:
    def group(label: str, di: float, flagged: bool) -> GroupAuditResult:
        return GroupAuditResult(
            attribute=label.split(":")[0],
            group_label=label,
            privileged_value="White",
            unprivileged_value=label.split(":")[1],
            n_privileged=500,
            n_unprivileged=300,
            metrics={
                MetricName.DISPARATE_IMPACT.value: MetricResult(
                    name="di", value=di, ci=ConfidenceInterval(di - 0.04, di + 0.04)
                ),
                MetricName.SPD.value: MetricResult(name="spd", value=-0.2),
                MetricName.EQUALIZED_ODDS.value: MetricResult(name="eo", value=0.14),
            },
            flagged=flagged,
        )

    return AuditResult(
        composite_score=0.61,
        composite_band="Medium Risk",
        min_di=0.55,
        groups=[group("race:Black", 0.55, True), group("race:Asian", 0.95, False)],
    )


def test_scorecard_pdf_is_valid() -> None:
    pdf = generate_scorecard_pdf(
        _audit(), ScorecardMeta(model_name="credit-model", dataset="synthetic", seed=42)
    )
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 2000  # a real document, not an empty shell


def test_model_card_has_sections() -> None:
    md = generate_model_card(_audit(), model_name="credit-model", dataset="synthetic")
    assert "# Model Card" in md
    assert "race:Black" in md
    assert "Composite Bias Score" in md
    assert "FLAGGED" in md


def test_datasheet_has_sections() -> None:
    md = generate_datasheet(
        DatasheetInfo(name="german_credit", n_rows=1000, protected_attributes=["sex", "age"])
    )
    assert "# Datasheet" in md
    assert "Motivation" in md and "Composition" in md
    assert "1,000 rows" in md
