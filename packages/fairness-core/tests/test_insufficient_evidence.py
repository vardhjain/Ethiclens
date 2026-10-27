"""Regression checks: insufficient evidence is never promoted to a fairness verdict."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairness_core.audit import AttributeSpec, run_audit
from fairness_core.reporting.model_card import generate_model_card
from fairness_core.reporting.scorecard import _verdict_text
from fairness_core.types import Classification, MetricName


def _frame(
    p_pred: list[int],
    u_pred: list[int],
    p_true: list[int] | None = None,
    u_true: list[int] | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "group": ["A"] * len(p_pred) + ["B"] * len(u_pred),
            "prediction": p_pred + u_pred,
        }
    )
    if p_true is not None and u_true is not None:
        df["truth"] = p_true + u_true
    return df


def _audit(
    df: pd.DataFrame,
    *,
    compute_ci: bool = True,
    reference: str = "A",
    comparison: str = "B",
):
    return run_audit(
        lambda x: x["prediction"].to_numpy(),
        df,
        [AttributeSpec("group", reference, [comparison])],
        target="truth" if "truth" in df else None,
        feature_columns=["prediction"],
        compute_ci=compute_ci,
        n_boot=100,
        seed=19,
    )


@pytest.mark.parametrize("small_reference", [False, True])
def test_both_rate_subgroups_must_meet_minimum(small_reference: bool) -> None:
    p = [1] * 360 + [0] * 40
    u = [1] * 2 + [0] * 18
    if small_reference:
        p, u = u, p
    df = _frame(p, u)
    group = _audit(df).groups[0]
    di = group.metric(MetricName.DISPARATE_IMPACT)
    spd = group.metric(MetricName.SPD)
    assert di is not None and spd is not None
    assert di.value is not None
    assert di.classification == Classification.INSUFFICIENT_DATA.value
    assert spd.classification == Classification.INSUFFICIENT_DATA.value
    assert di.ci is None
    assert di.p_value is None
    assert not group.flagged
    result = _audit(df)
    assert result.composite_score is None
    assert result.composite_band is None
    assert result.min_di is None


def test_eo_requires_positive_and_negative_labels_on_both_sides() -> None:
    p_pred = [1] * 200 + [0] * 200
    u_pred = [0] * 200 + [1] * 200
    p_true = [1] * 200 + [0] * 200
    u_true = [1] * 20 + [0] * 380
    result = _audit(_frame(p_pred, u_pred, p_true, u_true))
    group = result.groups[0]
    eo = group.metric(MetricName.EQUALIZED_ODDS)
    di = group.metric(MetricName.DISPARATE_IMPACT)
    assert eo is not None and di is not None
    assert eo.value is not None and eo.value > 0.10
    assert eo.classification == Classification.INSUFFICIENT_DATA.value
    assert eo.ci is None
    assert not group.flagged
    assert di.value == pytest.approx(1.0)
    assert result.composite_score is None
    assert result.min_di == pytest.approx(1.0)
    assert "insufficient evidence" in result.metadata["composite_reason"]
    assert result.metadata["insufficient_groups"] == [group.group_label]


def test_unlabeled_audit_does_not_earn_perfect_equalized_odds_credit() -> None:
    result = _audit(_frame([1] * 360 + [0] * 40, [1] * 120 + [0] * 280))
    group = result.groups[0]
    di = group.metric(MetricName.DISPARATE_IMPACT)
    eo = group.metric(MetricName.EQUALIZED_ODDS)
    assert di is not None and eo is not None
    assert di.classification == Classification.FAIL.value
    assert di.ci is not None and di.ci.high < 0.80
    assert group.flagged
    assert eo.value is None
    assert result.composite_score is None
    assert result.composite_band is None
    assert result.min_di == pytest.approx(120 / 280 / (360 / 400))
    assert "ground-truth" in result.metadata["composite_reason"]
    assert "not assessed" not in _verdict_text(result, 0.8).lower()


def test_undefined_di_is_null_not_nan_or_zero() -> None:
    result = _audit(_frame([0] * 200, [0] * 200))
    group = result.groups[0]
    di = group.metric(MetricName.DISPARATE_IMPACT)
    assert di is not None
    assert di.value is None
    assert di.classification == Classification.INSUFFICIENT_DATA.value
    assert group.flagged is False
    assert result.composite_score is None


def test_explicit_nonexistent_comparison_group_is_not_considered_a_pass() -> None:
    result = _audit(_frame([1] * 150 + [0] * 150, []), comparison="missing")
    group = result.groups[0]
    di = group.metric(MetricName.DISPARATE_IMPACT)
    spd = group.metric(MetricName.SPD)
    assert di is not None and di.value is None
    assert spd is not None and spd.value is None
    assert di.classification == Classification.INSUFFICIENT_DATA.value
    assert group.flagged is False
    assert result.composite_band is None


def test_no_ci_never_means_statistically_supported_violation() -> None:
    result = _audit(
        _frame([1] * 360 + [0] * 40, [1] * 80 + [0] * 320),
        compute_ci=False,
    )
    group = result.groups[0]
    di = group.metric(MetricName.DISPARATE_IMPACT)
    assert di is not None
    assert di.classification == Classification.FAIL.value
    assert di.ci is None
    assert group.flagged is False


def test_inconclusive_scorecard_does_not_claim_all_groups_are_fair() -> None:
    df = _frame([1] * 360 + [0] * 40, [1] * 2 + [0] * 18)
    result = _audit(df)
    verdict = _verdict_text(result, 0.8)
    assert "Inconclusive" in verdict
    card = generate_model_card(result)
    assert "INCONCLUSIVE" in card
    assert "within thresholds" not in card


def test_labeled_large_groups_still_have_complete_composite() -> None:
    rng = np.random.default_rng(5)
    p_true = rng.binomial(1, 0.5, 400).tolist()
    u_true = rng.binomial(1, 0.5, 400).tolist()
    p_pred = rng.binomial(1, 0.65, 400).tolist()
    u_pred = rng.binomial(1, 0.55, 400).tolist()
    result = _audit(_frame(p_pred, u_pred, p_true, u_true))
    assert result.composite_score is not None
    assert result.composite_band is not None
    assert result.metadata["insufficient_groups"] == []
    assert result.metadata["composite_complete"] is True
    assert result.metadata["composite_reason"] is None



def test_partial_di_evidence_never_produces_a_selective_minimum() -> None:
    """One adequately powered cohort cannot mask an underpowered second group."""
    df = pd.DataFrame(
        {
            "group": ["A"] * 400 + ["B"] * 400 + ["C"] * 20,
            "prediction": (
                [1] * 360 + [0] * 40
                + [1] * 120 + [0] * 280
                + [1] + [0] * 19
            ),
        }
    )
    result = run_audit(
        lambda x: x["prediction"].to_numpy(),
        df,
        [AttributeSpec("group", "A")],
        feature_columns=["prediction"],
        compute_ci=False,
    )
    assert len(result.groups) == 2
    b_di = result.groups[0].metric(MetricName.DISPARATE_IMPACT)
    c_di = result.groups[1].metric(MetricName.DISPARATE_IMPACT)
    assert b_di is not None and b_di.computable
    assert c_di is not None and not c_di.computable
    assert result.min_di is None
    assert result.composite_score is None


def test_di_headline_available_without_labels_but_not_composite() -> None:
    result = _audit(_frame([1] * 360 + [0] * 40, [1] * 120 + [0] * 280))
    assert result.min_di == pytest.approx(1 / 3)
    assert result.composite_score is None
    assert result.composite_band is None
    assert result.metadata["composite_complete"] is False
