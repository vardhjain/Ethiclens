"""End-to-end audit + statistics tests."""

from __future__ import annotations

import numpy as np
import pytest

from fairness_core.audit import AttributeSpec, run_audit
from fairness_core.cli import make_biased_dataset
from fairness_core.stats import (
    disparate_impact_ci,
    has_sufficient_data,
    two_proportion_ztest,
)
from fairness_core.types import Classification, MetricName


def _toy_model_dataset():
    from sklearn.linear_model import LogisticRegression

    df, target = make_biased_dataset(n=3000, seed=1)
    features = ["income", "credit_score", "debt_ratio"]
    model = LogisticRegression(max_iter=1000).fit(df[features], df[target])
    return model, df, target, features


def test_run_audit_flags_biased_group() -> None:
    model, df, target, features = _toy_model_dataset()
    result = run_audit(
        model,
        df,
        [AttributeSpec("race")],
        target=target,
        feature_columns=features,
        n_boot=400,
        seed=1,
    )
    assert result.composite_score is not None
    black = next(g for g in result.groups if g.group_label == "race:Black")
    di = black.metric(MetricName.DISPARATE_IMPACT)
    assert di.value is not None and di.value < 0.80  # injected bias is detected


def test_equalized_odds_insufficient_without_labels() -> None:
    model, df, _target, features = _toy_model_dataset()
    # Audit WITHOUT a target -> Equalized Odds must be INSUFFICIENT_DATA, not faked.
    result = run_audit(
        model,
        df,
        [AttributeSpec("race")],
        target=None,
        feature_columns=features,
        compute_ci=False,
    )
    g = result.groups[0]
    eo = g.metric(MetricName.EQUALIZED_ODDS)
    assert eo.value is None
    assert eo.classification == Classification.INSUFFICIENT_DATA.value


def test_two_proportion_ztest_detects_difference() -> None:
    _z, p = two_proportion_ztest(80, 100, 40, 100)
    assert p < 0.001
    _z0, p0 = two_proportion_ztest(50, 100, 50, 100)
    assert p0 == pytest.approx(1.0)


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    sf = rng.choice(["A", "B"], size=n)
    y_pred = (rng.uniform(size=n) < np.where(sf == "A", 0.6, 0.4)).astype(int)
    ci = disparate_impact_ci(sf, y_pred, "A", "B", n_boot=500, seed=0)
    assert ci.low < ci.high
    assert ci.low <= (0.4 / 0.6) <= ci.high + 0.15  # point estimate near CI


def test_min_subgroup_policy() -> None:
    assert has_sufficient_data(150) is True
    assert has_sufficient_data(50) is False
    assert has_sufficient_data(200, n_positive=10, n_negative=200, error_metric=True) is False
    assert has_sufficient_data(200, n_positive=40, n_negative=60, error_metric=True) is True
