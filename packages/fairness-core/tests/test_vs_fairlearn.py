"""Cross-validate the from-scratch metrics against Fairlearn to 1e-9.

This is the correctness proof: our hand-written formulas must agree with the
field-standard library to floating-point tolerance. If Fairlearn is not
installed the suite is skipped (it is an optional ``[validation]`` extra).
"""

from __future__ import annotations

import numpy as np
import pytest

fairlearn_metrics = pytest.importorskip("fairlearn.metrics")

from fairness_core.metrics import (  # noqa: E402
    calculate_disparate_impact,
    calculate_eo,
    calculate_spd,
    compute_group_rates,
)

TOL = 1e-9


@pytest.fixture
def labelled_two_group() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 5000
    sf = rng.choice(["A", "B"], size=n)
    # Group B is selected less and has different error behaviour.
    base = np.where(sf == "A", 0.6, 0.35)
    y_true = (rng.uniform(size=n) < np.where(sf == "A", 0.55, 0.45)).astype(int)
    y_pred = (rng.uniform(size=n) < base).astype(int)
    return y_true, y_pred, sf


def _privileged(sf: np.ndarray, y_pred: np.ndarray) -> tuple[str, str]:
    rates = {v: compute_group_rates(sf, y_pred, v).selection_rate for v in np.unique(sf)}
    priv = max(rates, key=rates.get)
    unpriv = min(rates, key=rates.get)
    return priv, unpriv


def test_disparate_impact_matches_demographic_parity_ratio(labelled_two_group) -> None:
    y_true, y_pred, sf = labelled_two_group
    priv, unpriv = _privileged(sf, y_pred)
    p = compute_group_rates(sf, y_pred, priv)
    u = compute_group_rates(sf, y_pred, unpriv)
    ours = calculate_disparate_impact(p.selection_rate, u.selection_rate)
    theirs = fairlearn_metrics.demographic_parity_ratio(y_true, y_pred, sensitive_features=sf)
    assert ours == pytest.approx(theirs, abs=TOL)


def test_spd_matches_demographic_parity_difference(labelled_two_group) -> None:
    y_true, y_pred, sf = labelled_two_group
    priv, unpriv = _privileged(sf, y_pred)
    p = compute_group_rates(sf, y_pred, priv)
    u = compute_group_rates(sf, y_pred, unpriv)
    ours = abs(calculate_spd(p.selection_rate, u.selection_rate))
    theirs = fairlearn_metrics.demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
    assert ours == pytest.approx(theirs, abs=TOL)


def test_group_rates_match_fairlearn_metricframe(labelled_two_group) -> None:
    y_true, y_pred, sf = labelled_two_group
    mf = fairlearn_metrics.MetricFrame(
        metrics={
            "tpr": fairlearn_metrics.true_positive_rate,
            "fpr": fairlearn_metrics.false_positive_rate,
            "sel": fairlearn_metrics.selection_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sf,
    )
    for g in np.unique(sf):
        r = compute_group_rates(sf, y_pred, g, y_true)
        assert r.tpr == pytest.approx(mf.by_group.loc[g, "tpr"], abs=TOL)
        assert r.fpr == pytest.approx(mf.by_group.loc[g, "fpr"], abs=TOL)
        assert r.selection_rate == pytest.approx(mf.by_group.loc[g, "sel"], abs=TOL)


def test_equalized_odds_matches_fairlearn(labelled_two_group) -> None:
    y_true, y_pred, sf = labelled_two_group
    priv, unpriv = _privileged(sf, y_pred)
    p = compute_group_rates(sf, y_pred, priv, y_true)
    u = compute_group_rates(sf, y_pred, unpriv, y_true)
    ours = calculate_eo(p.tpr, p.fpr, u.tpr, u.fpr)
    theirs = fairlearn_metrics.equalized_odds_difference(y_true, y_pred, sensitive_features=sf)
    assert ours == pytest.approx(theirs, abs=TOL)
