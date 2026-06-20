"""Equalized-Odds CI, classification, and EO-based flagging in run_audit."""

from __future__ import annotations

import numpy as np

from fairness_core import run_audit
from fairness_core.audit import AttributeSpec
from fairness_core.datasets import make_biased_lending_dataset
from fairness_core.metrics import classify_equalized_odds
from fairness_core.stats import equalized_odds_ci
from fairness_core.types import MetricName


def test_classify_equalized_odds() -> None:
    assert classify_equalized_odds(0.05) == "Acceptable"
    assert classify_equalized_odds(0.20) == "Flagged"


def test_equalized_odds_ci_brackets_the_gap() -> None:
    rng = np.random.default_rng(0)
    n = 3000
    sf = rng.choice(["A", "B"], size=n)
    y_true = (rng.uniform(size=n) < np.where(sf == "A", 0.5, 0.4)).astype(int)
    # Group B gets far more positive predictions -> inflated FPR -> an Equalized-Odds gap.
    y_pred = (rng.uniform(size=n) < np.where(sf == "B", 0.6, 0.3)).astype(int)
    ci = equalized_odds_ci(sf, y_pred, y_true, "A", "B", n_boot=400, seed=0)
    assert 0.0 <= ci.low <= ci.high
    assert ci.high > 0.0


def test_run_audit_reports_eo_ci_and_classification() -> None:
    from sklearn.linear_model import LogisticRegression

    df, target = make_biased_lending_dataset(n=3000, seed=1)
    feats = ["income", "credit_score", "debt_ratio"]
    model = LogisticRegression(max_iter=2000, random_state=1).fit(df[feats], df[target])
    result = run_audit(
        model,
        df,
        [AttributeSpec("race")],
        target=target,
        feature_columns=feats,
        n_boot=300,
        seed=1,
    )
    black = next(g for g in result.groups if g.group_label == "race:Black")
    eo = black.metric(MetricName.EQUALIZED_ODDS)
    assert eo is not None and eo.value is not None
    assert eo.ci is not None  # the engine now puts a bootstrap CI on Equalized Odds
    assert eo.classification in {"Flagged", "Acceptable"}
    # The injected-bias group is flagged (here via both DI and the new EO criterion).
    assert black.flagged is True
