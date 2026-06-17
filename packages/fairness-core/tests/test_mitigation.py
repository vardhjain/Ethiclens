"""Mitigation: recommendation structure (TS-INT-002) and *measured* effect."""

from __future__ import annotations

import numpy as np
import pytest

from fairness_core.datasets import make_biased_lending_dataset
from fairness_core.mitigation import (
    get_recommendations,
    mitigate_and_reaudit,
    pareto_frontier,
    reweighing_sample_weights,
)
from fairness_core.types import (
    AuditResult,
    GroupAuditResult,
    MetricName,
    MetricResult,
)

FEATURES = ["income", "credit_score", "debt_ratio"]


def _flagged_audit() -> AuditResult:
    """Mirror TS-INT-002: Race:Black flagged (DI 0.62), Gender:Female passing (DI 0.83)."""

    def group(label: str, di: float, flagged: bool) -> GroupAuditResult:
        return GroupAuditResult(
            attribute=label.split(":")[0],
            group_label=label,
            privileged_value="ref",
            unprivileged_value=label.split(":")[1],
            n_privileged=500,
            n_unprivileged=300,
            metrics={MetricName.DISPARATE_IMPACT.value: MetricResult(name="di", value=di)},
            flagged=flagged,
        )

    return AuditResult(
        composite_score=0.6,
        composite_band="Medium Risk",
        min_di=0.62,
        groups=[group("Race:Black", 0.62, True), group("Gender:Female", 0.83, False)],
    )


def _trained_model(n: int = 3000, seed: int = 7):
    from sklearn.linear_model import LogisticRegression

    df, target = make_biased_lending_dataset(n=n, seed=seed, disadvantage=0.6)
    model = LogisticRegression(max_iter=2000, random_state=seed).fit(df[FEATURES], df[target])
    return model, df, target


# --- TS-INT-002: recommendation structure ----------------------------------


def test_recommendations_only_for_flagged_groups() -> None:
    recs = get_recommendations(_flagged_audit())
    assert "Race:Black" in recs
    assert "Gender:Female" not in recs  # DI 0.83 > 0.80 threshold


def test_recommendations_ranked_and_complete() -> None:
    recs = get_recommendations(_flagged_audit())["Race:Black"]
    assert len(recs) >= 2
    for r in recs:
        assert r.rank and r.strategy_name and r.description
        assert r.estimated_di_improvement is not None
        assert r.measured is False  # honestly labelled as a projection
    ranks = [r.rank for r in recs]
    assert ranks == list(range(1, len(recs) + 1))  # 1,2,3... no gaps
    # Ranked best-projected first.
    assert recs[0].estimated_di_improvement >= recs[-1].estimated_di_improvement


# --- reweighing math -------------------------------------------------------


def test_reweighing_balances_group_outcome() -> None:
    rng = np.random.default_rng(0)
    s = rng.choice(["A", "B"], size=1000)
    y = (rng.uniform(size=1000) < np.where(s == "A", 0.7, 0.3)).astype(int)
    w = reweighing_sample_weights(y, s)
    # In the *weighted* data, P(Y=1 | A) should be equal across groups.
    rate_a = w[(s == "A") & (y == 1)].sum() / w[s == "A"].sum()
    rate_b = w[(s == "B") & (y == 1)].sum() / w[s == "B"].sum()
    assert rate_a == pytest.approx(rate_b, abs=1e-9)


# --- measured mitigation (the differentiator) ------------------------------


def test_threshold_optimizer_crosses_threshold() -> None:
    model, df, target = _trained_model()
    res = mitigate_and_reaudit(
        model,
        df,
        "race",
        "White",
        "Black",
        target=target,
        feature_columns=FEATURES,
        strategy="threshold_optimizer",
        n_boot=200,
        seed=7,
    )
    assert res.di_before < 0.80  # the biased baseline
    assert res.di_after > res.di_before  # mitigation helped
    assert res.crossed_threshold  # demographic parity pushes DI past 0.80
    assert res.di_after_ci is not None


def test_reweighing_improves_di() -> None:
    model, df, target = _trained_model()
    res = mitigate_and_reaudit(
        model,
        df,
        "race",
        "White",
        "Black",
        target=target,
        feature_columns=FEATURES,
        strategy="reweighing",
        n_boot=200,
        seed=7,
    )
    assert res.di_after > res.di_before
    assert res.accuracy_after == pytest.approx(res.accuracy_before, abs=0.15)


def test_unknown_strategy_raises() -> None:
    model, df, target = _trained_model(n=800)
    with pytest.raises(KeyError):
        mitigate_and_reaudit(
            model,
            df,
            "race",
            "White",
            "Black",
            target=target,
            feature_columns=FEATURES,
            strategy="magic",
        )


# --- Pareto frontier -------------------------------------------------------


def test_pareto_frontier_traces_tradeoff() -> None:
    from sklearn.linear_model import LogisticRegression

    _, df, target = _trained_model(n=2500, seed=5)
    points = pareto_frontier(
        lambda: LogisticRegression(max_iter=1000),
        df,
        "race",
        "White",
        "Black",
        target=target,
        feature_columns=FEATURES,
        grid_size=6,
        n_boot=150,
        seed=5,
    )
    assert len(points) >= 3
    assert any(p.label == "baseline" for p in points)
    assert any(p.pareto_optimal for p in points)
    # A fairness-constrained point should achieve higher DI than the baseline.
    baseline = next(p for p in points if p.label == "baseline")
    assert max(p.di for p in points) > baseline.di


def test_cli_mitigate_runs() -> None:
    from fairness_core.cli import main

    assert main(["mitigate", "--n", "800", "--n-boot", "40", "--seed", "7"]) == 0
