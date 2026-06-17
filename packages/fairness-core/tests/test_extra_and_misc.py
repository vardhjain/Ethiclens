"""Coverage for the depth metrics, counterfactual probe, seeds, types and CLI."""

from __future__ import annotations

import numpy as np
import pytest

from fairness_core.cli import main, make_biased_dataset, render_scorecard
from fairness_core.metrics import (
    average_odds_difference,
    compute_composite_bias_score,
    equal_opportunity_difference,
    expected_calibration_error,
    fpr_balance_difference,
    predictive_parity_difference,
    theil_index,
)
from fairness_core.metrics.group import compute_group_rates, selection_rate
from fairness_core.profiles import ProfileConfig, counterfactual_probe, generate_profiles
from fairness_core.profiles.generator import profiles_to_frame
from fairness_core.seeds import DEFAULT_SEED, rng, set_global_seed
from fairness_core.stats import bootstrap_ci, two_proportion_ztest
from fairness_core.types import (
    AuditResult,
    Classification,
    ConfidenceInterval,
    GroupAuditResult,
    MetricName,
    MetricResult,
)

# --- depth metrics ---------------------------------------------------------


def test_predictive_parity_and_fpr_balance() -> None:
    assert predictive_parity_difference(0.7, 0.5) == pytest.approx(0.2)
    assert fpr_balance_difference(0.3, 0.1) == pytest.approx(0.2)


def test_equal_opportunity_and_average_odds() -> None:
    assert equal_opportunity_difference(0.9, 0.7) == pytest.approx(0.2)
    # signed average of the (unpriv - priv) TPR and FPR gaps
    assert average_odds_difference(0.9, 0.2, 0.7, 0.1) == pytest.approx(-0.15)


def test_expected_calibration_error_perfect_is_zero() -> None:
    # Scores exactly equal to outcomes in each bin -> ECE 0.
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y_true, y_score, n_bins=5) == pytest.approx(0.0)
    assert np.isnan(expected_calibration_error([], []))


def test_theil_index_equality_is_zero() -> None:
    assert theil_index(np.ones(50)) == pytest.approx(0.0)
    assert theil_index(np.array([1.0, 2.0, 3.0])) > 0.0
    assert np.isnan(theil_index(np.zeros(3)))


def test_composite_handles_zero_di() -> None:
    # DI = 0 -> goodness 0; exercises the di <= 0 branch.
    assert 0.0 <= compute_composite_bias_score(0.0, 0.0, 0.0) <= 1.0


# --- group helpers ---------------------------------------------------------


def test_selection_rate_empty_mask_is_nan() -> None:
    yp = np.array([1, 0, 1])
    assert np.isnan(selection_rate(yp, np.array([False, False, False])))


def test_compute_group_rates_bool_predictions() -> None:
    sens = np.array(["A", "A", "B", "B"])
    yp = np.array([True, False, True, True])
    r = compute_group_rates(sens, yp, "B")
    assert r.selection_rate == pytest.approx(1.0)
    assert r.n == 2


# --- counterfactual probe --------------------------------------------------


def test_counterfactual_probe_detects_attribute_sensitivity() -> None:
    import pandas as pd

    # x kept low so the +0.6 counterfactual bump never clips at 1.0.
    df = pd.DataFrame({"x": [0.1, 0.2, 0.3], "race": ["Black", "Black", "Black"]})

    def predict_score(frame: pd.DataFrame) -> np.ndarray:
        # Model that flips decision purely on the protected attribute.
        bump = (frame["race"] == "White").to_numpy(dtype=float) * 0.6
        return np.clip(frame["x"].to_numpy(dtype=float) + bump, 0, 1)

    res = counterfactual_probe(predict_score, df, "race", "Black", "White")
    assert res.n == 3
    assert res.flip_rate == pytest.approx(1.0)  # every decision flips
    assert res.mean_score_gap == pytest.approx(0.6, abs=1e-9)


def test_counterfactual_probe_empty_subset() -> None:
    import pandas as pd

    df = pd.DataFrame({"x": [0.1], "race": ["White"]})
    res = counterfactual_probe(lambda f: np.array([0.5]), df, "race", "Black", "White")
    assert res.n == 0 and np.isnan(res.flip_rate)


# --- seeds -----------------------------------------------------------------


def test_set_global_seed_is_deterministic() -> None:
    set_global_seed(123)
    a = np.random.rand(5)
    set_global_seed(123)
    b = np.random.rand(5)
    np.testing.assert_allclose(a, b)
    assert set_global_seed() == DEFAULT_SEED
    assert rng(7).integers(0, 100) == rng(7).integers(0, 100)


# --- generator extras ------------------------------------------------------


def test_generator_extra_attributes_and_frame() -> None:
    cfg = ProfileConfig(extra_categorical={"education": ["HS", "BSc", "MSc"]})
    profiles = generate_profiles(20, cfg, seed=3)
    assert all(p["education"] in {"HS", "BSc", "MSc"} for p in profiles)
    frame = profiles_to_frame(profiles)
    assert list(frame.columns)[:3] == ["age", "gender", "race"]
    assert len(frame) == 20


def test_generator_invalid_age_and_negative_n() -> None:
    with pytest.raises(ValueError):
        generate_profiles(5, ProfileConfig(age=(60, 20)))
    with pytest.raises(ValueError):
        generate_profiles(-1)


def test_generator_weighted_sampling_ok() -> None:
    cfg = ProfileConfig(gender=("M", "F"), gender_weights=(0.5, 0.5))
    assert len(generate_profiles(10, cfg, seed=1)) == 10


# --- stats edge cases ------------------------------------------------------


def test_two_proportion_ztest_zero_n() -> None:
    z, p = two_proportion_ztest(0, 0, 1, 10)
    assert np.isnan(z) and np.isnan(p)


def test_bootstrap_percentile_method() -> None:
    data = np.concatenate([np.ones(60), np.zeros(40)])
    ci = bootstrap_ci(lambda a: float(np.mean(a)), data, n_boot=300, method="percentile", seed=0)
    assert ci.method == "bootstrap-percentile"
    assert ci.low < 0.6 < ci.high


# --- types -----------------------------------------------------------------


def test_confidence_interval_predicates() -> None:
    ci = ConfidenceInterval(low=0.4, high=0.6)
    assert ci.contains(0.5) and ci.excludes(0.8) and ci.excludes(0.2)


def test_metric_result_computable_flag() -> None:
    ok = MetricResult(name="di", value=0.5)
    bad = MetricResult(name="eo", value=None, classification=Classification.INSUFFICIENT_DATA.value)
    assert ok.computable and not bad.computable


# --- CLI rendering ---------------------------------------------------------


def test_make_biased_dataset_shape() -> None:
    df, target = make_biased_dataset(n=500, seed=2)
    assert target == "approved" and len(df) == 500
    assert {"income", "credit_score", "race", "approved"} <= set(df.columns)


def test_render_scorecard_contains_verdict() -> None:
    result = AuditResult(
        composite_score=0.61,
        composite_band="Medium Risk",
        min_di=0.55,
        groups=[
            GroupAuditResult(
                attribute="race",
                group_label="race:Black",
                privileged_value="White",
                unprivileged_value="Black",
                n_privileged=400,
                n_unprivileged=300,
                metrics={
                    MetricName.DISPARATE_IMPACT.value: MetricResult(
                        name="di",
                        value=0.55,
                        group_label="race:Black",
                        ci=ConfidenceInterval(0.50, 0.60),
                    ),
                    MetricName.SPD.value: MetricResult(name="spd", value=-0.28),
                    MetricName.EQUALIZED_ODDS.value: MetricResult(name="eo", value=0.14),
                },
                flagged=True,
            )
        ],
    )
    out = render_scorecard(result)
    assert "Fairness Scorecard" in out
    assert "race:Black" in out and "FLAG" in out
    assert "Medium Risk" in out


def test_cli_main_demo_runs() -> None:
    # Small + few bootstraps so the end-to-end CLI path stays fast.
    assert main(["demo", "--n", "600", "--n-boot", "40", "--seed", "1"]) == 0
