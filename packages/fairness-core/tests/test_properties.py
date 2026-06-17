"""Property-based tests (Hypothesis) for the metric invariants."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fairness_core.metrics import (
    calculate_disparate_impact,
    calculate_spd,
    classify_composite_score,
    compute_composite_bias_score,
)
from fairness_core.profiles import ProfileConfig, generate_profiles

rate = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
positive_rate = st.floats(min_value=1e-6, max_value=1.0, allow_nan=False)


@given(priv=positive_rate, unpriv=rate)
def test_di_non_negative(priv: float, unpriv: float) -> None:
    assert calculate_disparate_impact(priv, unpriv) >= 0.0


@given(r=positive_rate)
def test_di_is_one_at_parity(r: float) -> None:
    assert calculate_disparate_impact(r, r) == pytest.approx(1.0)


@given(priv=rate, unpriv=rate)
def test_spd_in_range(priv: float, unpriv: float) -> None:
    assert -1.0 <= calculate_spd(priv, unpriv) <= 1.0


@given(
    di=st.floats(min_value=1e-6, max_value=5.0),
    spd=st.floats(min_value=-1.0, max_value=1.0),
    eo=st.floats(min_value=0.0, max_value=1.0),
)
def test_composite_bounded(di: float, spd: float, eo: float) -> None:
    score = compute_composite_bias_score(di, spd, eo)
    assert 0.0 <= score <= 1.0


def test_composite_perfect_fairness_is_one() -> None:
    assert compute_composite_bias_score(1.0, 0.0, 0.0) == pytest.approx(1.0)


@given(score=st.floats(min_value=0.0, max_value=1.0))
def test_classify_composite_total(score: float) -> None:
    assert classify_composite_score(score) in {"High Risk", "Medium Risk", "Low Risk"}


@given(n=st.integers(min_value=0, max_value=300))
def test_generate_profiles_count_and_bounds(n: int) -> None:
    config = ProfileConfig(age=(21, 40))
    profiles = generate_profiles(n, config, seed=n)
    assert len(profiles) == n
    assert all(21 <= p["age"] <= 40 for p in profiles)
    assert all(p["gender"] in set(config.gender) for p in profiles)
