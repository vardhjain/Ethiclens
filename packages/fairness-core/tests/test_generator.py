"""TS-UNIT-002: synthetic profile attribute-bounds validation."""

from __future__ import annotations

import pytest

from fairness_core.profiles import ProfileConfig, generate_profiles


@pytest.fixture
def config() -> ProfileConfig:
    return ProfileConfig(
        age=(18, 65),
        gender=("Male", "Female", "Non-binary"),
        race=("White", "Black", "Hispanic", "Asian"),
    )


def test_exact_count(config: ProfileConfig) -> None:
    profiles = generate_profiles(n=1000, config=config)
    assert len(profiles) == 1000


def test_age_bounds(config: ProfileConfig) -> None:
    profiles = generate_profiles(n=1000, config=config)
    assert all(isinstance(p["age"], int) for p in profiles)
    assert all(18 <= p["age"] <= 65 for p in profiles)


def test_categorical_membership(config: ProfileConfig) -> None:
    profiles = generate_profiles(n=1000, config=config)
    assert all(p["gender"] in set(config.gender) for p in profiles)
    assert all(p["race"] in set(config.race) for p in profiles)


def test_deterministic(config: ProfileConfig) -> None:
    assert generate_profiles(500, config, seed=7) == generate_profiles(500, config, seed=7)


def test_weighted_sampling_validates() -> None:
    with pytest.raises(ValueError):
        generate_profiles(10, ProfileConfig(gender_weights=(0.5, 0.5)))  # wrong length
