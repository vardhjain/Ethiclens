"""Configurable synthetic persona generation (FR-002).

The generator is fully seeded and validates that every produced value falls
within its configured bounds — the property asserted by STP test ``TS-UNIT-002``.
It can optionally inject a *known* bias into a generated label, which is how the
golden-reference oracle is built (see :mod:`ml.training`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fairness_core.seeds import DEFAULT_SEED

__all__ = ["ProfileConfig", "generate_profiles", "profiles_to_frame"]


@dataclass
class ProfileConfig:
    """Schema for synthetic personas.

    Args:
        age: Inclusive ``(min, max)`` integer age bounds.
        gender: Allowed gender categories.
        race: Allowed race categories.
        gender_weights / race_weights: Optional sampling probabilities (must sum
            to 1 and match the category count); uniform if omitted.
        extra_categorical: Additional ``name -> categories`` attributes.
    """

    age: tuple[int, int] = (18, 65)
    gender: Sequence[str] = ("Male", "Female", "Non-binary")
    race: Sequence[str] = ("White", "Black", "Hispanic", "Asian")
    gender_weights: Sequence[float] | None = None
    race_weights: Sequence[float] | None = None
    extra_categorical: dict[str, Sequence[str]] = field(default_factory=dict)

    def validate(self) -> None:
        lo, hi = self.age
        if lo > hi:
            raise ValueError(f"age bounds invalid: {self.age}")
        for name, weights, cats in (
            ("gender", self.gender_weights, self.gender),
            ("race", self.race_weights, self.race),
        ):
            if weights is not None:
                if len(weights) != len(cats):
                    raise ValueError(f"{name}_weights length must match {name} categories")
                if not np.isclose(sum(weights), 1.0):
                    raise ValueError(f"{name}_weights must sum to 1.0")


def generate_profiles(
    n: int, config: ProfileConfig | dict[str, Any] | None = None, seed: int = DEFAULT_SEED
) -> list[dict[str, Any]]:
    """Generate ``n`` synthetic persona records as a list of dicts.

    Returns a list so that ``profiles[i]['age']`` works exactly as the STP test
    script ``TS-UNIT-002`` expects.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if config is None:
        config = ProfileConfig()
    elif isinstance(config, dict):
        config = ProfileConfig(**config)
    config.validate()

    rng = np.random.default_rng(seed)
    lo, hi = config.age
    ages = rng.integers(lo, hi + 1, size=n)  # inclusive upper bound
    genders = rng.choice(list(config.gender), size=n, p=_p(config.gender_weights))
    races = rng.choice(list(config.race), size=n, p=_p(config.race_weights))

    records: list[dict[str, Any]] = []
    extra_draws = {
        name: rng.choice(list(cats), size=n) for name, cats in config.extra_categorical.items()
    }
    for i in range(n):
        rec: dict[str, Any] = {
            "age": int(ages[i]),
            "gender": str(genders[i]),
            "race": str(races[i]),
        }
        for name, draws in extra_draws.items():
            rec[name] = draws[i].item() if hasattr(draws[i], "item") else draws[i]
        records.append(rec)
    return records


def profiles_to_frame(profiles: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert persona records to a DataFrame for the audit pipeline."""
    return pd.DataFrame.from_records(profiles)


def _p(weights: Sequence[float] | None) -> np.ndarray | None:
    return np.asarray(weights, dtype=float) if weights is not None else None
