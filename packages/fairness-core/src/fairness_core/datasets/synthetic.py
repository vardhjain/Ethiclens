"""Synthetic, labelled datasets with a known disparate impact."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fairness_core.seeds import DEFAULT_SEED

__all__ = ["make_biased_lending_dataset"]


def make_biased_lending_dataset(
    n: int = 4000, seed: int = DEFAULT_SEED, disadvantage: float = 0.6
) -> tuple[pd.DataFrame, str]:
    """A synthetic lending dataset exhibiting disparate impact by race.

    The bias is modelled the way it most often appears in the real world:
    **proxy discrimination**. The disadvantaged group has systematically weaker
    *inputs* (lower income & credit, higher debt) due to historical inequality,
    so a model trained only on those legitimate features — never on race —
    still under-selects the group. The label is a *fair function of the
    features* (no direct race term). ``disadvantage`` scales the gap and is used
    by the golden oracle to dial the Disparate Impact to a target value.

    Returns ``(dataframe, target_column)``.
    """
    rng = np.random.default_rng(seed)
    races = rng.choice(["White", "Black", "Hispanic", "Asian"], size=n, p=[0.45, 0.25, 0.20, 0.10])
    genders = rng.choice(["Male", "Female"], size=n)
    disadv = (races == "Black").astype(float) * disadvantage

    income = (rng.normal(60_000, 16_000, n) - 16_000 * disadv).clip(12_000, 200_000)
    credit = (rng.normal(690, 65, n) - 55 * disadv).clip(300, 850)
    debt_ratio = (rng.uniform(0.05, 0.55, n) + 0.08 * disadv).clip(0.0, 0.95)

    # Fair creditworthiness signal: a function of features only (no race term).
    z = 0.50 * (credit - 690) / 65 + 0.40 * (income - 60_000) / 16_000 - 0.60 * (debt_ratio - 0.30)
    approve_prob = 1.0 / (1.0 + np.exp(-z))
    approved = (rng.uniform(0, 1, n) < approve_prob).astype(int)

    df = pd.DataFrame(
        {
            "income": income,
            "credit_score": credit,
            "debt_ratio": debt_ratio,
            "race": races,
            "gender": genders,
            "approved": approved,
        }
    )
    return df, "approved"
