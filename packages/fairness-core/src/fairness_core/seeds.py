"""Deterministic seeding.

Reproducibility is not a nicety for a fairness audit — it is a *requirement*.
The same model + data + config must always produce the same metric values and
the same flag decisions, otherwise an audit trail means nothing. This module
centralises seeding so every entry point (CLI, API worker, notebook) is
deterministic.
"""

from __future__ import annotations

import os
import random

import numpy as np

#: The project-wide default seed. The golden-reference audit is pinned to it.
DEFAULT_SEED: int = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed Python, NumPy and ``PYTHONHASHSEED``. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Seed the legacy global RNG too: some third-party libraries still use it.
    np.random.seed(seed)  # noqa: NPY002
    return seed


def rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return a fresh, independent NumPy ``Generator`` for local use.

    Prefer this over the global NumPy state inside library functions so that
    concurrent audits cannot interfere with one another.
    """
    return np.random.default_rng(seed)
