"""Real fairness-benchmark loaders.

Each loader returns a :class:`FairnessDataset` with a standard shape so the same
audit code runs across all of them. These require network access on first use
(results are cached under ``data/raw``); the bundled synthetic oracle in
``fairness_core.datasets`` does not.
"""

from __future__ import annotations

from ml.datasets.loaders import (
    FairnessDataset,
    available_datasets,
    load_adult,
    load_compas,
    load_dataset,
    load_folktables_acsincome,
    load_german_credit,
)

__all__ = [
    "FairnessDataset",
    "available_datasets",
    "load_dataset",
    "load_german_credit",
    "load_compas",
    "load_adult",
    "load_folktables_acsincome",
]
