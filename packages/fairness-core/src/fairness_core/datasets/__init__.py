"""Built-in datasets for demos, tests and the golden oracle.

``make_biased_lending_dataset`` is a synthetic, *labelled* dataset with a known,
tunable disparate impact — used by the CLI demo and as the bias-injection oracle
(the golden-reference model is trained on it). Real benchmark loaders
(Folktables, COMPAS, German Credit) live in the top-level ``ml/datasets`` package
because they require network access and heavier optional dependencies.
"""

from __future__ import annotations

from fairness_core.datasets.synthetic import make_biased_lending_dataset

__all__ = ["make_biased_lending_dataset"]
