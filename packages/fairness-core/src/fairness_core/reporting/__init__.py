"""Report artefacts: the Fairness Scorecard PDF, Model Cards, and Datasheets.

ReportLab (pure-Python, no native dependencies) renders the PDF, so it builds
identically on Windows, macOS and Linux. Model Cards (Mitchell et al. 2019) and
Datasheets (Gebru et al. 2021) are produced as Markdown.
"""

from __future__ import annotations

from fairness_core.reporting.datasheet import generate_datasheet
from fairness_core.reporting.model_card import generate_model_card
from fairness_core.reporting.scorecard import ScorecardMeta, generate_scorecard_pdf

__all__ = [
    "generate_scorecard_pdf",
    "ScorecardMeta",
    "generate_model_card",
    "generate_datasheet",
]
