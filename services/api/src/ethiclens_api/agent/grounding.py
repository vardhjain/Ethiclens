"""Deterministic number-grounding validator, shared by Stage 4 (narrative) and Stage 5 (Q&A).

Every numeric token an LLM writes about an audit must be traceable back to the source
scorecard JSON, with tolerance for rounding and percentage-vs-fraction formatting. This
is the guardrail that keeps the LLM from ever fabricating a metric.
"""

from __future__ import annotations

import re

#: Matches numbers, including percentages, negatives, and decimals (e.g. "55%", "-0.12", "1,234").
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*%?")


def extract_source_numbers(json_text: str) -> set[str]:
    """Every normalized numeric token appearing in serialized source JSON."""
    return {_normalize(tok) for tok in _NUMBER_RE.findall(json_text)}


def is_grounded(text: str, source_numbers: set[str]) -> bool:
    """True if every numeric token in ``text`` is traceable to ``source_numbers``."""
    for token in _NUMBER_RE.findall(text):
        if not _close_to_any(_normalize(token), source_numbers):
            return False
    return True


def _normalize(token: str) -> str:
    """Normalize a numeric token for tolerant comparison (drop '%', trailing zeros, commas)."""
    cleaned = token.replace(",", "").rstrip("%")
    try:
        return f"{float(cleaned):.2f}"
    except ValueError:
        return cleaned


def _close_to_any(value: str, source_numbers: set[str]) -> bool:
    try:
        target = float(value)
    except ValueError:
        return value in source_numbers
    for candidate in source_numbers:
        try:
            if abs(float(candidate) - target) <= 0.011:
                return True
            # Percentage vs fraction: "55.00" claimed against source fraction "0.55".
            if abs(float(candidate) * 100 - target) <= 1.0:
                return True
        except ValueError:
            continue
    return False
