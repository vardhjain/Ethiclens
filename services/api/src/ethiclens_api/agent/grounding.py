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


#: Both the claim and every source number are already independently rounded to 2
#: decimals by _normalize() before this comparison runs, so this only needs to absorb
#: residual rounding-boundary drift (e.g. a source of 0.5949999 normalizing to "0.59"
#: while a slightly different intermediate computation rounds to "0.60"). It must stay
#: well under 0.01: a full extra hundredth of slack would let a narrative claim "0.79"
#: pass against a source value of "0.80" — a different four-fifths-threshold verdict,
#: not a rounding difference.
_ABS_TOLERANCE = 0.005
_PCT_TOLERANCE = _ABS_TOLERANCE * 100


def _close_to_any(value: str, source_numbers: set[str]) -> bool:
    try:
        target = float(value)
    except ValueError:
        return value in source_numbers
    for candidate in source_numbers:
        try:
            candidate_value = float(candidate)
        except ValueError:
            continue
        if abs(candidate_value - target) <= _ABS_TOLERANCE:
            return True
        # Percentage vs fraction, checked both ways: a claim of "55%" against a source
        # fraction "0.55", or (symmetrically) a claim of "0.55" against a source
        # expressed on a 0-100 scale.
        if (
            abs(candidate_value * 100 - target) <= _PCT_TOLERANCE
            or abs(candidate_value - target * 100) <= _PCT_TOLERANCE
        ):
            return True
    return False
