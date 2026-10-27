"""Deterministic evidence checks for LLM audit explanations.

Numeric matching alone cannot establish truth. This module also rejects a
bounded set of important categorical contradictions. It is not a general
natural-language entailment verifier.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

_log = logging.getLogger("ethiclens.agent.grounding")
_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)"
    r"(?:\.\d+)?(?:[eE][+-]?\d+)?%?(?![\w.])"
)
_GROUP_RE = re.compile(r"\b[a-zA-Z][\w.-]*:[a-zA-Z][\w.-]*\b")
_CLEARANCE_RE = re.compile(
    r"\b(?:legally compliant|legally cleared|certified fair|regulatory clearance|"
    r"approved for deployment|cleared for deployment|safe to deploy|guaranteed fair)\b",
    re.IGNORECASE,
)
_NO_FLAG_RE = re.compile(
    r"\b(?:no (?:groups? (?:were |are )?)?flagged|none (?:were |are )?flagged|"
    r"no (?:bias|disparit(?:y|ies)|violations?) (?:was |were )?detected|"
    r"(?:all|every) (?:audited )?groups? (?:passed|pass|meet|are fair))\b",
    re.IGNORECASE,
)
_PASS_RE = re.compile(
    r"\b(?:all (?:groups? |metrics )?(?:passed|meet)|"
    r"the (?:audit|model) (?:passed|passes|is fair|is unbiased)|"
    r"(?:fairness|compliance) (?:is )?(?:verified|confirmed))\b",
    re.IGNORECASE,
)
_MEASURED_RE = re.compile(
    r"\b(?:measured (?:mitigation|improvement|gain|impact)|"
    r"held[- ]out (?:improvement|gain|results?|data)|"
    r"(?:improvement|gain|impact) (?:was|is) measured)\b",
    re.IGNORECASE,
)
_NOT_MEASURED_RE = re.compile(
    r"\b(?:not|never|unavailable|no)\s+(?:actually )?measured\b", re.IGNORECASE
)
_EO_COMPUTED_RE = re.compile(
    r"\b(?:equalized odds|EO)\s+(?:was|were|is|has been)?\s*"
    r"(?:computed|calculated|measured|passed|acceptable)\b",
    re.IGNORECASE,
)

_ABS_TOLERANCE = 0.005
_PCT_TOLERANCE = 0.5


def _normalize(token: str) -> str:
    """Normalize numerical tokens to the precision of displayed scorecards."""
    cleaned = token.replace(",", "").rstrip("%")
    try:
        value = float(cleaned)
        return f"{value:.2f}" if math.isfinite(value) else cleaned
    except ValueError:
        return cleaned


def extract_source_numbers(json_text: str) -> set[str]:
    """Use numeric JSON values, not digits embedded in keys or free-text fields."""
    data = json.loads(json_text)
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, bool) or item is None:
            return
        if isinstance(item, (int, float)):
            if math.isfinite(item):
                found.add(_normalize(str(item)))
        elif isinstance(item, str):
            # Preserve genuinely numeric strings, including CSV-derived percentages.
            if _NUMBER_RE.fullmatch(item.strip()):
                found.add(_normalize(item.strip()))
        elif isinstance(item, dict):
            for value in item.values():
                visit(value)
        elif isinstance(item, list):
            for value in item:
                visit(value)

    visit(data)
    return found


def _close_to_any(token: str, source_numbers: set[str]) -> bool:
    try:
        target = float(_normalize(token))
    except ValueError:
        return _normalize(token) in source_numbers
    for candidate in source_numbers:
        try:
            value = float(candidate)
        except ValueError:
            continue
        if abs(value - target) <= _ABS_TOLERANCE:
            return True
        # "55%" may restate a source rate of 0.55; "55 applicants" may not.
        if token.endswith("%") and abs(value * 100 - target) <= _PCT_TOLERANCE:
            return True
        # Support a fraction from a source recorded on the 0-100 scale.
        if not token.endswith("%") and 0 <= target <= 1:
            if abs(value - target * 100) <= _PCT_TOLERANCE:
                return True
    return False


def is_grounded(text: str, source_numbers: set[str]) -> bool:
    """Check that numeric tokens are supported by numeric evidence."""
    for token in _NUMBER_RE.findall(text):
        if not _close_to_any(token, source_numbers):
            _log.warning("Grounding rejected unsupported numeric claim")
            return False
    return True


def is_explanation_grounded(text: str, scorecard: dict) -> bool:
    """Check numbers and high-impact contradictions; not a full semantics proof."""
    evidence = extract_source_numbers(json.dumps(scorecard, default=str))
    if not is_grounded(text, evidence):
        return False
    if _CLEARANCE_RE.search(text):
        _log.warning("Grounding rejected a regulatory/deployment clearance claim")
        return False

    groups = scorecard.get("groups") or []
    labels = {
        group["group_label"]
        for group in groups
        if isinstance(group, dict) and isinstance(group.get("group_label"), str)
    }
    if any(label not in labels for label in _GROUP_RE.findall(text)):
        _log.warning("Grounding rejected an unsupported group label")
        return False

    # Reject direct reversals of a named group's explicit flagged state.
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("group_label"), str):
            continue
        label = re.escape(group["group_label"])
        if group.get("flagged") and re.search(
            rf"{label}[^.!?]{{0,65}}\b(?:was|is|were) not flagged\b",
            text,
            re.IGNORECASE,
        ):
            _log.warning("Grounding rejected a reversed group flag")
            return False
        if not group.get("flagged") and re.search(
            rf"{label}[^.!?]{{0,65}}\b(?:was|is|were) flagged\b",
            text,
            re.IGNORECASE,
        ):
            _log.warning("Grounding rejected a fabricated group flag")
            return False

    flagged = any(group.get("flagged") for group in groups if isinstance(group, dict))
    if flagged and _NO_FLAG_RE.search(text):
        _log.warning("Grounding rejected a contradictory no-flags claim")
        return False
    incomplete = scorecard.get("composite_score") is None or not groups
    if incomplete and _PASS_RE.search(text):
        _log.warning("Grounding rejected an unsupported positive verdict")
        return False
    if scorecard.get("measured_mitigation") is None:
        if _MEASURED_RE.search(text) and not _NOT_MEASURED_RE.search(text):
            _log.warning("Grounding rejected an unmeasured mitigation claim")
            return False
    if scorecard.get("has_labels") is False and _EO_COMPUTED_RE.search(text):
        _log.warning("Grounding rejected label-dependent metric claim")
        return False
    return True
