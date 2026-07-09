from __future__ import annotations

from ethiclens_api.agent.grounding import extract_source_numbers, is_grounded


def test_extract_source_numbers_normalizes_percent_and_commas():
    numbers = extract_source_numbers('{"a": 0.55, "b": "1,234", "c": "12%"}')
    assert "0.55" in numbers
    assert "1234.00" in numbers
    assert "12.00" in numbers


def test_is_grounded_true_for_exact_match():
    assert is_grounded("The score is 0.55.", {"0.55"})


def test_is_grounded_true_for_percentage_vs_fraction():
    assert is_grounded("That's 55%.", {"0.55"})


def test_is_grounded_false_for_fabricated_number():
    assert not is_grounded("42% of applicants were affected.", {"0.55", "0.62"})


def test_is_grounded_true_when_text_has_no_numbers():
    assert is_grounded("No numeric claims here.", {"0.55"})
