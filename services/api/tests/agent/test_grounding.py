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


def test_is_grounded_false_when_claim_crosses_the_four_fifths_threshold():
    """0.79 and 0.80 are both already-rounded values on opposite sides of the
    four-fifths rule — that's a materially different regulatory statement, not a
    rounding difference, and must not be treated as grounded."""
    assert not is_grounded("The disparate impact is 0.79.", {"0.80"})


def test_is_grounded_true_for_reverse_percentage_vs_fraction():
    """The percentage<->fraction check must work in both directions: a claim on the
    0-1 scale against a source expressed on the 0-100 scale, not just the reverse."""
    assert is_grounded("The rate is 0.55.", {"55.00"})


def test_is_grounded_true_for_small_rounding_drift():
    # Source value 0.5663793103448276 normalizes to "0.57" (see extract_source_numbers);
    # an LLM restating it as "0.57" or with one more digit of precision ("0.566") is
    # still grounded, just not restating a different threshold verdict.
    assert is_grounded("The score is 0.57.", {"0.57"})
    assert is_grounded("The score is 0.566.", {"0.57"})


def test_is_grounded_false_logs_the_offending_token(caplog):
    with caplog.at_level("WARNING", logger="ethiclens.agent.grounding"):
        assert not is_grounded("42% of applicants were affected.", {"0.55"})
    assert any("42%" in record.message for record in caplog.records)
