from __future__ import annotations

from ethiclens_api.agent.grounding import (
    extract_source_numbers,
    is_explanation_grounded,
    is_grounded,
)


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


_AUDIT = {
    "composite_score": None,
    "has_labels": False,
    "measured_mitigation": None,
    "groups": [
        {"group_label": "race:Black", "flagged": True, "metrics": {"di": {"value": 0.55}}},
        {"group_label": "race:White", "flagged": False},
    ],
}


def test_numeric_evidence_excludes_json_keys_and_free_text() -> None:
    values = extract_source_numbers(
        '{"method_v42": 0.55, "group": "race:group999", '
        '"reason": "42% of customers", "stored_percent": "12%"}'
    )
    assert values == {"0.55", "12.00"}
    assert not is_grounded("42% were affected.", values)


def test_unmarked_count_does_not_match_fraction_via_percent_conversion() -> None:
    assert not is_grounded("55 customers were affected.", {"0.55"})
    assert is_grounded("55% received approval.", {"0.55"})


def test_fabricated_group_label_is_rejected_without_numerical_claims() -> None:
    assert not is_explanation_grounded("race:Asian was flagged.", _AUDIT)


def test_no_flag_claim_is_rejected_if_any_group_flagged() -> None:
    assert not is_explanation_grounded("No groups were flagged.", _AUDIT)
    assert not is_explanation_grounded("All groups passed.", _AUDIT)


def test_group_specific_reversed_or_fabricated_flag_is_rejected() -> None:
    assert not is_explanation_grounded("race:Black was not flagged.", _AUDIT)
    assert not is_explanation_grounded("race:White was flagged.", _AUDIT)


def test_regulatory_clearance_and_fake_measured_mitigation_are_rejected() -> None:
    assert not is_explanation_grounded("This model is legally compliant.", _AUDIT)
    assert not is_explanation_grounded("Measured mitigation improvement was 0.55.", _AUDIT)
    assert is_explanation_grounded("The mitigation was not measured.", _AUDIT)


def test_unlabeled_eo_and_unsubstantiated_positive_verdict_are_rejected() -> None:
    assert not is_explanation_grounded("Equalized Odds was computed.", _AUDIT)
    assert not is_explanation_grounded("The audit passes.", _AUDIT)


def test_grounded_group_and_factual_disparity_are_accepted() -> None:
    assert is_explanation_grounded("race:Black was flagged; DI was 0.55.", _AUDIT)


def test_explicit_numeric_group_names_remain_labels_not_evidence() -> None:
    scorecard = {"groups": [{"group_label": "group42:cohort7", "flagged": False}]}
    numbers = extract_source_numbers(__import__("json").dumps(scorecard))
    assert numbers == set()


def test_sentence_final_fabricated_number_is_not_ignored() -> None:
    assert not is_grounded("The measured score is 0.91.", {"0.55"})
    assert not is_grounded("There were 42.", {"0.55"})
    assert is_grounded("The measured score is 0.55.", {"0.55"})
