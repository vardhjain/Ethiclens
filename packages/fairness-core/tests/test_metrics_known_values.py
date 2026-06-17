"""Known-value tests reproducing the STP unit scripts *verbatim*.

These assertions are copied straight from EthicLens STP test scripts
TS-UNIT-001 / 003 / 004. They are the contract: if any of these change, the
golden numbers a reviewer can check against the document have drifted.
"""

from __future__ import annotations

import pytest

from fairness_core.metrics import (
    calculate_disparate_impact,
    calculate_spd,
    classify_composite_score,
    classify_disparate_impact,
    classify_spd,
    compute_composite_bias_score,
)


class TestDisparateImpact:  # TS-UNIT-001
    def test_known_ratio(self) -> None:
        assert calculate_disparate_impact(0.80, 0.40) == pytest.approx(0.50)
        assert classify_disparate_impact(0.50) == "FAIL"

    def test_equality_edge_case(self) -> None:
        assert calculate_disparate_impact(0.80, 0.80) == pytest.approx(1.0)
        assert classify_disparate_impact(1.0) == "PASS"

    def test_zero_rate_edge_case(self) -> None:
        assert calculate_disparate_impact(0.80, 0.0) == pytest.approx(0.0)
        assert classify_disparate_impact(0.0) == "FAIL"

    def test_privileged_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be zero"):
            calculate_disparate_impact(0.0, 0.40)


class TestSPD:  # TS-UNIT-003
    def test_known_difference(self) -> None:
        assert calculate_spd(0.70, 0.40) == pytest.approx(-0.30, abs=1e-4)

    def test_zero_difference(self) -> None:
        assert calculate_spd(0.60, 0.60) == pytest.approx(0.0, abs=1e-10)

    def test_classification(self) -> None:
        assert classify_spd(-0.08) == "Acceptable"
        assert classify_spd(-0.15) == "Flagged"
        assert classify_spd(0.12) == "Flagged"


class TestComposite:  # TS-UNIT-004
    def test_golden_composite(self) -> None:
        # DI 0.6, SPD -0.25, EO 0.15 with default weights -> exactly 0.7150.
        assert compute_composite_bias_score(0.6, -0.25, 0.15) == pytest.approx(0.7150, abs=1e-3)

    def test_classification_bands(self) -> None:
        assert classify_composite_score(0.55) == "High Risk"
        assert classify_composite_score(0.85) == "Low Risk"

    def test_favoured_flip_is_symmetric(self) -> None:
        # A DI > 1 (favoured group) is scored the same as its mirror below 1.
        assert compute_composite_bias_score(1.25, 0.0, 0.0) == pytest.approx(
            compute_composite_bias_score(0.80, 0.0, 0.0)
        )
