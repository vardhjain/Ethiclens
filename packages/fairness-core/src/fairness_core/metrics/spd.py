"""Statistical Parity Difference (SPD).

SPD is the difference between the unprivileged and privileged positive-outcome
rates::

    SPD = P(Y_hat = 1 | unprivileged) - P(Y_hat = 1 | privileged)

It is zero at parity, negative when the unprivileged group is selected less
often, and positive when it is selected more often. Values in the open interval
(-0.1, 0.1) are treated as acceptable (per the EthicLens STP glossary).

Signatures match STP test script ``TS-UNIT-003``.
"""

from __future__ import annotations

from fairness_core.types import SPD_ACCEPTABLE, Classification

__all__ = ["SPD_ACCEPTABLE", "calculate_spd", "classify_spd"]


def calculate_spd(privileged_rate: float, unprivileged_rate: float) -> float:
    """Return ``unprivileged_rate - privileged_rate``.

    Examples:
        >>> calculate_spd(0.70, 0.40)
        -0.3
        >>> calculate_spd(0.60, 0.60)
        0.0
    """
    return unprivileged_rate - privileged_rate


def classify_spd(spd: float, threshold: float = SPD_ACCEPTABLE) -> str:
    """``Acceptable`` if ``|spd| < threshold`` (open interval) else ``Flagged``.

    Examples:
        >>> classify_spd(-0.08)
        'Acceptable'
        >>> classify_spd(-0.15)
        'Flagged'
        >>> classify_spd(0.12)
        'Flagged'
    """
    return Classification.ACCEPTABLE.value if abs(spd) < threshold else Classification.FLAGGED.value
