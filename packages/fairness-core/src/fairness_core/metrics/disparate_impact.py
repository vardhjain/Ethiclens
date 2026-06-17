"""Disparate Impact (the 4/5ths rule).

Disparate Impact (DI) is the ratio of the unprivileged group's positive-outcome
rate to the privileged group's rate::

    DI = P(Y_hat = 1 | unprivileged) / P(Y_hat = 1 | privileged)

A DI below 0.80 is evidence of adverse impact under the EEOC Uniform Guidelines
on Employee Selection Procedures (1978) — the "four-fifths rule".

The signature of :func:`calculate_disparate_impact` matches the EthicLens STP
test script ``TS-UNIT-001`` exactly, including the division-by-zero guard.
"""

from __future__ import annotations

from fairness_core.types import DI_THRESHOLD, Classification

__all__ = ["DI_THRESHOLD", "calculate_disparate_impact", "classify_disparate_impact"]


def calculate_disparate_impact(privileged_rate: float, unprivileged_rate: float) -> float:
    """Return the Disparate Impact ratio ``unprivileged_rate / privileged_rate``.

    Args:
        privileged_rate: Positive-outcome rate of the privileged (favoured) group.
        unprivileged_rate: Positive-outcome rate of the unprivileged group.

    Returns:
        The DI ratio. ``1.0`` means perfect parity; ``< 0.80`` fails the 4/5ths rule;
        ``0.0`` is maximum disparity (the unprivileged group is never selected).

    Raises:
        ValueError: If ``privileged_rate`` is zero (the ratio is undefined).

    Examples:
        >>> calculate_disparate_impact(0.80, 0.40)
        0.5
        >>> calculate_disparate_impact(0.80, 0.80)
        1.0
        >>> calculate_disparate_impact(0.80, 0.0)
        0.0
    """
    if privileged_rate == 0:
        raise ValueError("Privileged group selection rate cannot be zero.")
    return unprivileged_rate / privileged_rate


def classify_disparate_impact(di: float, threshold: float = DI_THRESHOLD) -> str:
    """Classify a DI value as ``PASS`` (>= threshold) or ``FAIL`` (< threshold)."""
    return Classification.PASS.value if di >= threshold else Classification.FAIL.value
