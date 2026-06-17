"""Synthetic demographic personas (FR-002) and counterfactual probing.

In the original spec this module was the *basis* of auditing — which is the
methodological flaw EthicLens fixes. Here it has two **legitimate** roles:

1. Generating configurable synthetic cohorts (:func:`generate_profiles`), e.g.
   for stress-testing edge-case distributions or as the bias-injection oracle.
2. **Counterfactual probing** (:func:`counterfactual_probe`) on *real* records —
   an individual-fairness test that flips only the protected attribute.
"""

from __future__ import annotations

from fairness_core.profiles.counterfactual import CounterfactualResult, counterfactual_probe
from fairness_core.profiles.generator import ProfileConfig, generate_profiles, profiles_to_frame

__all__ = [
    "CounterfactualResult",
    "ProfileConfig",
    "counterfactual_probe",
    "generate_profiles",
    "profiles_to_frame",
]
