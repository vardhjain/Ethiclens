# %% [markdown]
# # Why synthetic-persona auditing is broken — and the fix
#
# The original EthicLens spec proposed measuring a model's fairness by feeding it
# **fabricated demographic personas**. This notebook demonstrates, with runnable code,
# *why that cannot work* — and the two legitimate things synthetic data is good for.
#
# Open in Jupyter (`jupytext --to notebook` or VS Code) or run directly:
# `python ml/notebooks/why_synthetic_auditing_is_broken.py`.

# %%
from pathlib import Path

import numpy as np

from fairness_core.metrics.group import confusion_rates
from fairness_core.profiles import counterfactual_probe, generate_profiles
from fairness_core.seeds import set_global_seed

set_global_seed(42)
DOCS = Path(__file__).resolve().parents[2] / "docs"

# %% [markdown]
# ## 1. Synthetic personas have no ground-truth label `Y`
#
# Error-based metrics — Equalized Odds, Equal Opportunity, predictive parity, calibration —
# are defined on the **true outcome** `Y`. Generate personas and there simply is no `Y` to
# compute a true-positive rate against. The metric is *uncomputable*, not merely noisy.

# %%
profiles = generate_profiles(1000, seed=1)
print("A synthetic persona:", profiles[0])
print("Keys available:", sorted(profiles[0].keys()))
print("Is there a ground-truth outcome `Y`? ->", "approved" in profiles[0])
# -> False. Without Y, Equalized Odds / Equal Opportunity cannot be evaluated at all.

# %% [markdown]
# EthicLens makes this explicit: on a label-free cohort the engine returns
# `INSUFFICIENT_DATA` for error-based metrics rather than fabricating a value.

# %% [markdown]
# ## 2. The "selection rate" on synthetic data is an artefact of *your* sampling
#
# Disparate Impact compares positive-prediction rates across groups. On synthetic personas
# the feature distribution is whatever you chose in `config` — so the measured DI reflects
# your sampling decisions and the model's response to out-of-distribution noise, not the
# real-world disparate impact on the population the model will actually face.

# %% [markdown]
# ## 3. The impossibility theorem (Chouldechova 2017; Kleinberg et al. 2016)
#
# When two groups have **different base rates** of the outcome, a classifier *cannot*
# simultaneously equalise predictive parity (PPV) **and** the error rates (FPR/TPR).
# Below we show this on data with different base rates: as we tune a threshold to equalise
# one criterion, the other diverges.

# %%
rng = np.random.default_rng(0)
n = 4000
group = rng.choice(["A", "B"], size=n)
# Group A has a higher base rate of the positive outcome than group B.
base_rate = np.where(group == "A", 0.6, 0.3)
y_true = (rng.uniform(size=n) < base_rate).astype(int)
# A reasonably calibrated score that correlates with the true outcome.
score = np.clip(0.5 * y_true + rng.normal(0.25, 0.25, size=n), 0, 1)

thresholds = np.linspace(0.2, 0.8, 25)
ppv_gap, fpr_gap = [], []
for t in thresholds:
    y_pred = (score >= t).astype(int)
    _, fpr_a, ppv_a = confusion_rates(y_true, y_pred, group == "A")
    _, fpr_b, ppv_b = confusion_rates(y_true, y_pred, group == "B")
    ppv_gap.append(abs((ppv_a or 0) - (ppv_b or 0)))
    fpr_gap.append(abs((fpr_a or 0) - (fpr_b or 0)))

print(
    "Min |PPV gap|:",
    round(min(ppv_gap), 3),
    "at threshold",
    round(float(thresholds[int(np.argmin(ppv_gap))]), 2),
)
print(
    "Min |FPR gap|:",
    round(min(fpr_gap), 3),
    "at threshold",
    round(float(thresholds[int(np.argmin(fpr_gap))]), 2),
)
print("They are minimised at *different* thresholds — you cannot have both.")

# %%
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, ppv_gap, label="|Predictive parity gap| (PPV)", color="#2980b9")
    ax.plot(thresholds, fpr_gap, label="|Error-rate gap| (FPR)", color="#c0392b")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Absolute gap between groups")
    ax.set_title(
        "Impossibility: PPV-fairness and FPR-fairness cannot both be zero\n"
        "(groups have different base rates)"
    )
    ax.legend()
    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    fig.savefig(DOCS / "impossibility-theorem.png", dpi=150)
    print(f"Saved chart -> {DOCS / 'impossibility-theorem.png'}")
except Exception as exc:  # pragma: no cover
    print("matplotlib unavailable:", exc)

# %% [markdown]
# ## 4. The fix: real labelled data + synthetic data in its *legitimate* roles
#
# - **Audit on real, labelled benchmarks** (Folktables/ACS, COMPAS, German Credit) where `Y`
#   exists and selection rates mean something.
# - **Counterfactual probing** on *real* records: flip only the protected attribute and measure
#   the change — a genuine individual-fairness signal.
# - **Bias-injection oracle**: manufacture data with a *known* disparate impact to verify the
#   measurement engine itself (this is the golden-reference model).

# %%
import pandas as pd

# A model that reacts to the protected attribute is individually unfair; the probe catches it.
real_records = pd.DataFrame({"x": rng.uniform(size=200), "race": ["Black"] * 200})


def biased_predict(frame: pd.DataFrame) -> np.ndarray:
    return frame["x"].to_numpy() + (frame["race"] == "White").to_numpy() * 0.5


probe = counterfactual_probe(biased_predict, real_records, "race", "Black", "White")
print(f"Counterfactual flip rate when race Black->White: {probe.flip_rate:.2%}")
print("A non-zero flip rate is direct evidence of individual unfairness.")

# %% [markdown]
# **Takeaway:** synthetic personas cannot *measure* fairness, but they are invaluable for
# *testing* the measurement engine and probing individual fairness. EthicLens uses them only
# for those purposes — and audits real, labelled data for the actual fairness verdict.
