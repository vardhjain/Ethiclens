"""Train (and verify) the golden-reference biased model — the project's oracle.

This script produces ``models/golden/calibrated_bias_model.pkl``: a deliberately
biased classifier whose Disparate Impact for ``race:Black`` lands at ≈ 0.55.

The value is an **empirically-pinned regression anchor**, reproduced under a
frozen seed, dataset and code version — *not* an analytically-known constant (a
trained classifier's selection-rate ratio has no closed form). The CI band
``[0.53, 0.57]`` absorbs platform floating-point non-determinism. If a future
change makes the engine measure a different DI, ``--verify`` fails the build.

    python -m ml.training.train_calibrated_bias_model            # train + save
    python -m ml.training.train_calibrated_bias_model --verify   # rebuild + assert band
    python -m ml.training.train_calibrated_bias_model --calibrate # sweep disadvantage→DI
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import sklearn
from sklearn.linear_model import LogisticRegression

from fairness_core import run_audit, set_global_seed
from fairness_core.audit import AttributeSpec
from fairness_core.datasets import make_biased_lending_dataset
from fairness_core.types import MetricName

# --- Frozen golden configuration -------------------------------------------
GOLDEN_SEED = 42
GOLDEN_N = 8000
GOLDEN_DISADVANTAGE = 0.46  # calibrated so race:Black DI lands ≈ 0.55
FEATURES = ["income", "credit_score", "debt_ratio"]
TARGET_DI = 0.55
DI_BAND = (0.53, 0.57)

_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = _ROOT / "models" / "golden" / "calibrated_bias_model.pkl"
META_PATH = _ROOT / "models" / "golden" / "calibrated_bias_model.meta.json"


def train_model(
    disadvantage: float = GOLDEN_DISADVANTAGE,
) -> tuple[LogisticRegression, object, str]:
    """Return ``(model, dataframe, target_column)`` for a frozen biased dataset."""
    set_global_seed(GOLDEN_SEED)
    df, target = make_biased_lending_dataset(GOLDEN_N, GOLDEN_SEED, disadvantage)
    model = LogisticRegression(max_iter=2000, random_state=GOLDEN_SEED)
    model.fit(df[FEATURES], df[target])
    return model, df, target


def measure_black_di(model: LogisticRegression, df: object, target: str) -> float:
    """Audit the model and return the ``race:Black`` Disparate Impact point estimate."""
    result = run_audit(
        model,
        df,
        [AttributeSpec("race")],
        target=target,
        feature_columns=FEATURES,
        compute_ci=False,
        seed=GOLDEN_SEED,
    )
    black = next(g for g in result.groups if g.group_label == "race:Black")
    di = black.metric(MetricName.DISPARATE_IMPACT)
    if di is None or di.value is None:
        raise RuntimeError("race:Black Disparate Impact could not be computed.")
    return di.value


def save_golden() -> float:
    """Train, persist the model + metadata, and return the achieved DI."""
    model, df, target = train_model()
    di = measure_black_di(model, df, target)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    META_PATH.write_text(
        json.dumps(
            {
                "name": "calibrated_bias_model",
                "model": "LogisticRegression",
                "sklearn_version": sklearn.__version__,
                "seed": GOLDEN_SEED,
                "n": GOLDEN_N,
                "disadvantage": GOLDEN_DISADVANTAGE,
                "features": FEATURES,
                "protected_attribute": "race",
                "measured_race_black_di": round(di, 4),
                "di_band": list(DI_BAND),
                "note": "Empirically pinned under a frozen seed/data/code version; not analytic.",
            },
            indent=2,
        )
    )
    print(f"Saved golden model -> {MODEL_PATH}")
    print(f"race:Black Disparate Impact = {di:.4f}  (target {TARGET_DI}, band {DI_BAND})")
    return di


def verify_golden() -> int:
    """Rebuild the golden model and assert its DI is within the pinned band."""
    model, df, target = train_model()
    di = measure_black_di(model, df, target)
    low, high = DI_BAND
    ok = low <= di <= high
    status = "PASS" if ok else "FAIL"
    print(f"[golden-audit] race:Black DI = {di:.4f}  band {DI_BAND}  -> {status}")
    if not ok:
        print(f"::error:: golden DI {di:.4f} drifted outside {DI_BAND}", file=sys.stderr)
        return 1
    return 0


def calibrate() -> int:
    """Sweep the disadvantage parameter and print the resulting DI (a tuning aid)."""
    print(f"{'disadvantage':>12} | {'race:Black DI':>14}")
    print("-" * 30)
    for d in [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60]:
        model, df, target = train_model(disadvantage=d)
        di = measure_black_di(model, df, target)
        marker = "  <-- target" if DI_BAND[0] <= di <= DI_BAND[1] else ""
        print(f"{d:>12.2f} | {di:>14.4f}{marker}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/verify the golden bias oracle.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify", action="store_true", help="rebuild and assert the DI band")
    group.add_argument("--calibrate", action="store_true", help="sweep disadvantage→DI")
    args = parser.parse_args(argv)
    if args.verify:
        return verify_golden()
    if args.calibrate:
        return calibrate()
    save_golden()
    return 0


if __name__ == "__main__":
    sys.exit(main())
