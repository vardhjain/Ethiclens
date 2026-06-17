"""The golden-reference correctness gate.

If the engine's measured Disparate Impact for the frozen golden model ever leaves
the pinned band, this test (and CI) fail. It is the project's proof that the
fairness math stays correct over time.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from ml.datasets import available_datasets, load_dataset
from ml.training.train_calibrated_bias_model import (
    DI_BAND,
    FEATURES,
    MODEL_PATH,
    measure_black_di,
    train_model,
    verify_golden,
)

from fairness_core.profiles import counterfactual_probe


def test_golden_di_in_band() -> None:
    model, df, target = train_model()
    di = measure_black_di(model, df, target)
    assert DI_BAND[0] <= di <= DI_BAND[1], f"golden DI {di:.4f} left band {DI_BAND}"
    assert di == pytest.approx(0.55, abs=0.02)


def test_verify_golden_passes() -> None:
    assert verify_golden() == 0


def test_golden_artifact_loads_and_predicts() -> None:
    if not MODEL_PATH.exists():
        pytest.skip("golden model artifact not built yet (run the training script)")
    model = joblib.load(MODEL_PATH)
    _, df, _ = train_model()
    preds = model.predict(df[FEATURES].head(10))
    assert set(np.unique(preds)).issubset({0, 1})


def test_golden_model_is_individually_race_blind() -> None:
    """The bias is *structural* (proxy), not direct: flipping race changes nothing.

    The model never sees ``race`` — only income/credit/debt — so a counterfactual
    race flip leaves every prediction unchanged (flip-rate 0). Yet the model still
    shows group-level disparate impact, because the *features* differ by race.
    This is the crux of why group and individual fairness diverge.
    """
    model, df, _ = train_model()

    def score(frame: object) -> np.ndarray:
        return model.predict_proba(frame[FEATURES])[:, 1]  # type: ignore[index]

    res = counterfactual_probe(score, df, "race", "Black", "White")
    assert res.flip_rate == pytest.approx(0.0)
    assert res.mean_score_gap == pytest.approx(0.0)


def test_dataset_registry() -> None:
    assert available_datasets() == ["acsincome", "adult", "compas", "german_credit"]
    with pytest.raises(KeyError):
        load_dataset("nonexistent")
