"""Generate the three canned predictions CSVs for the hosted agent's demo path.

Run once locally (needs network access for COMPAS/Adult on first fetch) and
commit the output — the agent serves these from
``services/api/src/ethiclens_api/agent/demo_data/`` so recruiters never need
to bring their own CSV (per the build plan: "95% of visitors should never
need to upload anything").

Each dataset is deliberately picked to exercise a different guardrail path:

- **compas**: true labels + a continuous score column -> full metric suite
  (DI/SPD/Equalized Odds) and the *measured* (not projected) mitigation
  simulation. Adverse outcome direction (decile_score=high -> flagged
  high-risk) — the direction-flip case the whole schema-inference design
  exists for.
- **adult_income**: true labels, no continuous score -> full metric suite,
  but mitigation falls back to projected-only. Favorable outcome direction —
  the deliberate contrast with COMPAS.
- **synthetic_hiring**: no true labels, no score -> only DI/SPD computable;
  the report states Equalized Odds was skipped. Reuses the project's own
  golden reference model (DI ~= 0.55 for race:Black), tying the demo back to
  the benchmark story already documented elsewhere in this repo.

    python -m ml.cli.make_agent_demo_csvs
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from fairness_core.datasets import make_biased_lending_dataset
from ml.datasets import load_adult, load_compas

_OUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "services"
    / "api"
    / "src"
    / "ethiclens_api"
    / "agent"
    / "demo_data"
)
_GOLDEN_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "golden" / "calibrated_bias_model.pkl"
)
_SEED = 42
_SAMPLE_ROWS = 1800


def _write(df: pd.DataFrame, name: str) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _OUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path} ({len(df)} rows, {len(df.columns)} cols)")


def make_compas() -> None:
    ds = load_compas(use_cache=False)  # avoid requiring pyarrow just for this one-off script
    df = ds.frame.copy()
    df["risk_score"] = (df["decile_score"] / 10.0).round(3)
    df["decision"] = (df["decile_score"] >= 5).astype(int)
    out = df[
        ["race", "sex", "priors_count", "risk_score", "decision", "two_year_recid"]
    ].dropna()
    out = out.sample(n=min(_SAMPLE_ROWS, len(out)), random_state=_SEED).reset_index(drop=True)
    _write(out, "compas")


def make_adult_income() -> None:
    ds = load_adult(use_cache=False)
    df = ds.frame.copy()
    features = ds.feature_columns
    x_train, x_test, y_train, y_test = train_test_split(
        df[features], df[ds.target], test_size=0.3, random_state=_SEED, stratify=df[ds.target]
    )
    model = LogisticRegression(max_iter=2000, random_state=_SEED)
    model.fit(x_train, y_train)
    decision = model.predict(x_test)

    out = x_test.copy()
    out["sex"] = df.loc[x_test.index, "sex"]
    out["race"] = df.loc[x_test.index, "race"]
    out["decision"] = decision
    out["actual_income_above_50k"] = y_test.to_numpy()
    out = out.sample(n=min(_SAMPLE_ROWS, len(out)), random_state=_SEED).reset_index(drop=True)
    _write(out, "adult_income")


def make_synthetic_hiring() -> None:
    model = joblib.load(_GOLDEN_MODEL_PATH)
    df, _target = make_biased_lending_dataset(n=8000, seed=_SEED, disadvantage=0.46)
    features = ["income", "credit_score", "debt_ratio"]
    decision = model.predict(df[features])

    out = df[["race", "gender", "income", "credit_score", "debt_ratio"]].copy()
    out["income"] = out["income"].round(0).astype(int)
    out["credit_score"] = out["credit_score"].round(0).astype(int)
    out["debt_ratio"] = out["debt_ratio"].round(3)
    out["decision"] = decision  # no true-label column, by design (demonstrates that path)
    rng = np.random.default_rng(_SEED)
    idx = rng.choice(len(out), size=min(_SAMPLE_ROWS, len(out)), replace=False)
    out = out.iloc[idx].reset_index(drop=True)
    _write(out, "synthetic_hiring")


if __name__ == "__main__":
    make_compas()
    make_adult_income()
    make_synthetic_hiring()
