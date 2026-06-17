"""Canonical fairness benchmark loaders, normalised to a common shape.

Primary: **Folktables ACSIncome** (Ding et al. 2021), the field-endorsed,
reproducible replacement for UCI Adult. Secondaries: **COMPAS** (the canonical
recidivism controversy, for the impossibility-theorem demonstration) and
**German Credit** (a small, fast credit-domain analogue to the lending example).
UCI **Adult** is included as a legacy comparator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fairness_core.audit import AttributeSpec

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
_COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/"
    "compas-scores-two-years.csv"
)


@dataclass
class FairnessDataset:
    """A benchmark normalised for the EthicLens audit pipeline."""

    name: str
    frame: pd.DataFrame
    target: str
    positive_label: int
    feature_columns: list[str]
    protected: list[AttributeSpec] = field(default_factory=list)
    description: str = ""

    def __post_init__(self) -> None:
        missing = [c for c in [self.target, *self.feature_columns] if c not in self.frame.columns]
        if missing:
            raise ValueError(f"{self.name}: missing columns {missing}")


def _cache_path(name: str) -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR / f"{name}.parquet"


def load_german_credit(use_cache: bool = True) -> FairnessDataset:
    """German Credit (OpenML ``credit-g``). Protected: sex (from personal status), age."""
    cache = _cache_path("german_credit")
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
    else:
        from sklearn.datasets import fetch_openml

        raw = fetch_openml("credit-g", version=1, as_frame=True)
        df = raw.frame.copy()
        # personal_status encodes sex; map to a clean binary attribute.
        male = {"male single", "male div/sep", "male mar/wid"}
        df["sex"] = df["personal_status"].apply(lambda s: "Male" if s in male else "Female")
        df["age_group"] = (df["age"] >= 25).map({True: "Older", False: "Younger"})
        df["label"] = (df["class"] == "good").astype(int)
        if use_cache:
            df.to_parquet(cache)
    features = [c for c in df.columns if c not in {"class", "label", "sex", "age_group"}]
    return FairnessDataset(
        name="german_credit",
        frame=df,
        target="label",
        positive_label=1,
        feature_columns=[c for c in features if df[c].dtype.kind in "biufc"],
        protected=[
            AttributeSpec("sex", privileged_value="Male", unprivileged_values=["Female"]),
            AttributeSpec("age_group", privileged_value="Older", unprivileged_values=["Younger"]),
        ],
        description="1,000 German loan applications; good/bad credit risk.",
    )


def load_compas(use_cache: bool = True) -> FairnessDataset:
    """ProPublica COMPAS two-year recidivism, with the standard ProPublica filters."""
    cache = _cache_path("compas")
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
    else:
        raw = pd.read_csv(_COMPAS_URL)
        df = raw[
            (raw["days_b_screening_arrest"] <= 30)
            & (raw["days_b_screening_arrest"] >= -30)
            & (raw["is_recid"] != -1)
            & (raw["c_charge_degree"] != "O")
            & (raw["score_text"] != "N/A")
        ].copy()
        df["length_of_stay"] = (
            pd.to_datetime(df["c_jail_out"]) - pd.to_datetime(df["c_jail_in"])
        ).dt.days.clip(lower=0)
        if use_cache:
            df.to_parquet(cache)
    features = ["age", "priors_count", "length_of_stay", "juv_fel_count", "juv_misd_count"]
    return FairnessDataset(
        name="compas",
        frame=df,
        target="two_year_recid",
        positive_label=1,
        feature_columns=[c for c in features if c in df.columns],
        protected=[
            AttributeSpec(
                "race", privileged_value="Caucasian", unprivileged_values=["African-American"]
            ),
            AttributeSpec("sex", privileged_value="Male", unprivileged_values=["Female"]),
        ],
        description="ProPublica COMPAS; demonstrates the predictive-parity impossibility result.",
    )


def load_adult(use_cache: bool = True) -> FairnessDataset:
    """UCI Adult / Census Income (OpenML ``adult``). Legacy comparator."""
    cache = _cache_path("adult")
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
    else:
        from sklearn.datasets import fetch_openml

        raw = fetch_openml("adult", version=2, as_frame=True)
        df = raw.frame.copy()
        df["label"] = (df["class"].astype(str).str.contains(">50K")).astype(int)
        if use_cache:
            df.to_parquet(cache)
    features = ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"]
    return FairnessDataset(
        name="adult",
        frame=df,
        target="label",
        positive_label=1,
        feature_columns=[c for c in features if c in df.columns],
        protected=[
            AttributeSpec("sex", privileged_value="Male", unprivileged_values=["Female"]),
            AttributeSpec("race", privileged_value="White"),
        ],
        description="UCI Adult census income (>50K). Legacy comparator; prefer Folktables.",
    )


def load_folktables_acsincome(
    states: list[str] | None = None, survey_year: str = "2018", use_cache: bool = True
) -> FairnessDataset:
    """Folktables ACSIncome (Ding et al. 2021). Primary benchmark."""
    cache = _cache_path(f"acsincome_{survey_year}")
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
    else:
        from folktables import ACSDataSource, ACSIncome

        source = ACSDataSource(
            survey_year=survey_year, horizon="1-Year", survey="person", root_dir=str(_DATA_DIR)
        )
        acs = source.get_data(states=states or ["CA"], download=True)
        feats, label, _ = ACSIncome.df_to_pandas(acs)
        df = feats.copy()
        df["label"] = label.to_numpy().astype(int).ravel()
        df["SEX"] = df["SEX"].map({1: "Male", 2: "Female"}).fillna("Unknown")
        # RAC1P: 1 = White alone, 2 = Black/African American alone (ACS codes).
        df["RAC1P_label"] = df["RAC1P"].map({1: "White", 2: "Black"}).fillna("Other")
        if use_cache:
            df.to_parquet(cache)
    feature_cols = ["AGEP", "SCHL", "WKHP", "COW", "OCCP"]
    return FairnessDataset(
        name="acsincome",
        frame=df,
        target="label",
        positive_label=1,
        feature_columns=[c for c in feature_cols if c in df.columns],
        protected=[
            AttributeSpec("SEX", privileged_value="Male", unprivileged_values=["Female"]),
            AttributeSpec("RAC1P_label", privileged_value="White", unprivileged_values=["Black"]),
        ],
        description="US Census ACS income > $50k; modern, reproducible Adult replacement.",
    )


_REGISTRY = {
    "german_credit": load_german_credit,
    "compas": load_compas,
    "adult": load_adult,
    "acsincome": load_folktables_acsincome,
}


def available_datasets() -> list[str]:
    return sorted(_REGISTRY)


def load_dataset(name: str, **kwargs: object) -> FairnessDataset:
    """Load a benchmark by name. See :func:`available_datasets`."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {available_datasets()}")
    return _REGISTRY[name](**kwargs)  # type: ignore[operator]
