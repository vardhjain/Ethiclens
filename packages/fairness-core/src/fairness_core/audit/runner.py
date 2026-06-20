"""The end-to-end audit pipeline (FR-003).

``run_audit`` takes a model (or a plain predict callable) and a labelled
dataset, and produces an :class:`AuditResult`: per-group Disparate Impact, SPD
and (when ground-truth labels are present) Equalized Odds, each with a bootstrap
confidence interval, plus an aggregate Composite Bias Score.

Key design decision (the methodological fix): **error-based metrics are only
computed when ``target`` labels are supplied.** On a label-free cohort,
Equalized Odds is reported as ``INSUFFICIENT_DATA`` rather than fabricated.
A group is flagged only when its DI confidence interval lies entirely below
0.80 (falling back to the point estimate when no CI is available).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from fairness_core.metrics import (
    GroupRates,
    calculate_disparate_impact,
    calculate_eo,
    calculate_spd,
    classify_composite_score,
    classify_disparate_impact,
    classify_equalized_odds,
    classify_spd,
    compute_composite_bias_score,
    compute_group_rates,
)
from fairness_core.seeds import DEFAULT_SEED
from fairness_core.stats import (
    disparate_impact_ci,
    equalized_odds_ci,
    has_sufficient_data,
    two_proportion_ztest,
)
from fairness_core.types import (
    DI_THRESHOLD,
    EO_THRESHOLD,
    AuditResult,
    Classification,
    GroupAuditResult,
    MetricName,
    MetricResult,
)


class _Model(Protocol):
    def predict(self, X: Any) -> Any: ...


@dataclass
class AttributeSpec:
    """Configuration for one protected attribute in an audit.

    If ``privileged_value`` is ``None`` it is inferred as the group with the
    highest selection rate. If ``unprivileged_values`` is empty, every other
    observed value is treated as an unprivileged group.
    """

    name: str
    privileged_value: object | None = None
    unprivileged_values: list[object] | None = None


def predict_labels(
    model: _Model | Any, X: pd.DataFrame, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(y_pred, scores)``. ``scores`` is ``None`` for hard classifiers."""
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(X))[:, 1]
        return (scores >= threshold).astype(int), scores
    if callable(getattr(model, "predict", None)):
        return (np.asarray(model.predict(X)) == 1).astype(int), None
    if callable(model):  # a bare predict function
        out = np.asarray(model(X))
        if out.ndim == 2 and out.shape[1] == 2:
            scores = out[:, 1]
            return (scores >= threshold).astype(int), scores
        return (out == 1).astype(int), None
    raise TypeError("model must have predict/predict_proba or be callable")


def _coerce_specs(protected_attrs: list[str] | list[AttributeSpec]) -> list[AttributeSpec]:
    return [a if isinstance(a, AttributeSpec) else AttributeSpec(name=a) for a in protected_attrs]


def run_audit(
    model: _Model | Any,
    data: pd.DataFrame,
    protected_attrs: list[str] | list[AttributeSpec],
    *,
    target: str | None = None,
    feature_columns: list[str] | None = None,
    threshold: float = 0.5,
    di_threshold: float = DI_THRESHOLD,
    compute_ci: bool = True,
    n_boot: int = 2000,
    seed: int = DEFAULT_SEED,
) -> AuditResult:
    """Audit ``model`` on ``data`` across each protected attribute group."""
    specs = _coerce_specs(protected_attrs)
    attr_names = [s.name for s in specs]
    drop = set(attr_names) | ({target} if target else set())
    features = feature_columns or [c for c in data.columns if c not in drop]

    X = data[features]
    y_pred, _scores = predict_labels(model, X, threshold=threshold)
    y_true = data[target].to_numpy() if target is not None else None
    has_labels = y_true is not None

    groups: list[GroupAuditResult] = []
    for spec in specs:
        groups.extend(
            _audit_attribute(spec, data, y_pred, y_true, di_threshold, compute_ci, n_boot, seed)
        )

    composite, band, min_di = _aggregate(groups)
    return AuditResult(
        composite_score=composite,
        composite_band=band,
        min_di=min_di,
        groups=groups,
        has_labels=has_labels,
        metadata={"n_rows": len(data), "features": features, "threshold": threshold},
    )


def _audit_attribute(
    spec: AttributeSpec,
    data: pd.DataFrame,
    y_pred: np.ndarray,
    y_true: np.ndarray | None,
    di_threshold: float,
    compute_ci: bool,
    n_boot: int,
    seed: int,
) -> list[GroupAuditResult]:
    sensitive = data[spec.name].to_numpy()
    values = list(pd.unique(sensitive))

    # Infer the privileged (highest selection rate) group if not specified.
    priv = spec.privileged_value
    if priv is None:
        priv = max(values, key=lambda v: compute_group_rates(sensitive, y_pred, v).selection_rate)
    unpriv_values = spec.unprivileged_values or [v for v in values if v != priv]

    priv_rates = compute_group_rates(sensitive, y_pred, priv, y_true)
    results: list[GroupAuditResult] = []
    for uv in unpriv_values:
        u_rates = compute_group_rates(sensitive, y_pred, uv, y_true)
        results.append(
            _audit_pair(
                spec.name,
                str(priv),
                str(uv),
                priv_rates,
                u_rates,
                sensitive,
                y_pred,
                y_true,
                priv,
                uv,
                di_threshold,
                compute_ci,
                n_boot,
                seed,
            )
        )
    return results


def _audit_pair(
    attribute: str,
    priv_label: str,
    unpriv_label: str,
    p: GroupRates,
    u: GroupRates,
    sensitive: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray | None,
    priv_value: object,
    unpriv_value: object,
    di_threshold: float,
    compute_ci: bool,
    n_boot: int,
    seed: int,
) -> GroupAuditResult:
    label = f"{attribute}:{unpriv_label}"
    metrics: dict[str, MetricResult] = {}

    # --- Disparate Impact --------------------------------------------------
    di_val: float | None = None
    flagged_di = False
    try:
        di_val = calculate_disparate_impact(p.selection_rate, u.selection_rate)
    except ValueError:
        di_val = None

    di_ci = None
    p_value = None
    if di_val is not None:
        enough = has_sufficient_data(u.n)
        if compute_ci and enough:
            di_ci = disparate_impact_ci(
                sensitive, y_pred, priv_value, unpriv_value, n_boot=n_boot, seed=seed
            )
        # Significance of the selection-rate gap.
        _, p_value = two_proportion_ztest(
            round(u.selection_rate * u.n),
            u.n,
            round(p.selection_rate * p.n),
            p.n,
        )
        # Flag on the CI when we have one (excludes 0.80 below), else point estimate.
        if di_ci is not None and not np.isnan(di_ci.high):
            flagged_di = di_ci.high < di_threshold
        else:
            flagged_di = di_val < di_threshold
        cls = (
            Classification.INSUFFICIENT_DATA.value
            if not enough
            else classify_disparate_impact(di_val, di_threshold)
        )
        metrics[MetricName.DISPARATE_IMPACT.value] = MetricResult(
            name=MetricName.DISPARATE_IMPACT.value,
            value=di_val,
            group_label=label,
            classification=cls,
            ci=di_ci,
            p_value=p_value,
            n=u.n,
            privileged_rate=p.selection_rate,
            unprivileged_rate=u.selection_rate,
        )

    # --- Statistical Parity Difference ------------------------------------
    spd_val = calculate_spd(p.selection_rate, u.selection_rate)
    metrics[MetricName.SPD.value] = MetricResult(
        name=MetricName.SPD.value,
        value=spd_val,
        group_label=label,
        classification=classify_spd(spd_val),
        n=u.n,
        privileged_rate=p.selection_rate,
        unprivileged_rate=u.selection_rate,
    )

    # --- Equalized Odds (requires labels) ---------------------------------
    # Error-rate bias (e.g. COMPAS) is invisible to the selection-rate DI rule, so we
    # also flag when we are statistically confident the Equalized-Odds gap is large.
    flagged_eo = False
    if (
        y_true is not None
        and p.tpr is not None
        and p.fpr is not None
        and u.tpr is not None
        and u.fpr is not None
    ):
        eo_val = calculate_eo(p.tpr, p.fpr, u.tpr, u.fpr)
        n_pos = round((u.base_rate or 0.0) * u.n)
        enough_eo = has_sufficient_data(u.n, n_pos, u.n - n_pos, error_metric=True)
        eo_ci = (
            equalized_odds_ci(
                sensitive, y_pred, y_true, priv_value, unpriv_value, n_boot=n_boot, seed=seed
            )
            if compute_ci and enough_eo
            else None
        )
        if eo_ci is not None and not np.isnan(eo_ci.low):
            flagged_eo = eo_ci.low > EO_THRESHOLD
        else:
            flagged_eo = eo_val > EO_THRESHOLD
        eo_cls = (
            Classification.INSUFFICIENT_DATA.value
            if not enough_eo
            else classify_equalized_odds(eo_val)
        )
        metrics[MetricName.EQUALIZED_ODDS.value] = MetricResult(
            name=MetricName.EQUALIZED_ODDS.value,
            value=eo_val,
            group_label=label,
            classification=eo_cls,
            ci=eo_ci,
            n=u.n,
            details={
                "tpr_priv": p.tpr,
                "tpr_unpriv": u.tpr,
                "fpr_priv": p.fpr,
                "fpr_unpriv": u.fpr,
            },
        )
    else:
        metrics[MetricName.EQUALIZED_ODDS.value] = MetricResult(
            name=MetricName.EQUALIZED_ODDS.value,
            value=None,
            group_label=label,
            n=u.n,
            classification=Classification.INSUFFICIENT_DATA.value,
            details={"reason": "Equalized Odds requires ground-truth labels."},
        )

    return GroupAuditResult(
        attribute=attribute,
        group_label=label,
        privileged_value=priv_label,
        unprivileged_value=unpriv_label,
        n_privileged=p.n,
        n_unprivileged=u.n,
        metrics=metrics,
        flagged=flagged_di or flagged_eo,
    )


def _di_value(g: GroupAuditResult) -> float | None:
    m = g.metric(MetricName.DISPARATE_IMPACT)
    return m.value if m is not None else None


def _metric_value(g: GroupAuditResult, name: MetricName, default: float = 0.0) -> float:
    m = g.metric(name)
    return m.value if m is not None and m.value is not None else default


def _aggregate(groups: list[GroupAuditResult]) -> tuple[float | None, str | None, float | None]:
    """Composite from the worst (lowest-DI) group; this drives the headline verdict."""
    scored = [(g, v) for g in groups if (v := _di_value(g)) is not None]
    if not scored:
        return None, None, None
    worst_g, di = min(scored, key=lambda pair: pair[1])
    spd = _metric_value(worst_g, MetricName.SPD)
    eo = _metric_value(worst_g, MetricName.EQUALIZED_ODDS)
    composite = compute_composite_bias_score(di, spd, eo)
    return composite, classify_composite_score(composite), di
