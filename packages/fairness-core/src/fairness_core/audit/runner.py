"""The end-to-end audit pipeline (FR-003).

``run_audit`` takes a model (or a plain predict callable) and a labelled
dataset, and produces an :class:`AuditResult`: per-group Disparate Impact, SPD
and (when ground-truth labels are present) Equalized Odds, each with a bootstrap
confidence interval, plus an aggregate Composite Bias Score.

Key design decision (the methodological fix): **error-based metrics are only
computed when ``target`` labels are supplied.** On a label-free cohort,
Equalized Odds is reported as ``INSUFFICIENT_DATA`` rather than fabricated.
A group is flagged only when its confidence interval supports a threshold
violation and both comparator groups meet minimum sample floors. Unavailable
metrics never count as evidence of fairness or contribute to the composite.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    unprivileged_values: Sequence[object] | None = None


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
    if composite is not None:
        composite_reason = None
    elif not groups:
        composite_reason = "No group comparisons are available for a composite score."
    elif not has_labels:
        composite_reason = (
            "The three-metric composite requires Equalized Odds, which cannot be "
            "evaluated without ground-truth outcome labels."
        )
    else:
        composite_reason = (
            "The three-metric composite requires reliable Disparate Impact, "
            "Statistical Parity Difference and Equalized Odds estimates for EVERY "
            "group comparison; at least one metric has insufficient evidence."
        )
    return AuditResult(
        composite_score=composite,
        composite_band=band,
        min_di=min_di,
        groups=groups,
        has_labels=has_labels,
        metadata={
            "n_rows": len(data),
            "features": features,
            "threshold": threshold,
            "insufficient_groups": [
                g.group_label for g in groups if any(not m.computable for m in g.metrics.values())
            ],
            "composite_complete": composite is not None,
            "composite_reason": composite_reason,
        },
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

    # BOTH groups must meet the rate floor. A tiny privileged/reference group
    # can make the denominator and all comparisons just as unstable.
    rates_defined = bool(np.isfinite(p.selection_rate) and np.isfinite(u.selection_rate))
    enough_rates = has_sufficient_data(p.n) and has_sufficient_data(u.n)
    rate_reason = (
        "Selection-rate comparisons require at least 100 members in EACH group."
        if not enough_rates
        else "A comparison group has no observed selection rate."
    )

    # --- Disparate Impact --------------------------------------------------
    di_val: float | None = None
    if rates_defined and p.selection_rate > 0:
        di_val = calculate_disparate_impact(p.selection_rate, u.selection_rate)
        if not np.isfinite(di_val):
            di_val = None

    di_ci = None
    p_value = None
    if di_val is not None and enough_rates:
        if compute_ci:
            di_ci = disparate_impact_ci(
                sensitive, y_pred, priv_value, unpriv_value, n_boot=n_boot, seed=seed
            )
        _, p_value = two_proportion_ztest(
            round(u.selection_rate * u.n),
            u.n,
            round(p.selection_rate * p.n),
            p.n,
        )
    di_ci_valid = di_ci is not None and np.isfinite(di_ci.low) and np.isfinite(di_ci.high)
    # A numerical ratio crossing 0.80 is NOT proof of a statistically
    # supported violation. Never fall back to that point estimate for a flag.
    flagged_di = bool(
        enough_rates and di_ci_valid and di_ci is not None and di_ci.high < di_threshold
    )
    di_ok = enough_rates and di_val is not None
    metrics[MetricName.DISPARATE_IMPACT.value] = MetricResult(
        name=MetricName.DISPARATE_IMPACT.value,
        value=di_val,
        group_label=label,
        classification=(
            classify_disparate_impact(di_val, di_threshold)
            if di_ok and di_val is not None
            else Classification.INSUFFICIENT_DATA.value
        ),
        ci=di_ci if di_ci_valid else None,
        p_value=p_value,
        n=u.n,
        privileged_rate=p.selection_rate if rates_defined else None,
        unprivileged_rate=u.selection_rate if rates_defined else None,
        details={
            "n_privileged": p.n,
            "n_unprivileged": u.n,
            **(
                {
                    "reason": rate_reason
                    if not enough_rates or not rates_defined
                    else "Disparate Impact is undefined when the reference selection rate is zero."
                }
                if not di_ok
                else {}
            ),
        },
    )

    # --- Statistical Parity Difference ------------------------------------
    spd_val = calculate_spd(p.selection_rate, u.selection_rate) if rates_defined else None
    metrics[MetricName.SPD.value] = MetricResult(
        name=MetricName.SPD.value,
        value=spd_val,
        group_label=label,
        classification=(
            classify_spd(spd_val)
            if enough_rates and spd_val is not None
            else Classification.INSUFFICIENT_DATA.value
        ),
        n=u.n,
        privileged_rate=p.selection_rate if rates_defined else None,
        unprivileged_rate=u.selection_rate if rates_defined else None,
        details={"reason": rate_reason} if not enough_rates or not rates_defined else {},
    )

    # --- Equalized Odds (requires labels AND enough labels in BOTH groups) ---
    flagged_eo = False
    if (
        y_true is not None
        and p.tpr is not None
        and p.fpr is not None
        and u.tpr is not None
        and u.fpr is not None
    ):
        eo_val = calculate_eo(p.tpr, p.fpr, u.tpr, u.fpr)
        p_pos = int(np.count_nonzero((sensitive == priv_value) & (y_true == 1)))
        u_pos = int(np.count_nonzero((sensitive == unpriv_value) & (y_true == 1)))
        enough_eo = has_sufficient_data(
            p.n, p_pos, p.n - p_pos, error_metric=True
        ) and has_sufficient_data(u.n, u_pos, u.n - u_pos, error_metric=True)
        eo_ci = (
            equalized_odds_ci(
                sensitive, y_pred, y_true, priv_value, unpriv_value, n_boot=n_boot, seed=seed
            )
            if compute_ci and enough_eo
            else None
        )
        eo_ci_valid = eo_ci is not None and np.isfinite(eo_ci.low) and np.isfinite(eo_ci.high)
        flagged_eo = bool(
            enough_eo and eo_ci_valid and eo_ci is not None and eo_ci.low > EO_THRESHOLD
        )
        metrics[MetricName.EQUALIZED_ODDS.value] = MetricResult(
            name=MetricName.EQUALIZED_ODDS.value,
            value=eo_val,
            group_label=label,
            classification=(
                classify_equalized_odds(eo_val)
                if enough_eo
                else Classification.INSUFFICIENT_DATA.value
            ),
            ci=eo_ci if eo_ci_valid else None,
            n=u.n,
            details={
                "tpr_priv": p.tpr,
                "tpr_unpriv": u.tpr,
                "fpr_priv": p.fpr,
                "fpr_unpriv": u.fpr,
                "n_positive_privileged": p_pos,
                "n_negative_privileged": p.n - p_pos,
                "n_positive_unprivileged": u_pos,
                "n_negative_unprivileged": u.n - u_pos,
                **(
                    {
                        "reason": "Equalized Odds requires at least 30 positive and 30 negative "
                        "labels in EACH comparison group."
                    }
                    if not enough_eo
                    else {}
                ),
            },
        )
    else:
        metrics[MetricName.EQUALIZED_ODDS.value] = MetricResult(
            name=MetricName.EQUALIZED_ODDS.value,
            value=None,
            group_label=label,
            n=u.n,
            classification=Classification.INSUFFICIENT_DATA.value,
            details={
                "reason": (
                    "Equalized Odds requires ground-truth labels."
                    if y_true is None
                    else "Both groups must have positive and negative ground-truth labels."
                )
            },
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


def _aggregate(groups: list[GroupAuditResult]) -> tuple[float | None, str | None, float | None]:
    """Keep the valid DI headline independent from the full three-metric composite.

    A minimum DI is publishable if *every* comparison has a sufficient DI
    estimate, even if EO is unavailable (for example, no ground truth).
    Do not silently select only the well-powered groups: that could conceal
    an unassessable disadvantaged group. The composite still requires all
    three metrics on every comparison.
    """
    if not groups:
        return None, None, None

    di_pairs: list[tuple[GroupAuditResult, float]] = []
    for group in groups:
        di = group.metric(MetricName.DISPARATE_IMPACT)
        if di is None or not di.computable or di.value is None:
            return None, None, None
        di_pairs.append((group, di.value))

    worst_g, min_di = min(di_pairs, key=lambda pair: pair[1])
    if any(
        not all(
            metric is not None and metric.computable
            for metric in (
                group.metric(MetricName.SPD),
                group.metric(MetricName.EQUALIZED_ODDS),
            )
        )
        for group in groups
    ):
        return None, None, min_di

    spd = worst_g.metric(MetricName.SPD)
    eo = worst_g.metric(MetricName.EQUALIZED_ODDS)
    assert spd is not None and spd.value is not None
    assert eo is not None and eo.value is not None
    composite = compute_composite_bias_score(min_di, spd.value, eo.value)
    return composite, classify_composite_score(composite), min_di
