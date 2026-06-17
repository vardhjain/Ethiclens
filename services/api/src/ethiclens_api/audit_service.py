"""The bridge between the API and the ``fairness_core`` engine.

Loads or trains a model, builds the dataset, runs the audit in a worker thread
(so the event loop stays responsive), and persists results. Every fairness
number comes from ``fairness_core`` — the API computes nothing itself.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ethiclens_api.db import SessionLocal
from ethiclens_api.models import (
    AuditSession,
    FairnessMetric,
    SessionStatus,
    UploadedModel,
)
from fairness_core import run_audit
from fairness_core.audit import AttributeSpec
from fairness_core.datasets import make_biased_lending_dataset
from fairness_core.types import AuditResult, MetricName

DEFAULT_FEATURES = ["income", "credit_score", "debt_ratio"]
DEFAULT_TARGET = "approved"
_METRIC_TYPES = [MetricName.DISPARATE_IMPACT, MetricName.SPD, MetricName.EQUALIZED_ODDS]


@dataclass
class _AuditInputs:
    dataset: str
    seed: int
    model_uri: str | None
    model_framework: str | None
    target: str
    feature_columns: list[str]
    specs: list[tuple[str, str | None, list[str]]]


def _build_dataframe(dataset: str, seed: int) -> tuple[pd.DataFrame, str]:
    # "synthetic" and "golden" share the synthetic lending dataset (the golden
    # model was trained on it). Real benchmarks are an extension point.
    df, target = make_biased_lending_dataset(n=8000, seed=seed)
    return df, target


def _resolve_model(inputs: _AuditInputs, df: pd.DataFrame, target: str, features: list[str]):
    if inputs.model_uri:
        from ethiclens_api.ingestion import ModelFramework, load_predictor

        framework = ModelFramework(inputs.model_framework or "sklearn")
        return load_predictor(inputs.model_uri, framework)
    # No uploaded model: train a transparent baseline so the flow is demoable.
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=2000, random_state=inputs.seed).fit(df[features], df[target])


def _compute(inputs: _AuditInputs) -> AuditResult:
    df, default_target = _build_dataframe(inputs.dataset, inputs.seed)
    target = inputs.target or default_target
    features = inputs.feature_columns or DEFAULT_FEATURES
    model = _resolve_model(inputs, df, target, features)
    specs = [AttributeSpec(n, p, u or None) for (n, p, u) in inputs.specs]
    return run_audit(model, df, specs, target=target, feature_columns=features, seed=inputs.seed)


async def _load_inputs(session: AsyncSession, session_id: UUID) -> _AuditInputs:
    obj = (
        await session.execute(
            select(AuditSession)
            .options(selectinload(AuditSession.protected_attributes))
            .where(AuditSession.id == session_id)
        )
    ).scalar_one()
    model_uri = model_fw = None
    if obj.model_id is not None:
        model = await session.get(UploadedModel, obj.model_id)
        if model is not None:
            model_uri, model_fw = model.file_uri, model.framework.value
    specs = [
        (pa.name, pa.privileged_value, list(pa.unprivileged_values or []))
        for pa in obj.protected_attributes
    ] or [("race", None, [])]
    return _AuditInputs(
        dataset=obj.dataset,
        seed=obj.seed,
        model_uri=model_uri,
        model_framework=model_fw,
        target=obj.target_column or DEFAULT_TARGET,
        feature_columns=list(obj.feature_columns or []),
        specs=specs,
    )


def _metric_rows(session_id: UUID, result: AuditResult) -> list[FairnessMetric]:
    rows: list[FairnessMetric] = []
    for group in result.groups:
        for mt in _METRIC_TYPES:
            m = group.metric(mt)
            if m is None:
                continue
            rows.append(
                FairnessMetric(
                    session_id=session_id,
                    group_label=group.group_label,
                    metric_type=mt.value,
                    value=m.value,
                    ci_low=m.ci.low if m.ci else None,
                    ci_high=m.ci.high if m.ci else None,
                    p_value=m.p_value,
                    n=m.n,
                    classification=m.classification,
                    privileged_rate=m.privileged_rate,
                    unprivileged_rate=m.unprivileged_rate,
                    details=_jsonable(m.details),
                )
            )
    return rows


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    return {k: (float(v) if isinstance(v, float) else v) for k, v in d.items()}


async def execute_audit(session_id: UUID) -> None:
    """Run and persist an audit for ``session_id`` (idempotent on metrics)."""
    async with SessionLocal() as session:
        obj = await session.get(AuditSession, session_id)
        if obj is None or obj.locked:
            return
        obj.status = SessionStatus.RUNNING
        await session.commit()

        try:
            inputs = await _load_inputs(session, session_id)
            result = await asyncio.to_thread(_compute, inputs)
        except Exception as exc:
            obj.status = SessionStatus.FAILED
            obj.error = str(exc)[:1000]
            await session.commit()
            return

        # Replace any prior metrics, then persist the new ones (NFR-DB-001).
        for old in list(
            (
                await session.execute(
                    select(FairnessMetric).where(FairnessMetric.session_id == session_id)
                )
            ).scalars()
        ):
            await session.delete(old)
        for row in _metric_rows(session_id, result):
            session.add(row)

        obj.composite_score = result.composite_score
        obj.composite_band = result.composite_band
        obj.min_di = result.min_di
        obj.status = SessionStatus.FLAGGED if result.flagged_groups else SessionStatus.COMPLETED
        obj.completed_at = datetime.now(UTC)
        await session.commit()
