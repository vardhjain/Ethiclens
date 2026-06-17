"""Apply a mitigation and persist the re-audit as a child session (FR-009)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID

from ethiclens_api.audit_service import (
    DEFAULT_FEATURES,
    _AuditInputs,
    _build_dataframe,
    _load_inputs,
    _resolve_model,
)
from ethiclens_api.db import SessionLocal
from ethiclens_api.models import (
    AuditSession,
    FairnessMetric,
    MitigationLog,
    SessionStatus,
)
from fairness_core import mitigate_and_reaudit, run_audit
from fairness_core.audit import AttributeSpec
from fairness_core.mitigation import MitigationResult
from fairness_core.types import DI_THRESHOLD


def _compute_mitigation(
    inputs: _AuditInputs, strategy: str, group_label: str | None
) -> tuple[MitigationResult, str]:
    df, default_target = _build_dataframe(inputs.dataset, inputs.seed)
    target = inputs.target or default_target
    features = inputs.feature_columns or DEFAULT_FEATURES
    model = _resolve_model(inputs, df, target, features)
    specs = [AttributeSpec(n, p, u or None) for (n, p, u) in inputs.specs]
    audit = run_audit(model, df, specs, target=target, feature_columns=features, seed=inputs.seed)

    flagged = audit.flagged_groups
    if not flagged:
        raise ValueError("No flagged groups to mitigate.")
    group = next((g for g in flagged if g.group_label == group_label), flagged[0])
    res = mitigate_and_reaudit(
        model,
        df,
        group.attribute,
        group.privileged_value,
        group.unprivileged_value,
        target=target,
        feature_columns=features,
        strategy=strategy,
        seed=inputs.seed,
    )
    return res, group.group_label


async def apply_mitigation(parent_id: UUID, strategy: str, group_label: str | None) -> UUID:
    """Run ``strategy``, persist a child (re-audited) session, and log it. Returns child id."""
    async with SessionLocal() as session:
        parent = await session.get(AuditSession, parent_id)
        if parent is None:
            raise ValueError("Parent session not found.")
        inputs = await _load_inputs(session, parent_id)
        res, glabel = await asyncio.to_thread(_compute_mitigation, inputs, strategy, group_label)

        child = AuditSession(
            owner_id=parent.owner_id,
            model_id=parent.model_id,
            dataset=parent.dataset,
            name=f"{parent.name} (mitigated: {res.strategy})",
            seed=parent.seed,
            target_column=parent.target_column,
            feature_columns=parent.feature_columns,
            parent_session_id=parent.id,
            min_di=res.di_after,
            status=SessionStatus.COMPLETED if res.crossed_threshold else SessionStatus.FLAGGED,
            completed_at=datetime.now(UTC),
        )
        session.add(child)
        await session.flush()  # assign child.id

        ci = res.di_after_ci
        session.add(
            FairnessMetric(
                session_id=child.id,
                group_label=glabel,
                metric_type="disparate_impact",
                value=res.di_after,
                ci_low=ci.low if ci else None,
                ci_high=ci.high if ci else None,
                n=res.n_test,
                classification="PASS" if res.di_after >= DI_THRESHOLD else "FAIL",
                details={"di_before": res.di_before, "accuracy_after": res.accuracy_after},
            )
        )
        session.add(
            MitigationLog(
                session_id=parent.id,
                group_label=glabel,
                strategy=res.strategy,
                strategy_name=res.strategy,
                rank=1,
                description=f"Applied {res.strategy} ({res.stage}); DI {res.di_before:.3f} "
                f"-> {res.di_after:.3f} on held-out data.",
                estimated_impact=res.di_improvement,
                measured=True,
                applied=True,
                applied_at=datetime.now(UTC),
                result_session_id=child.id,
            )
        )
        await session.commit()
        return child.id
