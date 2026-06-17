"""Audit orchestration: turn a model + dataset into an :class:`AuditResult`."""

from __future__ import annotations

from fairness_core.audit.runner import AttributeSpec, predict_labels, run_audit

__all__ = ["AttributeSpec", "predict_labels", "run_audit"]
