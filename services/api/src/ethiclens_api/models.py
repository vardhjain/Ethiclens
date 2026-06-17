"""SQLAlchemy 2.0 ORM — the five STP entities centred on ``AuditSession``.

A lightweight ``UploadedModel`` registry supports the upload-then-audit flow
(FR-001 → FR-004). Cross-database column types (``Uuid``, ``JSON``, string enums)
keep the schema portable: SQLite for tests, PostgreSQL in production.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from ethiclens_api.db import Base


def _now() -> datetime:
    return datetime.now(UTC)


# --- Enumerations ----------------------------------------------------------


class UserRole(StrEnum):
    COMPLIANCE_OFFICER = "compliance_officer"
    ML_ENGINEER = "ml_engineer"
    GOVERNANCE_APPROVER = "governance_approver"
    ADMIN = "admin"


class SessionStatus(StrEnum):
    DRAFT = "DRAFT"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    FLAGGED = "FLAGGED"
    ESCALATED = "ESCALATED"
    SIGNED_OFF = "SIGNED_OFF"


class ModelFramework(StrEnum):
    SKLEARN = "sklearn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    ONNX = "onnx"


def _enum(py_enum: type[StrEnum]) -> Enum:
    # native_enum=False stores the value as VARCHAR -> portable across SQLite/PG.
    return Enum(py_enum, native_enum=False, validate_strings=True, length=32)


# --- Tables ----------------------------------------------------------------


class UserAccount(Base):
    __tablename__ = "user_account"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(255), default="")
    role: Mapped[UserRole] = mapped_column(_enum(UserRole), default=UserRole.ML_ENGINEER)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    sessions: Mapped[list[AuditSession]] = relationship(
        back_populates="owner", foreign_keys="AuditSession.owner_id"
    )


class UploadedModel(Base):
    __tablename__ = "uploaded_model"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("user_account.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    framework: Mapped[ModelFramework] = mapped_column(_enum(ModelFramework))
    file_uri: Mapped[str] = mapped_column(String(1024))
    onnx_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String(64), default="")
    onnx_ready: Mapped[bool] = mapped_column(Boolean, default=False)
    model_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditSession(Base):
    __tablename__ = "audit_session"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("user_account.id"), index=True)
    model_id: Mapped[UUID | None] = mapped_column(ForeignKey("uploaded_model.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(255), default="Untitled audit")
    status: Mapped[SessionStatus] = mapped_column(
        _enum(SessionStatus), default=SessionStatus.DRAFT, index=True
    )
    dataset: Mapped[str] = mapped_column(String(64), default="synthetic")
    target_column: Mapped[str | None] = mapped_column(String(128), nullable=True)
    feature_columns: Mapped[list] = mapped_column(JSON, default=list)
    profile_config: Mapped[dict] = mapped_column(JSON, default=dict)

    composite_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    composite_band: Mapped[str | None] = mapped_column(String(32), nullable=True)
    min_di: Mapped[float | None] = mapped_column(Float, nullable=True)

    parent_session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("audit_session.id"), nullable=True
    )
    seed: Mapped[int] = mapped_column(Integer, default=42)
    model_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    signed_off_by: Mapped[UUID | None] = mapped_column(ForeignKey("user_account.id"), nullable=True)
    signed_off_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    approver_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    locked: Mapped[bool] = mapped_column(Boolean, default=False)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    owner: Mapped[UserAccount] = relationship(back_populates="sessions", foreign_keys=[owner_id])
    protected_attributes: Mapped[list[ProtectedAttribute]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    metrics: Mapped[list[FairnessMetric]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    mitigations: Mapped[list[MitigationLog]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        foreign_keys="MitigationLog.session_id",
    )


class ProtectedAttribute(Base):
    __tablename__ = "protected_attribute"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("audit_session.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(128))
    privileged_value: Mapped[str | None] = mapped_column(String(128), nullable=True)
    unprivileged_values: Mapped[list] = mapped_column(JSON, default=list)
    group_config: Mapped[dict] = mapped_column(JSON, default=dict)

    session: Mapped[AuditSession] = relationship(back_populates="protected_attributes")


class FairnessMetric(Base):
    __tablename__ = "fairness_metric"
    __table_args__ = (
        UniqueConstraint("session_id", "group_label", "metric_type", name="uq_metric"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("audit_session.id", ondelete="CASCADE"), index=True
    )
    group_label: Mapped[str] = mapped_column(String(128))
    metric_type: Mapped[str] = mapped_column(String(64))
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    n: Mapped[int | None] = mapped_column(Integer, nullable=True)
    classification: Mapped[str | None] = mapped_column(String(32), nullable=True)
    privileged_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    unprivileged_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    session: Mapped[AuditSession] = relationship(back_populates="metrics")


class MitigationLog(Base):
    __tablename__ = "mitigation_log"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("audit_session.id", ondelete="CASCADE"), index=True
    )
    group_label: Mapped[str] = mapped_column(String(128), default="")
    strategy: Mapped[str] = mapped_column(String(64))
    strategy_name: Mapped[str] = mapped_column(String(128), default="")
    rank: Mapped[int] = mapped_column(Integer, default=0)
    description: Mapped[str] = mapped_column(Text, default="")
    estimated_impact: Mapped[float | None] = mapped_column(Float, nullable=True)
    measured: Mapped[bool] = mapped_column(Boolean, default=False)
    applied: Mapped[bool] = mapped_column(Boolean, default=False)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    result_session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("audit_session.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    session: Mapped[AuditSession] = relationship(
        back_populates="mitigations", foreign_keys=[session_id]
    )
