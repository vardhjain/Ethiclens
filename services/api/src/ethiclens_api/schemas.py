"""Pydantic request/response models (the API's public contract)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from ethiclens_api.models import ModelFramework, UserRole

_orm = ConfigDict(from_attributes=True)


# --- Auth ------------------------------------------------------------------


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: UserRole


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str = ""
    role: UserRole = UserRole.ML_ENGINEER


class UserOut(BaseModel):
    model_config = _orm
    id: UUID
    email: EmailStr
    full_name: str
    role: UserRole


# --- Models ----------------------------------------------------------------


class ModelOut(BaseModel):
    model_config = _orm
    id: UUID
    filename: str
    framework: ModelFramework
    size_bytes: int
    onnx_ready: bool
    created_at: datetime


# --- Sessions --------------------------------------------------------------


class ProtectedAttrIn(BaseModel):
    name: str
    privileged_value: str | None = None
    unprivileged_values: list[str] = Field(default_factory=list)


class SessionCreate(BaseModel):
    name: str = "Untitled audit"
    dataset: str = "synthetic"  # synthetic | golden | (extensible: compas, ...)
    model_id: UUID | None = None
    protected_attributes: list[ProtectedAttrIn] = Field(default_factory=list)
    target: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    n_profiles: int = 1000
    seed: int = 42


class SessionOut(BaseModel):
    model_config = _orm
    id: UUID
    name: str
    status: str
    dataset: str
    composite_score: float | None
    composite_band: str | None
    min_di: float | None
    parent_session_id: UUID | None
    locked: bool
    created_at: datetime
    completed_at: datetime | None


class SessionStatusOut(BaseModel):
    status: str
    progress: float
    eta_seconds: float | None = None
    error: str | None = None


class MetricOut(BaseModel):
    model_config = _orm
    group_label: str
    metric_type: str
    value: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    n: int | None
    classification: str | None


class MetricsResponse(BaseModel):
    session_id: UUID
    composite_score: float | None
    composite_band: str | None
    min_di: float | None
    has_labels: bool = True
    metrics: list[MetricOut]


# --- Mitigation ------------------------------------------------------------


class RecommendationOut(BaseModel):
    rank: int
    strategy: str
    strategy_name: str
    description: str
    estimated_di_improvement: float
    stage: str
    measured: bool


class RecommendationsResponse(BaseModel):
    flagged: bool
    recommendations: dict[str, list[RecommendationOut]]


class MitigateRequest(BaseModel):
    strategy: str = "threshold_optimizer"
    group_label: str | None = None


# --- Governance ------------------------------------------------------------


class SignOffRequest(BaseModel):
    note: str = Field("", max_length=500)
