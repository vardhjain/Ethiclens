"""Model upload & ingestion (FR-001)."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ethiclens_api.config import get_settings
from ethiclens_api.db import get_session
from ethiclens_api.ingestion import (
    IngestionError,
    convert_to_onnx,
    sha256_file,
    validate_model_file,
)
from ethiclens_api.models import UploadedModel, UserAccount
from ethiclens_api.schemas import ModelOut
from ethiclens_api.security import get_current_user

router = APIRouter(prefix="/api/models", tags=["models"])


@router.post("/upload", response_model=ModelOut, status_code=status.HTTP_202_ACCEPTED)
async def upload_model(
    file: UploadFile,
    session: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> UploadedModel:
    settings = get_settings()
    storage = Path(settings.model_storage_dir)
    storage.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "model").suffix.lower()
    dest = storage / f"{uuid4().hex}{suffix}"
    dest.write_bytes(await file.read())

    try:
        framework = validate_model_file(dest, max_mb=settings.max_model_upload_mb)
    except IngestionError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    onnx_path = convert_to_onnx(dest, framework)  # best-effort
    model = UploadedModel(
        owner_id=user.id,
        filename=file.filename or dest.name,
        framework=framework,
        file_uri=str(dest),
        onnx_uri=str(onnx_path) if onnx_path else None,
        onnx_ready=onnx_path is not None,
        size_bytes=dest.stat().st_size,
        sha256=sha256_file(dest),
    )
    session.add(model)
    await session.commit()
    await session.refresh(model)
    return model


@router.get("", response_model=list[ModelOut])
async def list_models(
    session: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> list[UploadedModel]:
    result = await session.execute(select(UploadedModel).where(UploadedModel.owner_id == user.id))
    return list(result.scalars())


@router.get("/{model_id}", response_model=ModelOut)
async def get_model(
    model_id: UUID,
    session: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> UploadedModel:
    model = await session.get(UploadedModel, model_id)
    if model is None or model.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    return model
