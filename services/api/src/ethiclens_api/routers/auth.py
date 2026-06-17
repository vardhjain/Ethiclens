"""Authentication endpoints (FR: RBAC foundation)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ethiclens_api.db import get_session
from ethiclens_api.models import UserAccount
from ethiclens_api.schemas import Token, UserCreate, UserOut
from ethiclens_api.security import (
    authenticate,
    create_access_token,
    get_current_user,
    hash_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(body: UserCreate, session: AsyncSession = Depends(get_session)) -> UserAccount:
    exists = await session.execute(select(UserAccount).where(UserAccount.email == body.email))
    if exists.scalar_one_or_none() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
    user = UserAccount(
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        role=body.role,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@router.post("/login", response_model=Token)
async def login(
    form: OAuth2PasswordRequestForm = Depends(), session: AsyncSession = Depends(get_session)
) -> Token:
    user = await authenticate(session, form.username, form.password)
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(str(user.id), user.role.value)
    return Token(access_token=token, role=user.role)


@router.get("/me", response_model=UserOut)
async def me(user: UserAccount = Depends(get_current_user)) -> UserAccount:
    return user
