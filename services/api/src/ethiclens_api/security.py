"""Authentication & authorisation: bcrypt passwords, JWT, role-based access."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ethiclens_api.config import get_settings
from ethiclens_api.db import get_session
from ethiclens_api.models import UserAccount, UserRole

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
_BCRYPT_MAX = 72  # bcrypt truncates beyond 72 bytes


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode()[:_BCRYPT_MAX], bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode()[:_BCRYPT_MAX], hashed.encode())
    except ValueError:
        return False


def create_access_token(subject: str, role: str) -> str:
    settings = get_settings()
    expire = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {"sub": subject, "role": role, "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_session),
) -> UserAccount:
    settings = get_settings()
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_error
    except JWTError as exc:
        raise credentials_error from exc

    user = await session.get(UserAccount, UUID(user_id))
    if user is None:
        raise credentials_error
    return user


def require_role(
    *allowed: UserRole,
) -> Callable[[UserAccount], Coroutine[Any, Any, UserAccount]]:
    async def checker(user: UserAccount = Depends(get_current_user)) -> UserAccount:
        if allowed and user.role not in allowed and user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {[r.value for r in allowed]}",
            )
        return user

    return checker


async def authenticate(session: AsyncSession, email: str, password: str) -> UserAccount | None:
    result = await session.execute(select(UserAccount).where(UserAccount.email == email))
    user = result.scalar_one_or_none()
    if user is None or not verify_password(password, user.hashed_password):
        return None
    return user
