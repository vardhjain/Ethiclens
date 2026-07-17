"""Shared slowapi Limiter instance.

A single module-level instance so main.py (middleware/exception handler
registration) and routers (the ``@limiter.limit(...)`` decorators) reference the
same limiter. In-memory storage — no Redis dependency for this; state resets on
process restart, which is fine for a per-IP abuse guard rather than a durable count.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
