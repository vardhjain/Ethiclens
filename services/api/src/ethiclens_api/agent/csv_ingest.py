"""Predictions-CSV ingestion for the agent pipeline (no model files — see LIMITATIONS.md).

Parses in memory only; nothing here writes the raw CSV to disk or the database, per the
hosted "no data persistence" guardrail. Caps size and row count so an anonymous public
upload can't be used to exhaust memory.
"""

from __future__ import annotations

import io

import pandas as pd
from fastapi import HTTPException, UploadFile, status

#: Read the upload in bounded pieces rather than a single `.read()`, so a body larger
#: than the configured cap is rejected without ever materializing more than one chunk
#: past the limit in memory.
_READ_CHUNK_BYTES = 1024 * 1024


class CsvIngestError(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(status.HTTP_400_BAD_REQUEST, detail)


async def read_predictions_csv(file: UploadFile, *, max_mb: int, max_rows: int) -> pd.DataFrame:
    max_bytes = max_mb * 1024 * 1024
    chunks: list[bytes] = []
    total_bytes = 0
    while chunk := await file.read(_READ_CHUNK_BYTES):
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise CsvIngestError(f"Upload exceeds the {max_mb}MB limit")
        chunks.append(chunk)
    raw = b"".join(chunks)

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise CsvIngestError(f"Could not parse CSV: {exc}") from exc

    if len(df) > max_rows:
        raise CsvIngestError(f"Upload has {len(df)} rows; the limit is {max_rows}")
    if df.empty or len(df.columns) == 0:
        raise CsvIngestError("CSV has no rows or columns")

    return df
