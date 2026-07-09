"""Predictions-CSV ingestion for the agent pipeline (no model files — see LIMITATIONS.md).

Parses in memory only; nothing here writes the raw CSV to disk or the database, per the
hosted "no data persistence" guardrail. Caps size and row count so an anonymous public
upload can't be used to exhaust memory.
"""

from __future__ import annotations

import io

import pandas as pd
from fastapi import HTTPException, UploadFile, status


class CsvIngestError(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(status.HTTP_400_BAD_REQUEST, detail)


async def read_predictions_csv(file: UploadFile, *, max_mb: int, max_rows: int) -> pd.DataFrame:
    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > max_mb:
        raise CsvIngestError(f"Upload is {size_mb:.1f}MB; the limit is {max_mb}MB")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise CsvIngestError(f"Could not parse CSV: {exc}") from exc

    if len(df) > max_rows:
        raise CsvIngestError(f"Upload has {len(df)} rows; the limit is {max_rows}")
    if df.empty or len(df.columns) == 0:
        raise CsvIngestError("CSV has no rows or columns")

    return df
