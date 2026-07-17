from __future__ import annotations

import io

import pytest
from fastapi import UploadFile

from ethiclens_api.agent.csv_ingest import CsvIngestError, read_predictions_csv


def _upload(content: str) -> UploadFile:
    return UploadFile(io.BytesIO(content.encode()), filename="data.csv")


async def test_read_predictions_csv_parses_valid_csv() -> None:
    csv = "race,flag\nA,1\nB,0\n"
    df = await read_predictions_csv(_upload(csv), max_mb=5, max_rows=50_000)
    assert list(df.columns) == ["race", "flag"]
    assert len(df) == 2


async def test_read_predictions_csv_rejects_oversized_upload() -> None:
    # ~2MB of content against a 1MB cap, well under any row-count limit, so the
    # rejection is attributable to size alone, not row count.
    rows = [f"a,{'x' * 20_000}" for _ in range(100)]
    csv = "race,junk\n" + "\n".join(rows)
    assert len(csv.encode()) > 1024 * 1024

    with pytest.raises(CsvIngestError) as exc_info:
        await read_predictions_csv(_upload(csv), max_mb=1, max_rows=50_000)
    assert exc_info.value.status_code == 400
    assert "1MB" in exc_info.value.detail


async def test_read_predictions_csv_rejects_too_many_rows() -> None:
    csv = "race,flag\n" + "\n".join(f"A,{i % 2}" for i in range(10))
    with pytest.raises(CsvIngestError) as exc_info:
        await read_predictions_csv(_upload(csv), max_mb=5, max_rows=5)
    assert exc_info.value.status_code == 400
    assert "10 rows" in exc_info.value.detail


async def test_read_predictions_csv_rejects_too_many_columns() -> None:
    # Every column name is sent verbatim into the Stage 1 prompt; this bounds that.
    columns = [f"col{i}" for i in range(10)]
    csv = ",".join(columns) + "\n" + ",".join("1" for _ in columns) + "\n"
    with pytest.raises(CsvIngestError) as exc_info:
        await read_predictions_csv(_upload(csv), max_mb=5, max_rows=50_000, max_columns=5)
    assert exc_info.value.status_code == 400
    assert "10 columns" in exc_info.value.detail


async def test_read_predictions_csv_default_column_cap_allows_normal_csvs() -> None:
    csv = "race,flag,score\nA,1,0.5\n"
    df = await read_predictions_csv(_upload(csv), max_mb=5, max_rows=50_000)
    assert list(df.columns) == ["race", "flag", "score"]


async def test_read_predictions_csv_rejects_empty_csv() -> None:
    with pytest.raises(CsvIngestError):
        await read_predictions_csv(_upload(""), max_mb=5, max_rows=50_000)


async def test_read_predictions_csv_rejects_unparseable_csv() -> None:
    # Mismatched column counts across rows is a pandas ParserError, not a Python crash.
    csv = "a,b,c\n1,2\n3,4,5,6\n"
    with pytest.raises(CsvIngestError):
        await read_predictions_csv(_upload(csv), max_mb=5, max_rows=50_000)
