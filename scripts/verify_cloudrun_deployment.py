"""Verify that Cloud Run is running THIS commit with model ingestion disabled.

Called after deployment; no cloud SDK or external Python dependencies.
Checks the live service rather than relying on a green GitHub Actions job.
"""

from __future__ import annotations

import json
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def validate_health(body: object, expected_sha: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
        raise ValueError("expected SHA must be a 40-digit lowercase Git commit")
    if not isinstance(body, dict):
        raise ValueError("health endpoint must return an object")
    if body.get("status") != "ok" or body.get("service") != "ethiclens-api":
        raise ValueError("health endpoint returned an unexpected service or status")
    if body.get("revision") != expected_sha:
        raise ValueError(
            "deployed Git revision mismatch: expected "
            f"{expected_sha}, found {body.get('revision')!r}"
        )


def validate_openapi(body: object) -> None:
    if not isinstance(body, dict) or not isinstance(body.get("paths"), dict):
        raise ValueError("OpenAPI response does not contain a paths mapping")
    if any(path.startswith("/api/models") for path in body["paths"]):
        raise ValueError("unsafe model-file routes are exposed in the public API")


def _json_get(url: str) -> object:
    with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=25) as r:
        if r.status != 200:
            raise ValueError(f"{url} returned HTTP {r.status}")
        return json.loads(r.read().decode("utf-8"))


def verify_live(url: str, sha: str) -> None:
    base = url.rstrip("/")
    if not base.startswith("https://"):
        raise ValueError("Deployment smoke checks require an HTTPS service URL")

    # A newly deployed Cloud Run revision may be cold; retry short transient failures.
    for attempt in range(4):
        try:
            validate_health(_json_get(f"{base}/health"), sha)
            validate_openapi(_json_get(f"{base}/openapi.json"))
            break
        except (HTTPError, URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

    try:
        with urlopen(f"{base}/api/models/upload", timeout=25) as response:
            code = response.status
    except HTTPError as exc:
        code = exc.code
    if code != 404:
        raise ValueError(f"unsafe model upload endpoint returned {code}, expected 404")
    print(f"Verified live revision {sha} and model-upload lockdown at {base}")


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: verify_cloudrun_deployment.py <https-url> <40-char-sha>", file=sys.stderr)
        return 2
    try:
        verify_live(sys.argv[1], sys.argv[2])
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"Deployment verification FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
