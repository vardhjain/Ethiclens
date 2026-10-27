"""Offline smoke-test contract: never require a live Cloud Run service in CI."""

from __future__ import annotations

import pytest
from scripts.verify_cloudrun_deployment import validate_health, validate_openapi

_SHA = "a" * 40


def test_revision_verifier_accepts_the_exact_image_sha() -> None:
    validate_health({"status": "ok", "service": "ethiclens-api", "revision": _SHA}, _SHA)


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "ok", "service": "ethiclens-api"},
        {"status": "ok", "service": "ethiclens-api", "revision": "unknown"},
        {"status": "ok", "service": "ethiclens-api", "revision": "b" * 40},
        {"status": "ok", "service": "wrong-service", "revision": _SHA},
        {"status": "error", "service": "ethiclens-api", "revision": _SHA},
    ],
)
def test_revision_verifier_rejects_missing_stale_or_invalid_deploys(payload) -> None:
    with pytest.raises(ValueError):
        validate_health(payload, _SHA)


def test_revision_verifier_rejects_invalid_sha() -> None:
    with pytest.raises(ValueError, match="SHA"):
        validate_health({"status": "ok", "service": "ethiclens-api"}, "main")


def test_openapi_verifier_requires_model_upload_routes_to_be_absent() -> None:
    validate_openapi({"paths": {"/api/agent/run-audit": {"post": {}}}})
    with pytest.raises(ValueError, match="model-file"):
        validate_openapi({"paths": {"/api/models/upload": {"post": {}}}})
    with pytest.raises(ValueError, match="paths"):
        validate_openapi({"paths": []})
