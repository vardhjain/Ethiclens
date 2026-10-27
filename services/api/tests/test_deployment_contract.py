"""Offline smoke-test contract: never require a live Cloud Run service in CI."""

from __future__ import annotations

from pathlib import Path

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


def test_cloudrun_deploy_is_manual_only_and_requires_explicit_confirmation() -> None:
    """Normal code pushes must never launch cloud builds in portfolio mode."""
    repo_root = Path(__file__).resolve().parents[3]
    workflow = (repo_root / ".github/workflows/deploy-api.yml").read_text(encoding="utf-8")
    trigger_block = workflow.split("\non:\n", 1)[1].split("\npermissions:\n", 1)[0]
    assert "  workflow_dispatch:" in trigger_block
    assert "  push:" not in trigger_block
    assert "  pull_request:" not in trigger_block
    assert "  schedule:" not in trigger_block
    assert "confirm_deploy:" in trigger_block
    assert 'if [[ "$CONFIRM_DEPLOY" != "DEPLOY" ]]' in workflow
    assert "if [[ \"$GITHUB_REF\" != 'refs/heads/main' ]]" in workflow
    assert "GCP_WORKLOAD_IDENTITY_PROVIDER" in workflow
    assert "Refuse deployment of a superseded commit" in workflow
    assert "Verify deployed SHA and public security posture" in workflow
