"""Locust load test for the audit pipeline (NFR-PERF-002, honest smoke).

Run a modest, reproducible load against a running API and report median /
95th-percentile audit latency. This is a *smoke test* with honest extrapolation,
not a 500MB/90s production benchmark.

    locust -f infra/locust/locustfile.py --host http://localhost:8000 \
           --users 50 --spawn-rate 5 --run-time 5m
"""

from __future__ import annotations

import uuid

from locust import HttpUser, between, task


class AuditUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self) -> None:
        email = f"load_{uuid.uuid4().hex[:8]}@example.com"
        self.client.post("/api/auth/register", json={"email": email, "password": "password123"})
        resp = self.client.post(
            "/api/auth/login", data={"username": email, "password": "password123"}
        )
        self.headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}

    @task
    def run_audit(self) -> None:
        create = self.client.post(
            "/api/sessions/create",
            headers=self.headers,
            json={
                "dataset": "synthetic",
                "protected_attributes": [{"name": "race"}],
                "target": "approved",
            },
        )
        session_id = create.json()["id"]
        with self.client.post(
            f"/api/sessions/{session_id}/run",
            headers=self.headers,
            name="/api/sessions/[id]/run",
            catch_response=True,
        ) as r:
            if r.status_code != 202:
                r.failure(f"run returned {r.status_code}")
        self.client.get(
            f"/api/sessions/{session_id}/metrics",
            headers=self.headers,
            name="/api/sessions/[id]/metrics",
        )
