"""Tests for GET /jobs/{job_id}."""

from __future__ import annotations

from unittest.mock import patch

from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_get_job_missing_api_key_returns_401(client: TestClient, pending_job_id: str) -> None:
    resp = client.get(f"/jobs/{pending_job_id}")
    assert resp.status_code == 401


def test_get_job_invalid_api_key_returns_401_or_403(
    client: TestClient, pending_job_id: str
) -> None:
    resp = client.get(f"/jobs/{pending_job_id}", headers={"X-API-Key": "wrong"})
    assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Job states
# ---------------------------------------------------------------------------

def test_get_pending_job(client: TestClient, auth_headers: dict, pending_job_id: str) -> None:
    # Celery marks a task STARTED once a worker picks it up (task_track_started=True).
    # Without a live broker we mock this state to distinguish "known job" from "unknown".
    with patch("envit5.api.app.AsyncResult") as mock_ar:
        mock_ar.return_value.status = "STARTED"
        mock_ar.return_value.result = None

        resp = client.get(f"/jobs/{pending_job_id}", headers=auth_headers)

    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == pending_job_id
    assert body["status"] in ("pending", "started")


def test_get_completed_job(client: TestClient, auth_headers: dict, pending_job_id: str) -> None:
    with patch("envit5.api.app.AsyncResult") as mock_ar:
        mock_ar.return_value.status = "SUCCESS"
        mock_ar.return_value.result = "Xin chào thế giới"

        resp = client.get(f"/jobs/{pending_job_id}", headers=auth_headers)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert "translation" in body
    assert body["translation"]


def test_get_failed_job(client: TestClient, auth_headers: dict, pending_job_id: str) -> None:
    with patch("envit5.api.app.AsyncResult") as mock_ar:
        mock_ar.return_value.status = "FAILURE"
        mock_ar.return_value.result = RuntimeError("Triton unavailable")

        resp = client.get(f"/jobs/{pending_job_id}", headers=auth_headers)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert "error" in body


def test_get_unknown_job_returns_404(client: TestClient, auth_headers: dict) -> None:
    with patch("envit5.api.app.AsyncResult") as mock_ar:
        mock_ar.return_value.status = "PENDING"
        mock_ar.return_value.result = None

        resp = client.get("/jobs/does-not-exist-xyz", headers=auth_headers)

    # Unknown IDs that Celery has no record of must return 404, not 200 with a pending status.
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

def test_job_response_schema(client: TestClient, auth_headers: dict, pending_job_id: str) -> None:
    with patch("envit5.api.app.AsyncResult") as mock_ar:
        mock_ar.return_value.status = "STARTED"
        mock_ar.return_value.result = None

        resp = client.get(f"/jobs/{pending_job_id}", headers=auth_headers)

    body = resp.json()
    assert "job_id" in body
    assert "status" in body
    assert body["status"] in ("pending", "started", "done", "failed")
