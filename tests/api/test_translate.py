"""Tests for POST /translate."""

from __future__ import annotations

from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_submit_en_vi(client: TestClient, auth_headers: dict) -> None:
    resp = client.post(
        "/translate",
        json={"text": "Hello, how are you?", "source": "en", "target": "vi"},
        headers=auth_headers,
    )
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert isinstance(body["job_id"], str)
    assert body["job_id"]


def test_submit_vi_en(client: TestClient, auth_headers: dict) -> None:
    resp = client.post(
        "/translate",
        json={"text": "Xin chào thế giới", "source": "vi", "target": "en"},
        headers=auth_headers,
    )
    assert resp.status_code == 202
    assert "job_id" in resp.json()


def test_submit_with_direction_field(client: TestClient, auth_headers: dict) -> None:
    """The 'direction' shorthand (e.g. 'en-vi') is an accepted alternative."""
    resp = client.post(
        "/translate",
        json={"text": "Good morning", "direction": "en-vi"},
        headers=auth_headers,
    )
    assert resp.status_code == 202
    assert "job_id" in resp.json()


def test_submit_auto_detect(client: TestClient, auth_headers: dict) -> None:
    """Omitting source/target falls back to language auto-detection."""
    resp = client.post(
        "/translate",
        json={"text": "Good evening"},
        headers=auth_headers,
    )
    # Auto-detect is supported — must not return 4xx for a clearly English sentence.
    assert resp.status_code in (202, 200)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_missing_api_key_returns_401(client: TestClient) -> None:
    resp = client.post("/translate", json={"text": "Hello", "source": "en", "target": "vi"})
    assert resp.status_code == 401


def test_invalid_api_key_returns_401_or_403(client: TestClient) -> None:
    resp = client.post(
        "/translate",
        json={"text": "Hello", "source": "en", "target": "vi"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_missing_text_returns_422(client: TestClient, auth_headers: dict) -> None:
    resp = client.post(
        "/translate",
        json={"source": "en", "target": "vi"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_empty_text_returns_422(client: TestClient, auth_headers: dict) -> None:
    resp = client.post(
        "/translate",
        json={"text": "", "source": "en", "target": "vi"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_unsupported_direction_returns_422(client: TestClient, auth_headers: dict) -> None:
    resp = client.post(
        "/translate",
        json={"text": "Bonjour", "source": "fr", "target": "vi"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_conflicting_direction_fields_returns_422(client: TestClient, auth_headers: dict) -> None:
    """Providing both 'direction' and 'source'/'target' is ambiguous."""
    resp = client.post(
        "/translate",
        json={"text": "Hello", "direction": "en-vi", "source": "vi", "target": "en"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Celery task is actually enqueued
# ---------------------------------------------------------------------------

def test_celery_task_is_dispatched(client: TestClient, auth_headers: dict) -> None:
    client.post(
        "/translate",
        json={"text": "Hello", "source": "en", "target": "vi"},
        headers=auth_headers,
    )
    client.mock_task.delay.assert_called_once()
    call_kwargs = client.mock_task.delay.call_args
    # The worker must receive the original text.
    args, kwargs = call_kwargs
    all_args = list(args) + list(kwargs.values())
    assert "Hello" in all_args
