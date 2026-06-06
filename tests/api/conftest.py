"""API test fixtures — skipped automatically until src/envit5/api/ is implemented."""

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# Skip the entire api test module if the FastAPI app hasn't been written yet.
app_mod = pytest.importorskip("envit5.api.app", reason="envit5.api not implemented yet (Phase 3)")

from starlette.testclient import TestClient  # noqa: E402  (after importorskip guard)

from tests.conftest import TEST_API_KEY  # noqa: E402


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture()
def client(override_settings: None) -> Generator[TestClient, None, None]:  # noqa: ARG001
    """Sync TestClient wrapping the FastAPI app.

    Celery task submission is patched out so tests never need a running broker.
    """
    with patch("envit5.api.app.translate_task") as mock_task:
        mock_result = MagicMock()
        mock_result.id = "test-job-id-0001"
        mock_task.delay.return_value = mock_result

        with TestClient(app_mod.app, raise_server_exceptions=True) as c:
            c._mock_task = mock_task  # expose for assertions
            yield c


@pytest.fixture()
def pending_job_id(client: TestClient, auth_headers: dict) -> str:
    """Submit a translation job and return its job_id."""
    resp = client.post(
        "/translate",
        json={"text": "Hello world", "source": "en", "target": "vi"},
        headers=auth_headers,
    )
    assert resp.status_code == 202
    return resp.json()["job_id"]
