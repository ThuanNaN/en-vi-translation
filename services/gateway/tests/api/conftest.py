"""API test fixtures — skipped automatically until polyglot_gateway.api is implemented."""

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

from tests.conftest import TEST_API_KEY

# Skip the entire api test module if the FastAPI app hasn't been written yet.
app_mod = pytest.importorskip("polyglot_gateway.api.app", reason="polyglot_gateway.api not implemented yet (Phase 3)")


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture()
def client(override_settings: None) -> Generator[TestClient, None, None]:  # pylint: disable=unused-argument
    """Sync TestClient wrapping the FastAPI app.

    Celery task submission is patched out so tests never need a running broker.
    """
    with patch("polyglot_gateway.api.app.translate_task") as mock_task:
        mock_result = MagicMock()
        mock_result.id = "test-job-id-0001"
        mock_task.delay.return_value = mock_result

        with TestClient(app_mod.app, raise_server_exceptions=True) as c:
            c.mock_task = mock_task  # expose for assertions
            yield c


@pytest.fixture()
def pending_job_id(  # pylint: disable=redefined-outer-name
    client: TestClient, auth_headers: dict
) -> str:
    """Submit a translation job and return its job_id."""
    resp = client.post(
        "/translate",
        json={"text": "Hello world", "source": "en", "target": "vi"},
        headers=auth_headers,
    )
    assert resp.status_code == 202
    return resp.json()["job_id"]
