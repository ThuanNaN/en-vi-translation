"""Shared test fixtures."""

from __future__ import annotations

import pytest
from envit5.core import settings as _settings_mod

TEST_API_KEY = "test-key-abc123"


@pytest.fixture(autouse=True)
def override_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point all services at test-safe defaults so tests never hit real Triton/Redis."""
    monkeypatch.setenv("ENVIT5_TRITON_HTTP_URL", "localhost:8000")
    monkeypatch.setenv("ENVIT5_TRITON_GRPC_URL", "localhost:8001")
    monkeypatch.setenv("ENVIT5_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("ENVIT5_CELERY_BROKER_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("ENVIT5_CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
    monkeypatch.setenv("ENVIT5_API_KEYS", TEST_API_KEY)

    # Clear the lru_cache so each test starts with a fresh Settings instance.
    _settings_mod.get_settings.cache_clear()
    yield
    _settings_mod.get_settings.cache_clear()
