"""Shared test fixtures."""

from __future__ import annotations

import pytest
from polyglot_gateway.core import settings as _settings_mod

TEST_API_KEY = "test-key-abc123"


@pytest.fixture(autouse=True)
def override_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point all services at test-safe defaults so tests never hit real Triton/Redis."""
    monkeypatch.setenv(
        "BACKENDS",
        '{"en-vi":{"type":"triton","url":"localhost:8000","model_name":"translator_en_vi"},'
        '"vi-en":{"type":"triton","url":"localhost:8000","model_name":"translator_vi_en"}}',
    )
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
    monkeypatch.setenv("API_KEYS", TEST_API_KEY)

    # Clear the lru_cache so each test starts with a fresh Settings instance.
    _settings_mod.get_settings.cache_clear()
    yield
    _settings_mod.get_settings.cache_clear()
