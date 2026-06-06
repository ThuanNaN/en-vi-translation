"""Central configuration shared across the export script, Triton client, worker, and API.

Every field can be overridden with an environment variable prefixed ``ENVIT5_``
(e.g. ``ENVIT5_TRITON_HTTP_URL=triton:8000``) or via a local ``.env`` file.
"""

from __future__ import annotations
from functools import lru_cache
from typing import Annotated
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ENVIT5_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Triton ---
    triton_http_url: str = "localhost:8000"
    triton_grpc_url: str = "localhost:8001"
    triton_model_en_vi: str = "translator_en_vi"
    triton_model_vi_en: str = "translator_vi_en"

    # --- Pretrained HuggingFace checkpoints (source for ONNX export) ---
    hf_model_en_vi: str = "Helsinki-NLP/opus-mt-en-vi"
    hf_model_vi_en: str = "Helsinki-NLP/opus-mt-vi-en"

    # --- Generation ---
    max_new_tokens: int = 512
    num_beams: int = 1

    # --- Queue / cache (used from Phase 3 onward) ---
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    cache_ttl_seconds: int = 86400

    # --- API (Phase 3) ---
    # NoDecode: skip pydantic-settings' JSON-decode pass so the CSV validator handles it.
    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    request_timeout_seconds: float = 30.0

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow a comma-separated string as well as a JSON/list of keys."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def model_name_for(self, source: str, target: str) -> str:
        """Map a (source, target) language pair to its Triton model name."""
        pair = (source.lower(), target.lower())
        if pair == ("en", "vi"):
            return self.triton_model_en_vi
        if pair == ("vi", "en"):
            return self.triton_model_vi_en
        raise ValueError(f"Unsupported direction {source!r}->{target!r}; only en<->vi is served")


@lru_cache
def get_settings() -> Settings:
    return Settings()
