"""Central configuration shared across the gateway API and Celery worker.

Every field can be overridden with a plain environment variable
(e.g. ``REDIS_URL=redis://redis:6379/0``) or via a local ``.env`` file.

Backend registry — add a new language pair without code changes:
    BACKENDS='{"en-vi":{"type":"triton","url":"triton:8000","model_name":"translator_en_vi"}}'
Supported types: "triton" (tritonclient.http), "vllm" (OpenAI-compatible API).
"""

from __future__ import annotations
from functools import lru_cache
from typing import Annotated, Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class BackendConfig(BaseModel):
    type: Literal["triton", "vllm", "hf"]
    url: str
    model_name: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Backend registry: maps "src-tgt" direction key to serving backend ---
    backends: dict[str, BackendConfig] = Field(
        default_factory=lambda: {
            "en-vi": BackendConfig(type="triton", url="localhost:8000", model_name="translator_en_vi"),
            "vi-en": BackendConfig(type="triton", url="localhost:8000", model_name="translator_vi_en"),
        }
    )

    # --- Pretrained HuggingFace checkpoints (source for ONNX export only) ---
    hf_model_en_vi: str = "Helsinki-NLP/opus-mt-en-vi"
    hf_model_vi_en: str = "Helsinki-NLP/opus-mt-vi-en"

    # --- Generation ---
    max_new_tokens: int = 512
    num_beams: int = 1

    # --- Queue / cache ---
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    cache_ttl_seconds: int = 86400

    # --- API ---
    # NoDecode: skip pydantic-settings' JSON-decode pass so the CSV validator handles it.
    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    request_timeout_seconds: float = 30.0

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow a comma-separated string as well as a JSON list of keys."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def backend_for(self, src: str, tgt: str) -> BackendConfig:
        """Return the BackendConfig for the given language pair, or raise ValueError."""
        key = f"{src.lower()}-{tgt.lower()}"
        if key not in self.backends:
            supported = ", ".join(self.backends)
            raise ValueError(
                f"No backend configured for {src!r}->{tgt!r}. "
                f"Configured pairs: {supported}"
            )
        return self.backends[key]


@lru_cache
def get_settings() -> Settings:
    return Settings()
