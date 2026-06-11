"""Central configuration shared across the gateway API and Celery worker.

Every field can be overridden with a plain environment variable
(e.g. ``REDIS_URL=redis://redis:6379/0``) or via a local ``.env`` file.

Backend registry — three ways to configure (highest priority first):
  1. BACKENDS env var (JSON):
       BACKENDS='{"en-vi":{"type":"triton","url":"triton:8000","model_name":"translator_en_vi"}}'
  2. BACKENDS_CONFIG env var pointing to a YAML file (default: configs/models.yaml):
       BACKENDS_CONFIG=/path/to/models.yaml
  3. Hardcoded defaults (localhost Triton, dev only).

Supported types: "triton" (tritonclient.http), "vllm" (OpenAI-compatible API),
                 "custom" (plain HTTP /translate endpoint), "hf" (reserved).
"""

from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, PydanticBaseSettingsSource, SettingsConfigDict


class BackendConfig(BaseModel):
    type: Literal["triton", "vllm", "hf", "custom"]
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

    def backend_for(self, src: str, tgt: str, model: str | None = None) -> BackendConfig:
        """Return the BackendConfig for the given language pair and optional model variant.

        Lookup order:
          1. "src-tgt:model"  — direction-specific model override (e.g. "en-vi:llm")
          2. "src-tgt"        — direction-specific default (e.g. "en-vi")
          3. "model"          — direction-agnostic backend (e.g. "llm" handles any direction)
        """
        src, tgt = src.lower(), tgt.lower()
        model_l = model.lower() if model else None
        candidates = []
        if model_l:
            candidates.append(f"{src}-{tgt}:{model_l}")
        candidates.append(f"{src}-{tgt}")
        if model_l:
            candidates.append(model_l)
        for key in candidates:
            if key in self.backends:
                return self.backends[key]
        supported = ", ".join(self.backends)
        raise ValueError(
            f"No backend configured for {src!r}->{tgt!r}"
            + (f" model={model!r}" if model else "")
            + f". Configured: {supported}"
        )


    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        **_kwargs: object,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Load order: env vars > .env file > YAML config file > hardcoded defaults."""
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings, dotenv_settings]
        yaml_file = Path(os.environ.get("BACKENDS_CONFIG", "configs/models.yaml"))
        if yaml_file.exists():
            from pydantic_settings import YamlConfigSettingsSource  # lazy: needs pyyaml
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file))
        return tuple(sources)


@lru_cache
def get_settings() -> Settings:
    return Settings()
