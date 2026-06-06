"""Pydantic request and response models for the translation API."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator

_SUPPORTED_PAIRS = {("en", "vi"), ("vi", "en")}


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str | None = None
    target: str | None = None
    direction: str | None = None

    model_config = {"str_strip_whitespace": True}

    @model_validator(mode="after")
    def _resolve(self) -> "TranslateRequest":
        has_dir = self.direction is not None
        has_src_tgt = self.source is not None or self.target is not None

        if has_dir and has_src_tgt:
            raise ValueError("Provide either 'direction' or 'source'/'target', not both.")

        if has_dir:
            parts = self.direction.split("-")
            if len(parts) != 2 or not all(parts):
                raise ValueError("'direction' must be in the form 'en-vi' or 'vi-en'.")
            self.source, self.target = parts[0].lower(), parts[1].lower()

        if self.source is not None and self.target is not None:
            if (self.source, self.target) not in _SUPPORTED_PAIRS:
                raise ValueError(
                    f"Unsupported direction '{self.source}' -> '{self.target}'. "
                    "Only 'en' <-> 'vi' is supported."
                )

        if (self.source is None) != (self.target is None):
            raise ValueError("Provide both 'source' and 'target', or neither.")

        return self


class TranslateResponse(BaseModel):
    job_id: str


class JobResponse(BaseModel):
    job_id: str
    status: Literal["pending", "started", "done", "failed"]
    translation: str | None = None
    error: str | None = None
