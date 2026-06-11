"""Factory that maps a BackendConfig to the right BackendClient implementation."""
from __future__ import annotations

from polyglot_gateway.core.settings import BackendConfig
from polyglot_gateway.inference.base import BackendClient
from polyglot_gateway.inference.custom import CustomBackendClient
from polyglot_gateway.inference.triton import TritonBackendClient
from polyglot_gateway.inference.vllm import VLLMBackendClient


def make_backend_client(config: BackendConfig) -> BackendClient:
    if config.type == "triton":
        return TritonBackendClient(url=config.url)
    if config.type == "vllm":
        return VLLMBackendClient(url=config.url)
    if config.type == "custom":
        return CustomBackendClient(url=config.url)
    raise ValueError(f"Unknown backend type: {config.type!r}")
