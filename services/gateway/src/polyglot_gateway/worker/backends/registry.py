"""Factory that maps a BackendConfig to the right BackendClient implementation."""
from __future__ import annotations

from polyglot_gateway.core.settings import BackendConfig
from polyglot_gateway.worker.backends.base import BackendClient
from polyglot_gateway.worker.backends.triton import TritonBackendClient
from polyglot_gateway.worker.backends.vllm import VLLMBackendClient


def make_backend_client(config: BackendConfig) -> BackendClient:
    if config.type == "triton":
        return TritonBackendClient(url=config.url)
    if config.type == "vllm":
        return VLLMBackendClient(url=config.url, src_lang=config.src_lang, tgt_lang=config.tgt_lang)
    raise ValueError(f"Unknown backend type: {config.type!r}")
