"""BackendClient implementation for vLLM via the OpenAI-compatible API."""
from __future__ import annotations


class VLLMBackendClient:
    def __init__(self, url: str) -> None:
        self._url = url

    def translate(self, text: str, model_name: str) -> str:
        # Wire up when a vLLM serving container is available:
        #   from openai import OpenAI
        #   client = OpenAI(base_url=f"http://{self._url}/v1", api_key="ignored")
        #   resp = client.chat.completions.create(
        #       model=model_name,
        #       messages=[{"role": "user", "content": f"Translate to target language:\n{text}"}],
        #   )
        #   return resp.choices[0].message.content
        raise NotImplementedError("vLLM backend not yet wired — set BACKENDS to use triton")
