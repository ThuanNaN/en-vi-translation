"""BackendClient implementation for vLLM via the OpenAI-compatible chat API."""
from __future__ import annotations

import json
import urllib.request


class VLLMBackendClient:
    def __init__(self, url: str, src_lang: str | None = None, tgt_lang: str | None = None) -> None:
        self._endpoint = f"http://{url}/v1/chat/completions"
        self._src = src_lang or "the source language"
        self._tgt = tgt_lang or "the target language"

    def translate(self, text: str, model_name: str) -> str:
        payload = json.dumps({
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"Translate from {self._src} to {self._tgt}. "
                        "Output only the translation, no explanations."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "max_tokens": 512,
            "temperature": 0.0,
            # Disable Qwen3-family thinking/reasoning so the model does not emit
            # <think>...</think> tokens before the translation.
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
