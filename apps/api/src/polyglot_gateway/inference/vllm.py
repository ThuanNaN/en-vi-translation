"""BackendClient implementation for vLLM via the OpenAI-compatible chat API."""
from __future__ import annotations

import json
import urllib.request

_LANG_NAMES: dict[str, str] = {
    "en": "English",
    "vi": "Vietnamese",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "th": "Thai",
}


class VLLMBackendClient:
    def __init__(self, url: str) -> None:
        self._endpoint = f"http://{url}/v1/chat/completions"

    def translate(self, text: str, model_name: str, src: str = "", tgt: str = "") -> str:
        src_lang = _LANG_NAMES.get(src, src)
        tgt_lang = _LANG_NAMES.get(tgt, tgt)
        payload = json.dumps({
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"Translate from {src_lang} to {tgt_lang}. "
                        "Output only the translation, no explanations."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "max_tokens": 512,
            "temperature": 0.0,
            # Disable Qwen3-family thinking/reasoning tokens before the translation.
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
