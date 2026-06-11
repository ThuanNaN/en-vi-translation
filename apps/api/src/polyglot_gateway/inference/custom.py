"""BackendClient implementation for a custom FastAPI inference service.

Wire it up in configs/models.yaml:
  my-model:
    type: custom
    url: custom-service:9000
    model_name: my-custom-model
"""
from __future__ import annotations

import json
import urllib.request


class CustomBackendClient:
    def __init__(self, url: str) -> None:
        self._endpoint = f"http://{url}/translate"

    def translate(self, text: str, model_name: str, src: str = "", tgt: str = "") -> str:
        payload = json.dumps({"text": text, "src": src, "tgt": tgt, "model": model_name}).encode()
        req = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            return json.loads(resp.read())["translation"]
