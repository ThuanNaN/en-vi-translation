"""Thin wrapper around tritonclient.http for calling translation models on Triton."""
from __future__ import annotations

import time
import numpy as np
import tritonclient.http as tritonhttp
from envit5.core.metrics import triton_infer_duration_seconds
from envit5.core.settings import get_settings


def translate_via_triton(text: str, model_name: str) -> str:
    """Send one text string to a Triton model and return the translated string."""
    settings = get_settings()
    client = tritonhttp.InferenceServerClient(url=settings.triton_http_url)

    inp = tritonhttp.InferInput("INPUT_TEXT", [1, 1], "BYTES")
    inp.set_data_from_numpy(np.array([[text]], dtype=object))

    outputs = [tritonhttp.InferRequestedOutput("OUTPUT_TEXT")]

    start = time.perf_counter()
    result = client.infer(model_name=model_name, inputs=[inp], outputs=outputs)
    triton_infer_duration_seconds.labels(model=model_name).observe(
        time.perf_counter() - start
    )

    value = result.as_numpy("OUTPUT_TEXT").reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
