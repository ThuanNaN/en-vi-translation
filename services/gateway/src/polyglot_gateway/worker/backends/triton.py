"""BackendClient implementation that speaks the Triton HTTP inference protocol."""
from __future__ import annotations

import time
import numpy as np
import tritonclient.http as tritonhttp
from polyglot_gateway.core.metrics import triton_infer_duration_seconds


class TritonBackendClient:
    def __init__(self, url: str) -> None:
        self._url = url

    def translate(self, text: str, model_name: str) -> str:
        client = tritonhttp.InferenceServerClient(url=self._url)

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
