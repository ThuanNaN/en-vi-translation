"""Smoke test: translate one sentence each way through Triton over HTTP.

    pip install -e '.[client]'
    python scripts/smoke_test.py --url localhost:8000
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import tritonclient.http as httpclient


def translate(client: httpclient.InferenceServerClient, model_name: str, text: str) -> str:
    inp = httpclient.InferInput("INPUT_TEXT", [1, 1], "BYTES")
    inp.set_data_from_numpy(np.array([[text]], dtype=object))
    requested = httpclient.InferRequestedOutput("OUTPUT_TEXT")
    result = client.infer(model_name=model_name, inputs=[inp], outputs=[requested])
    value = result.as_numpy("OUTPUT_TEXT").reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="localhost:8000", help="Triton HTTP endpoint")
    args = parser.parse_args()

    client = httpclient.InferenceServerClient(url=args.url)
    if not client.is_server_ready():
        print(f"Triton at {args.url} is not ready.", file=sys.stderr)
        raise SystemExit(1)

    samples = [
        ("translator_en_vi", "Hello, how are you today?"),
        ("translator_vi_en", "Xin chào, hôm nay bạn thế nào?"),
    ]
    for model_name, text in samples:
        if not client.is_model_ready(model_name):
            print(f"[skip] model {model_name} is not ready", file=sys.stderr)
            continue
        print(f"[{model_name}] {text!r} -> {translate(client, model_name, text)!r}")


if __name__ == "__main__":
    main()
