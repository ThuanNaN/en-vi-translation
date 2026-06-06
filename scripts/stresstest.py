"""Stress test: hammer Triton with concurrent translation requests and print a latency report.

    pip install -e '.[client]'
    python scripts/stresstest.py --url localhost:8000 --requests 200 --concurrency 16
    python scripts/stresstest.py --direction en-vi --output report.txt
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import tritonclient.http as httpclient

SAMPLE_TEXTS: dict[str, list[str]] = {
    "translator_en_vi": [
        "Hello, how are you today?",
        "The weather is nice outside.",
        "Machine translation is improving rapidly.",
        "Neural networks have transformed natural language processing.",
        "Please translate this sentence into Vietnamese.",
        "The quick brown fox jumps over the lazy dog.",
        "Science and technology are advancing at an unprecedented pace.",
    ],
    "translator_vi_en": [
        "Xin chào, hôm nay bạn thế nào?",
        "Thời tiết hôm nay rất đẹp.",
        "Dịch máy đang được cải thiện nhanh chóng.",
        "Mạng nơ-ron đã thay đổi xử lý ngôn ngữ tự nhiên.",
        "Vui lòng dịch câu này sang tiếng Anh.",
        "Khoa học và công nghệ đang tiến bộ với tốc độ chưa từng có.",
        "Học máy là một nhánh của trí tuệ nhân tạo.",
    ],
}

MODELS = list(SAMPLE_TEXTS.keys())

_thread_local = threading.local()


def _get_client(url: str) -> httpclient.InferenceServerClient:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = httpclient.InferenceServerClient(url=url)
    return _thread_local.client


def _translate_one(url: str, model_name: str, text: str) -> str:
    client = _get_client(url)
    inp = httpclient.InferInput("INPUT_TEXT", [1, 1], "BYTES")
    inp.set_data_from_numpy(np.array([[text]], dtype=object))
    out = httpclient.InferRequestedOutput("OUTPUT_TEXT")
    result = client.infer(model_name=model_name, inputs=[inp], outputs=[out])
    value = result.as_numpy("OUTPUT_TEXT").reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)


def _worker(url: str, model_name: str, texts: list[str], idx: int) -> tuple[float, str | None]:
    text = texts[idx % len(texts)]
    t0 = time.perf_counter()
    try:
        _translate_one(url, model_name, text)
        return time.perf_counter() - t0, None
    except Exception as exc:
        return time.perf_counter() - t0, str(exc)


def run_stress(url: str, model_name: str, n_requests: int, concurrency: int) -> dict:
    texts = SAMPLE_TEXTS[model_name]
    latencies: list[float] = []
    errors: list[str] = []

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_worker, url, model_name, texts, i) for i in range(n_requests)]
        for fut in as_completed(futures):
            latency, err = fut.result()
            latencies.append(latency)
            if err:
                errors.append(err)
    total_seconds = time.perf_counter() - t_start

    return {
        "model": model_name,
        "n_requests": n_requests,
        "concurrency": concurrency,
        "total_seconds": total_seconds,
        "latencies": latencies,
        "errors": errors,
    }


def _percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def build_report(results: list[dict]) -> str:
    W = 60
    lines: list[str] = []
    lines.append("=" * W)
    lines.append("  STRESS TEST REPORT")
    lines.append("=" * W)

    for r in results:
        n = r["n_requests"]
        n_err = len(r["errors"])
        n_ok = n - n_err
        lats_ok = [l for l, e in zip(r["latencies"], [None] * n_ok + r["errors"]) if e is None]
        # simpler: collect all latencies regardless of error (error path still timed)
        lats = r["latencies"]

        lines.append(f"\nModel        : {r['model']}")
        lines.append(f"Requests     : {n}  (concurrency={r['concurrency']})")
        lines.append(f"Succeeded    : {n_ok}   Failed: {n_err}")
        lines.append(f"Duration     : {r['total_seconds']:.2f} s")
        lines.append(f"Throughput   : {n / r['total_seconds']:.2f} req/s")

        if lats:
            lines.append("Latency (s)  :")
            lines.append(f"  min  {min(lats):.3f}   mean {statistics.mean(lats):.3f}")
            lines.append(
                f"  p50  {_percentile(lats, 50):.3f}"
                f"   p95  {_percentile(lats, 95):.3f}"
                f"   p99  {_percentile(lats, 99):.3f}"
                f"   max  {max(lats):.3f}"
            )

        if r["errors"]:
            lines.append(f"First error  : {r['errors'][0]}")

        lines.append("-" * W)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="localhost:8000", help="Triton HTTP endpoint")
    parser.add_argument("--requests", type=int, default=100, help="Total requests per model")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent workers")
    parser.add_argument(
        "--direction",
        choices=["en-vi", "vi-en", "both"],
        default="both",
        help="Which translation direction(s) to test (default: both)",
    )
    parser.add_argument("--output", metavar="FILE", help="Also write the report to this file")
    args = parser.parse_args()

    client = httpclient.InferenceServerClient(url=args.url)
    if not client.is_server_ready():
        print(f"Triton at {args.url} is not ready.", file=sys.stderr)
        raise SystemExit(1)

    model_map = {
        "en-vi": ["translator_en_vi"],
        "vi-en": ["translator_vi_en"],
        "both": MODELS,
    }
    selected = model_map[args.direction]

    results = []
    for model_name in selected:
        if not client.is_model_ready(model_name):
            print(f"[skip] {model_name} not ready", file=sys.stderr)
            continue
        print(
            f"[stress] {model_name}: {args.requests} requests, concurrency={args.concurrency}",
            file=sys.stderr,
            flush=True,
        )
        results.append(run_stress(args.url, model_name, args.requests, args.concurrency))

    if not results:
        print("No models available.", file=sys.stderr)
        raise SystemExit(1)

    report = build_report(results)
    print(report)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(report + "\n")
        print(f"\nReport saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
