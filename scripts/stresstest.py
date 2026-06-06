"""Stress test and quality evaluation for translation endpoints.

Triton backend (direct, default):
    pip install -e '.[client]'
    python scripts/stresstest.py --url localhost:8000 --requests 200 --concurrency 16
    python scripts/stresstest.py --direction en-vi --output report.txt

FastAPI app backend (POST /translate + poll GET /jobs/{id}):
    python scripts/stresstest.py --target app --api-url http://localhost:8080 --api-key changeme
    python scripts/stresstest.py --target app --api-key changeme --direction en-vi

Quality evaluation (BLEU) using a HuggingFace dataset:
    pip install -e '.[eval]'
    python scripts/stresstest.py --eval-dataset talmp/en-vi-translation --eval-samples 1000
    python scripts/stresstest.py --target app --api-key changeme \\
        --eval-dataset talmp/en-vi-translation --eval-samples 1000
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# Built-in sample sentences used as stress-test input (cycled over n_requests).
_SAMPLE_TEXTS: dict[str, list[str]] = {
    "translator_en_vi": [
        "Hello, how are you?",
        "The weather is nice today.",
        "I would like to order a coffee, please.",
        "Where is the nearest hospital?",
        "Thank you very much for your help.",
        "Can you speak more slowly?",
        "What time does the train depart?",
        "I need to find a pharmacy.",
        "The food here is delicious.",
        "How much does this cost?",
    ],
    "translator_vi_en": [
        "Xin chào, bạn có khỏe không?",
        "Thời tiết hôm nay rất đẹp.",
        "Tôi muốn đặt một ly cà phê.",
        "Bệnh viện gần nhất ở đâu?",
        "Cảm ơn bạn rất nhiều vì đã giúp đỡ.",
        "Bạn có thể nói chậm hơn không?",
        "Tàu khởi hành lúc mấy giờ?",
        "Tôi cần tìm một hiệu thuốc.",
        "Món ăn ở đây rất ngon.",
        "Cái này giá bao nhiêu?",
    ],
}

# Column names in the HF dataset for each direction.
_DATASET_COLS: dict[str, tuple[str, str]] = {
    # talmp/en-vi-translation uses "input"/"output";
    "translator_en_vi": ("input", "output"),
    "translator_vi_en": ("output", "input"),
}

# Maps internal model name → API direction string for the app backend.
_MODEL_TO_DIRECTION: dict[str, str] = {
    "translator_en_vi": "en-vi",
    "translator_vi_en": "vi-en",
}

MODELS = list(_SAMPLE_TEXTS.keys())


# ---------------------------------------------------------------------------
# Triton backend
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_triton_client(url: str):
    if not hasattr(_thread_local, "client"):
        import tritonclient.http as httpclient  # type: ignore[import-untyped]

        _thread_local.client = httpclient.InferenceServerClient(url=url)
    return _thread_local.client


def _translate_one_triton(url: str, model_name: str, text: str) -> str:
    import numpy as np
    import tritonclient.http as httpclient  # type: ignore[import-untyped]

    client = _get_triton_client(url)
    inp = httpclient.InferInput("INPUT_TEXT", [1, 1], "BYTES")
    inp.set_data_from_numpy(np.array([[text]], dtype=object))
    out = httpclient.InferRequestedOutput("OUTPUT_TEXT")
    result = client.infer(model_name=model_name, inputs=[inp], outputs=[out])
    value = result.as_numpy("OUTPUT_TEXT").reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)


def _make_triton_translate_fn(url: str, model_name: str) -> Callable[[str], str]:
    def fn(text: str) -> str:
        return _translate_one_triton(url, model_name, text)

    return fn


# ---------------------------------------------------------------------------
# App backend (FastAPI: POST /translate → poll GET /jobs/{job_id})
# ---------------------------------------------------------------------------

def _translate_one_app(
    api_url: str,
    api_key: str,
    direction: str,
    text: str,
    poll_interval: float = 0.5,
    timeout: float = 60.0,
) -> str:
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}
    payload = json.dumps({"text": text, "direction": direction}).encode()

    req = urllib.request.Request(
        f"{api_url}/translate", data=payload, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            job_id = json.loads(resp.read())["job_id"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"POST /translate failed {exc.code}: {body}") from exc

    poll_url = f"{api_url}/jobs/{job_id}"
    poll_headers = {"X-API-Key": api_key}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        req = urllib.request.Request(poll_url, headers=poll_headers)
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"GET /jobs/{job_id} failed {exc.code}: {body}") from exc
        status = data["status"]
        if status == "done":
            return data["translation"]
        if status == "failed":
            raise RuntimeError(f"Translation failed: {data.get('error')}")
        time.sleep(poll_interval)

    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def _make_app_translate_fn(
    api_url: str,
    api_key: str,
    direction: str,
    poll_interval: float,
    timeout: float,
) -> Callable[[str], str]:
    def fn(text: str) -> str:
        return _translate_one_app(api_url, api_key, direction, text, poll_interval, timeout)

    return fn


def _check_app_ready(api_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{api_url}/docs", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Generic stress / eval runners
# ---------------------------------------------------------------------------

def _worker(
    translate_fn: Callable[[str], str], texts: list[str], idx: int
) -> tuple[float, str | None]:
    text = texts[idx % len(texts)]
    t0 = time.perf_counter()
    try:
        translate_fn(text)
        return time.perf_counter() - t0, None
    except Exception as exc:
        return time.perf_counter() - t0, str(exc)


def run_stress(
    translate_fn: Callable[[str], str],
    model_name: str,
    n_requests: int,
    concurrency: int,
) -> dict:
    texts = _SAMPLE_TEXTS[model_name]
    latencies: list[float] = []
    errors: list[str] = []

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_worker, translate_fn, texts, i) for i in range(n_requests)]
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


# ---------------------------------------------------------------------------
# Quality evaluation (BLEU) using a HuggingFace dataset
# ---------------------------------------------------------------------------

def _load_eval_pairs(
    dataset_name: str,
    model_name: str,
    n_samples: int,
    split: str,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Return (sources, references) sampled from a HF dataset."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        print(
            "datasets package not found. Install with: pip install -e '.[eval]'",
            file=sys.stderr,
        )
        raise SystemExit(1)

    src_col, ref_col = _DATASET_COLS[model_name]
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[eval] Loading '{dataset_name}' (split={split}), cache={_RAW_DIR}, sampling {n_samples} rows …",
        file=sys.stderr,
        flush=True,
    )
    ds = load_dataset(dataset_name, split=split, cache_dir=str(_RAW_DIR))
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n_samples, len(ds)))
    sample = ds.select(indices)
    sources = [str(row[src_col]) for row in sample]
    references = [str(row[ref_col]) for row in sample]
    return sources, references


def _eval_worker(
    translate_fn: Callable[[str], str],
    text: str,
) -> tuple[str, float, str | None]:
    t0 = time.perf_counter()
    try:
        hyp = translate_fn(text)
        return hyp, time.perf_counter() - t0, None
    except Exception as exc:
        return "", time.perf_counter() - t0, str(exc)


def run_eval(
    translate_fn: Callable[[str], str],
    model_name: str,
    dataset_name: str,
    n_samples: int,
    split: str,
    seed: int,
    concurrency: int,
) -> dict:
    try:
        import sacrebleu  # type: ignore[import-untyped]
    except ImportError:
        print(
            "sacrebleu not found. Install with: pip install -e '.[eval]'",
            file=sys.stderr,
        )
        raise SystemExit(1)

    sources, references = _load_eval_pairs(dataset_name, model_name, n_samples, split, seed)
    n = len(sources)
    hypotheses: list[str] = [""] * n
    latencies: list[float] = []
    errors: list[str] = []

    print(
        f"[eval] Translating {n} sentences with {model_name} (concurrency={concurrency}) …",
        file=sys.stderr,
        flush=True,
    )

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_idx = {
            pool.submit(_eval_worker, translate_fn, src): i for i, src in enumerate(sources)
        }
        done = 0
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            hyp, latency, err = fut.result()
            hypotheses[idx] = hyp
            latencies.append(latency)
            if err:
                errors.append(err)
            done += 1
            if done % 100 == 0 or done == n:
                print(f"[eval] {done}/{n} translated …", file=sys.stderr, flush=True)
    total_seconds = time.perf_counter() - t_start

    # Filter out failed rows before scoring.
    valid_hyps = [h for h in hypotheses if h]
    valid_refs = [references[i] for i, h in enumerate(hypotheses) if h]

    bleu_score = None
    chrf_score = None
    if valid_hyps:
        bleu_score = sacrebleu.corpus_bleu(valid_hyps, [valid_refs]).score
        chrf_score = sacrebleu.corpus_chrf(valid_hyps, [valid_refs]).score

    return {
        "mode": "eval",
        "model": model_name,
        "dataset": dataset_name,
        "n_samples": n,
        "n_valid": len(valid_hyps),
        "concurrency": concurrency,
        "total_seconds": total_seconds,
        "latencies": latencies,
        "errors": errors,
        "bleu": bleu_score,
        "chrf": chrf_score,
        "sources": sources[:5],
        "references": references[:5],
        "hypotheses": [hypotheses[i] for i in range(min(5, n))],
    }


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _percentile(data: list[float], p: float) -> float:
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def build_report(results: list[dict]) -> str:
    W = 64
    lines: list[str] = []
    lines.append("=" * W)
    lines.append("  STRESS / QUALITY EVALUATION REPORT")
    lines.append("=" * W)

    for r in results:
        lines.append("")
        if r.get("mode") == "eval":
            _append_eval_section(lines, r, W)
        else:
            _append_stress_section(lines, r, W)

    return "\n".join(lines)


def _append_stress_section(lines: list[str], r: dict, W: int) -> None:
    n = r["n_requests"]
    n_err = len(r["errors"])
    n_ok = n - n_err
    lats = r["latencies"]

    lines.append(f"Model        : {r['model']}")
    if r.get("backend"):
        lines.append(f"Backend      : {r['backend']}")
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


def _append_eval_section(lines: list[str], r: dict, W: int) -> None:
    n = r["n_samples"]
    n_ok = r["n_valid"]
    n_err = len(r["errors"])
    lats = r["latencies"]

    lines.append(f"Model        : {r['model']}")
    if r.get("backend"):
        lines.append(f"Backend      : {r['backend']}")
    lines.append(f"Dataset      : {r['dataset']}")
    lines.append(f"Samples      : {n}  (concurrency={r['concurrency']})")
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

    lines.append("Quality      :")
    if r["bleu"] is not None:
        lines.append(f"  BLEU  {r['bleu']:.2f}")
        lines.append(f"  chrF  {r['chrf']:.2f}")
    else:
        lines.append("  (no valid translations to score)")

    # Print a few examples.
    lines.append("Examples (first 5) :")
    for src, ref, hyp in zip(r["sources"], r["references"], r["hypotheses"]):
        lines.append(f"  SRC : {src[:80]}")
        lines.append(f"  REF : {ref[:80]}")
        lines.append(f"  HYP : {hyp[:80]}")
        lines.append("")

    if r["errors"]:
        lines.append(f"First error  : {r['errors'][0]}")

    lines.append("-" * W)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    # Backend selection.
    parser.add_argument(
        "--target",
        choices=["triton", "app"],
        default="triton",
        help="Backend to test: 'triton' (direct Triton HTTP) or 'app' (FastAPI via POST /translate). Default: triton",
    )

    # Triton args.
    triton_grp = parser.add_argument_group("triton backend")
    triton_grp.add_argument("--url", default="localhost:8000", help="Triton HTTP endpoint (host:port)")

    # App args.
    app_grp = parser.add_argument_group("app backend")
    app_grp.add_argument("--api-url", default="http://localhost:8080", help="FastAPI base URL")
    app_grp.add_argument("--api-key", default="changeme", help="X-API-Key header value")
    app_grp.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Seconds between job-status polls in app mode (default: 0.5)",
    )
    app_grp.add_argument(
        "--job-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for a single job to complete in app mode (default: 60)",
    )

    parser.add_argument("--requests", type=int, default=100, help="Total requests per model (stress mode)")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent workers")
    parser.add_argument(
        "--direction",
        choices=["en-vi", "vi-en", "both"],
        default="both",
        help="Which translation direction(s) to test (default: both)",
    )
    parser.add_argument("--output", metavar="FILE", help="Also write the report to this file")

    # Quality evaluation args.
    eval_grp = parser.add_argument_group("quality evaluation (requires pip install -e '.[eval]')")
    eval_grp.add_argument(
        "--eval-dataset",
        metavar="HF_DATASET",
        default=None,
        help="HuggingFace dataset to use for BLEU evaluation (e.g. talmp/en-vi-translation). "
             "When set, switches to quality-eval mode instead of latency stress test.",
    )
    eval_grp.add_argument(
        "--eval-samples",
        type=int,
        default=1000,
        help="Number of samples to randomly draw from the dataset (default: 1000)",
    )
    eval_grp.add_argument(
        "--eval-split",
        default="train",
        help="Dataset split to use (default: train)",
    )
    eval_grp.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )

    args = parser.parse_args()

    model_map = {
        "en-vi": ["translator_en_vi"],
        "vi-en": ["translator_vi_en"],
        "both": MODELS,
    }
    selected = model_map[args.direction]

    if args.target == "triton":
        try:
            import tritonclient.http as httpclient  # type: ignore[import-untyped]
        except ImportError:
            print("tritonclient not found. Install with: pip install -e '.[client]'", file=sys.stderr)
            raise SystemExit(1)

        client = httpclient.InferenceServerClient(url=args.url)
        if not client.is_server_ready():
            print(f"Triton at {args.url} is not ready.", file=sys.stderr)
            raise SystemExit(1)

        def _is_ready(model_name: str) -> bool:
            return client.is_model_ready(model_name)

        def _make_fn(model_name: str) -> Callable[[str], str]:
            return _make_triton_translate_fn(args.url, model_name)

        backend_label = f"triton ({args.url})"

    else:  # app
        if not _check_app_ready(args.api_url):
            print(f"FastAPI app at {args.api_url} is not reachable.", file=sys.stderr)
            raise SystemExit(1)

        def _is_ready(_model_name: str) -> bool:
            return True  # app doesn't expose per-model readiness

        def _make_fn(model_name: str) -> Callable[[str], str]:
            direction = _MODEL_TO_DIRECTION[model_name]
            return _make_app_translate_fn(
                args.api_url, args.api_key, direction, args.poll_interval, args.job_timeout
            )

        backend_label = f"app ({args.api_url})"

    results = []
    for model_name in selected:
        if not _is_ready(model_name):
            print(f"[skip] {model_name} not ready", file=sys.stderr)
            continue

        translate_fn = _make_fn(model_name)

        if args.eval_dataset:
            result = run_eval(
                translate_fn=translate_fn,
                model_name=model_name,
                dataset_name=args.eval_dataset,
                n_samples=args.eval_samples,
                split=args.eval_split,
                seed=args.eval_seed,
                concurrency=args.concurrency,
            )
        else:
            print(
                f"[stress] {model_name} ({backend_label}): {args.requests} requests, "
                f"concurrency={args.concurrency}, samples={len(_SAMPLE_TEXTS[model_name])}",
                file=sys.stderr,
                flush=True,
            )
            result = run_stress(translate_fn, model_name, args.requests, args.concurrency)

        result["backend"] = backend_label
        results.append(result)

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
