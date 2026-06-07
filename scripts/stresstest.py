"""
Stress test and quality evaluation for translation endpoints.

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
    python scripts/stresstest.py --target app --api-key changeme \
        --eval-dataset talmp/en-vi-translation --eval-samples 1000

Save all translation results to a JSON file:
    python scripts/stresstest.py --eval-dataset talmp/en-vi-translation \
        --results-file results/eval_en_vi.json
    python scripts/stresstest.py --target app --api-key changeme \
        --eval-dataset talmp/en-vi-translation --results-file results/app_eval.json
"""

from __future__ import annotations

import argparse
import datetime
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

_DATASET_COLS: dict[str, tuple[str, str]] = {
    # talmp/en-vi-translation uses "input" and "output".
    "translator_en_vi": ("input", "output"),
    "translator_vi_en": ("output", "input"),
}

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
# App backend: FastAPI POST /translate -> poll GET /jobs/{job_id}
# ---------------------------------------------------------------------------

def _translate_one_app(
    api_url: str,
    api_key: str,
    direction: str,
    text: str,
    poll_interval: float = 0.5,
    timeout: float = 60.0,
) -> str:
    api_url = api_url.rstrip("/")

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }

    payload = json.dumps(
        {
            "text": text,
            "direction": direction,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{api_url}/translate",
        data=payload,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=min(timeout, 10.0)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            job_id = data["job_id"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST /translate failed {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"POST /translate failed: {exc}") from exc

    poll_url = f"{api_url}/jobs/{job_id}"
    poll_headers = {"X-API-Key": api_key}
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        req = urllib.request.Request(poll_url, headers=poll_headers)

        try:
            with urllib.request.urlopen(req, timeout=min(timeout, 10.0)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GET /jobs/{job_id} failed {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"GET /jobs/{job_id} failed: {exc}") from exc

        status = data.get("status")

        if status == "done":
            return str(data.get("translation", ""))

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
        return _translate_one_app(
            api_url=api_url,
            api_key=api_key,
            direction=direction,
            text=text,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    return fn


def _check_app_ready(api_url: str) -> bool:
    api_url = api_url.rstrip("/")

    for path in ["/health", "/ready", "/docs", "/"]:
        try:
            with urllib.request.urlopen(f"{api_url}{path}", timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return True
        except Exception:
            continue

    return False


# ---------------------------------------------------------------------------
# Generic stress / eval runners
# ---------------------------------------------------------------------------

def _worker(
    translate_fn: Callable[[str], str],
    texts: list[str],
    idx: int,
) -> tuple[float, str, str | None]:
    text = texts[idx % len(texts)]
    t0 = time.perf_counter()

    try:
        hypothesis = translate_fn(text)
        return time.perf_counter() - t0, hypothesis, None
    except Exception as exc:
        return time.perf_counter() - t0, "", str(exc)


def run_stress(
    translate_fn: Callable[[str], str],
    model_name: str,
    n_requests: int,
    concurrency: int,
    collect_translations: bool = False,
) -> dict:
    texts = _SAMPLE_TEXTS[model_name]

    latencies: list[float] = []
    errors: list[str] = []
    per_item: list[dict] = []

    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_worker, translate_fn, texts, i): i
            for i in range(n_requests)
        }

        for fut in as_completed(futures):
            i = futures[fut]
            latency, hyp, err = fut.result()

            latencies.append(latency)

            if err:
                errors.append(err)

            if collect_translations:
                entry: dict = {
                    "index": i,
                    "source": texts[i % len(texts)],
                    "hypothesis": hyp,
                    "latency_s": round(latency, 4),
                }

                if err:
                    entry["error"] = err

                per_item.append(entry)

    total_seconds = time.perf_counter() - t_start

    result: dict = {
        "mode": "stress",
        "model": model_name,
        "n_requests": n_requests,
        "concurrency": concurrency,
        "total_seconds": total_seconds,
        "latencies": latencies,
        "errors": errors,
    }

    if collect_translations:
        result["translations"] = sorted(per_item, key=lambda x: x["index"])

    return result


# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------

def _load_eval_pairs(
    dataset_name: str,
    model_name: str,
    n_samples: int,
    split: str,
    seed: int,
) -> tuple[list[str], list[str]]:
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
        f"[eval] Loading '{dataset_name}' "
        f"(split={split}), cache={_RAW_DIR}, sampling {n_samples} rows ...",
        file=sys.stderr,
        flush=True,
    )

    ds = load_dataset(dataset_name, split=split, cache_dir=str(_RAW_DIR))

    missing_cols = [col for col in [src_col, ref_col] if col not in ds.column_names]
    if missing_cols:
        raise ValueError(
            f"Dataset '{dataset_name}' does not contain required columns: {missing_cols}. "
            f"Available columns: {ds.column_names}"
        )

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
    collect_translations: bool = False,
) -> dict:
    try:
        import sacrebleu  # type: ignore[import-untyped]
    except ImportError:
        print(
            "sacrebleu not found. Install with: pip install -e '.[eval]'",
            file=sys.stderr,
        )
        raise SystemExit(1)

    sources, references = _load_eval_pairs(
        dataset_name=dataset_name,
        model_name=model_name,
        n_samples=n_samples,
        split=split,
        seed=seed,
    )

    n = len(sources)

    hypotheses: list[str] = [""] * n
    latencies: list[float] = []
    errors: list[str] = []

    per_item_latencies: list[float] = [0.0] * n
    per_item_errors: list[str | None] = [None] * n

    print(
        f"[eval] Translating {n} sentences with {model_name} "
        f"(concurrency={concurrency}) ...",
        file=sys.stderr,
        flush=True,
    )

    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_idx = {
            pool.submit(_eval_worker, translate_fn, src): i
            for i, src in enumerate(sources)
        }

        done = 0

        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]

            hyp, latency, err = fut.result()

            hypotheses[idx] = hyp
            per_item_latencies[idx] = latency
            latencies.append(latency)

            if err:
                errors.append(err)
                per_item_errors[idx] = err

            done += 1

            if done % 100 == 0 or done == n:
                print(f"[eval] {done}/{n} translated ...", file=sys.stderr, flush=True)

    total_seconds = time.perf_counter() - t_start

    valid_hyps: list[str] = []
    valid_refs: list[str] = []

    for i, hyp in enumerate(hypotheses):
        if hyp:
            valid_hyps.append(hyp)
            valid_refs.append(references[i])

    bleu_score = None
    chrf_score = None

    if valid_hyps:
        bleu_score = sacrebleu.corpus_bleu(valid_hyps, [valid_refs]).score
        chrf_score = sacrebleu.corpus_chrf(valid_hyps, [valid_refs]).score

    result: dict = {
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

    if collect_translations:
        result["translations"] = [
            {
                "index": i,
                "source": sources[i],
                "reference": references[i],
                "hypothesis": hypotheses[i],
                "latency_s": round(per_item_latencies[i], 4),
                **({"error": per_item_errors[i]} if per_item_errors[i] else {}),
            }
            for i in range(n)
        ]

    return result


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0

    s = sorted(data)
    k = (len(s) - 1) * p / 100

    lo = int(k)
    hi = min(lo + 1, len(s) - 1)

    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def build_report(results: list[dict]) -> str:
    width = 64
    lines: list[str] = []

    lines.append("=" * width)
    lines.append("  STRESS / QUALITY EVALUATION REPORT")
    lines.append("=" * width)

    for r in results:
        lines.append("")

        if r.get("mode") == "eval":
            _append_eval_section(lines, r, width)
        else:
            _append_stress_section(lines, r, width)

    return "\n".join(lines)


def _append_stress_section(lines: list[str], r: dict, width: int) -> None:
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

    if r["total_seconds"] > 0:
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

    lines.append("-" * width)


def _append_eval_section(lines: list[str], r: dict, width: int) -> None:
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

    if r["total_seconds"] > 0:
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
        lines.append("  no valid translations to score")

    lines.append("Examples (first 5) :")

    for src, ref, hyp in zip(r["sources"], r["references"], r["hypotheses"]):
        lines.append(f"  SRC : {src[:80]}")
        lines.append(f"  REF : {ref[:80]}")
        lines.append(f"  HYP : {hyp[:80]}")
        lines.append("")

    if r["errors"]:
        lines.append(f"First error  : {r['errors'][0]}")

    lines.append("-" * width)


# ---------------------------------------------------------------------------
# JSON results writer
# ---------------------------------------------------------------------------

def _latency_stats(latencies: list[float]) -> dict:
    return {
        "min_s": round(min(latencies), 4),
        "mean_s": round(statistics.mean(latencies), 4),
        "p50_s": round(_percentile(latencies, 50), 4),
        "p95_s": round(_percentile(latencies, 95), 4),
        "p99_s": round(_percentile(latencies, 99), 4),
        "max_s": round(max(latencies), 4),
    }


def _write_results_json(results: list[dict], path: Path) -> None:
    """
    Serialize full results to JSON.

    The raw top-level latency list is replaced by summary stats to avoid
    making the file too large. Per-translation latencies are still preserved
    inside the "translations" field when --results-file is used.
    """
    json_results: list[dict] = []

    for r in results:
        latencies = r.get("latencies", [])

        entry = {
            k: v
            for k, v in r.items()
            if k != "latencies"
        }

        if latencies:
            entry["latency_stats"] = _latency_stats(latencies)

        json_results.append(entry)

    output = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "results": json_results,
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--target",
        choices=["triton", "app"],
        default="triton",
        help=(
            "Backend to test: 'triton' for direct Triton HTTP, "
            "or 'app' for FastAPI POST /translate. Default: triton"
        ),
    )

    triton_grp = parser.add_argument_group("triton backend")
    triton_grp.add_argument(
        "--url",
        default="localhost:8000",
        help="Triton HTTP endpoint, for example localhost:8000",
    )

    app_grp = parser.add_argument_group("app backend")
    app_grp.add_argument(
        "--api-url",
        default="http://localhost:8080",
        help="FastAPI base URL",
    )
    app_grp.add_argument(
        "--api-key",
        default="changeme",
        help="X-API-Key header value",
    )
    app_grp.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Seconds between job-status polls in app mode. Default: 0.5",
    )
    app_grp.add_argument(
        "--job-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for a single job to complete in app mode. Default: 60",
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Total requests per model in stress mode",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrent workers",
    )
    parser.add_argument(
        "--direction",
        choices=["en-vi", "vi-en", "both"],
        default="both",
        help="Which translation direction to test. Default: both",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Also write the text report to this file",
    )
    parser.add_argument(
        "--results-file",
        metavar="JSON_FILE",
        help=(
            "Save all translation results to this JSON file. "
            "Includes source, hypothesis, reference when available, latency, and errors."
        ),
    )

    eval_grp = parser.add_argument_group(
        "quality evaluation, requires pip install -e '.[eval]'"
    )
    eval_grp.add_argument(
        "--eval-dataset",
        metavar="HF_DATASET",
        default=None,
        help=(
            "HuggingFace dataset to use for BLEU / chrF evaluation, "
            "for example talmp/en-vi-translation. "
            "When set, switches to quality-eval mode instead of stress mode."
        ),
    )
    eval_grp.add_argument(
        "--eval-samples",
        type=int,
        default=1000,
        help="Number of samples to randomly draw from the dataset. Default: 1000",
    )
    eval_grp.add_argument(
        "--eval-split",
        default="train",
        help="Dataset split to use. Default: train",
    )
    eval_grp.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Random seed for sampling. Default: 42",
    )

    args = parser.parse_args()

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be greater than 0")

    if args.requests <= 0:
        raise ValueError("--requests must be greater than 0")

    if args.eval_samples <= 0:
        raise ValueError("--eval-samples must be greater than 0")

    args.api_url = args.api_url.rstrip("/")

    model_map = {
        "en-vi": ["translator_en_vi"],
        "vi-en": ["translator_vi_en"],
        "both": MODELS,
    }

    selected = model_map[args.direction]
    collect_translations = bool(args.results_file)

    if args.target == "triton":
        try:
            import tritonclient.http as httpclient  # type: ignore[import-untyped]
        except ImportError:
            print(
                "tritonclient not found. Install with: pip install -e '.[client]'",
                file=sys.stderr,
            )
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

    else:
        if not _check_app_ready(args.api_url):
            print(f"FastAPI app at {args.api_url} is not reachable.", file=sys.stderr)
            raise SystemExit(1)

        def _is_ready(_model_name: str) -> bool:
            return True

        def _make_fn(model_name: str) -> Callable[[str], str]:
            direction = _MODEL_TO_DIRECTION[model_name]

            return _make_app_translate_fn(
                api_url=args.api_url,
                api_key=args.api_key,
                direction=direction,
                poll_interval=args.poll_interval,
                timeout=args.job_timeout,
            )

        backend_label = f"app ({args.api_url})"

    results: list[dict] = []

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
                collect_translations=collect_translations,
            )
        else:
            print(
                f"[stress] {model_name} ({backend_label}): "
                f"{args.requests} requests, concurrency={args.concurrency}, "
                f"samples={len(_SAMPLE_TEXTS[model_name])}",
                file=sys.stderr,
                flush=True,
            )

            result = run_stress(
                translate_fn=translate_fn,
                model_name=model_name,
                n_requests=args.requests,
                concurrency=args.concurrency,
                collect_translations=collect_translations,
            )

        result["backend"] = backend_label
        results.append(result)

    if not results:
        print("No models available.", file=sys.stderr)
        raise SystemExit(1)

    report = build_report(results)
    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")

        print(f"\nReport saved to {args.output}", file=sys.stderr)

    if args.results_file:
        results_path = Path(args.results_file)
        _write_results_json(results, results_path)

        n_translations = sum(len(r.get("translations", [])) for r in results)

        print(
            f"\nResults ({n_translations} translations) saved to {results_path}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()