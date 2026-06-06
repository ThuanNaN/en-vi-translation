"""Stress test: hammer Triton with concurrent translation requests and print a latency report.

    pip install -e '.[client]'
    python scripts/stresstest.py --url localhost:8000 --requests 200 --concurrency 16
    python scripts/stresstest.py --direction en-vi --output report.txt

Quality evaluation (BLEU) using a HuggingFace dataset:

    pip install -e '.[eval]'
    python scripts/stresstest.py --eval-dataset talmp/en-vi-translation --eval-samples 1000
    python scripts/stresstest.py --eval-dataset talmp/en-vi-translation --eval-samples 1000 --direction en-vi
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tritonclient.http as httpclient

_DATA_DIR = Path(__file__).parent.parent / "data" / "tests"
_RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

_DATA_FILES: dict[str, str] = {
    "translator_en_vi": "en2vi.txt",
    "translator_vi_en": "vi2en.txt",
}

# Column names in the HF dataset for each direction.
_DATASET_COLS: dict[str, tuple[str, str]] = {
    # talmp/en-vi-translation uses "input"/"output"; generic HF translation datasets use "en"/"vi"
    "translator_en_vi": ("input", "output"),
    "translator_vi_en": ("vi", "en"),
}

MODELS = list(_DATA_FILES.keys())


def _load_texts(model_name: str) -> list[str]:
    path = _DATA_DIR / _DATA_FILES[model_name]
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    lines = [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Data file is empty: {path}")
    return lines


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
    texts = _load_texts(model_name)
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
    url: str,
    model_name: str,
    text: str,
) -> tuple[str, float, str | None]:
    t0 = time.perf_counter()
    try:
        hyp = _translate_one(url, model_name, text)
        return hyp, time.perf_counter() - t0, None
    except Exception as exc:
        return "", time.perf_counter() - t0, str(exc)


def run_eval(
    url: str,
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
            pool.submit(_eval_worker, url, model_name, src): i
            for i, src in enumerate(sources)
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
    parser.add_argument("--url", default="localhost:8000", help="Triton HTTP endpoint")
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

        if args.eval_dataset:
            results.append(
                run_eval(
                    url=args.url,
                    model_name=model_name,
                    dataset_name=args.eval_dataset,
                    n_samples=args.eval_samples,
                    split=args.eval_split,
                    seed=args.eval_seed,
                    concurrency=args.concurrency,
                )
            )
        else:
            texts = _load_texts(model_name)
            src = _DATA_DIR / _DATA_FILES[model_name]
            src_label = (
                str(src)
                if src.exists() and src.read_text(encoding="utf-8").strip()
                else "fallback samples"
            )
            print(
                f"[stress] {model_name}: {args.requests} requests, "
                f"concurrency={args.concurrency}, data={src_label} ({len(texts)} lines)",
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
