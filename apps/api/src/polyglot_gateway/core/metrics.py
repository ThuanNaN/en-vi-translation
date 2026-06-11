"""Prometheus metrics for the EN↔VI translation service.

All metric objects live here so both the API and worker import from one place.
In the worker container PROMETHEUS_MULTIPROC_DIR must be set before this
module is imported; prometheus_client then uses mmap files instead of
in-process memory so metrics are aggregated across all prefork children.
"""
from __future__ import annotations

import os
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    make_asgi_app,
    multiprocess,
)

# ---------------------------------------------------------------------------
# HTTP layer — instrumented by MetricsMiddleware in api/app.py
# ---------------------------------------------------------------------------

http_requests_total = Counter(
    "polyglot_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

http_request_duration_seconds = Histogram(
    "polyglot_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ---------------------------------------------------------------------------
# Translation layer — instrumented by tasks.py
# ---------------------------------------------------------------------------

translations_total = Counter(
    "polyglot_translations_total",
    "Translation task completions by direction and status (success/failure/retry)",
    ["direction", "status"],
)

translation_duration_seconds = Histogram(
    "polyglot_translation_duration_seconds",
    "End-to-end translation task duration in seconds",
    ["direction"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

cache_hits_total = Counter(
    "polyglot_cache_hits_total",
    "Redis translation cache hits",
    ["direction"],
)

cache_misses_total = Counter(
    "polyglot_cache_misses_total",
    "Redis translation cache misses",
    ["direction"],
)

text_chunks_histogram = Histogram(
    "polyglot_text_chunks",
    "Number of chunks per translation request",
    ["direction"],
    buckets=[1, 2, 3, 4, 6, 8, 12, 16, 32],
)

active_translations = Gauge(
    "polyglot_active_translations",
    "Currently in-flight translation tasks",
    ["direction"],
    multiprocess_mode="livesum",
)

# ---------------------------------------------------------------------------
# Triton layer — instrumented by triton_client.py
# ---------------------------------------------------------------------------

triton_infer_duration_seconds = Histogram(
    "polyglot_triton_infer_duration_seconds",
    "Triton inference duration per chunk in seconds",
    ["model"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)


def make_metrics_asgi_app():
    """Return an ASGI app that serves Prometheus metrics.

    Uses MultiProcessCollector when PROMETHEUS_MULTIPROC_DIR is set so
    that all worker subprocesses are aggregated correctly.
    """
    prom_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if prom_dir:
        os.makedirs(prom_dir, exist_ok=True)
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return make_asgi_app(registry=registry)
    return make_asgi_app()
