"""Celery application factory with Prometheus metrics HTTP server."""
from __future__ import annotations

import os
from celery import Celery
from celery.signals import worker_init, worker_process_shutdown
from prometheus_client import CollectorRegistry, multiprocess, start_http_server
from envit5.core.settings import get_settings


def _make_celery() -> Celery:
    s = get_settings()
    app = Celery("envit5", broker=s.celery_broker_url, backend=s.celery_result_backend)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_soft_time_limit=25,
        task_time_limit=30,
    )
    return app


celery_app = _make_celery()


@worker_init.connect
def _start_metrics_server(**_kwargs):
    """Start a Prometheus HTTP server in the main worker process before forking.

    In multiprocess mode (PROMETHEUS_MULTIPROC_DIR set) the server aggregates
    metric files written by all forked children on each scrape.
    """
    port = int(os.environ.get("ENVIT5_METRICS_PORT", "9091"))
    prom_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")

    if prom_dir:
        os.makedirs(prom_dir, exist_ok=True)
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        start_http_server(port, registry=registry)
    else:
        start_http_server(port)


@worker_process_shutdown.connect
def _cleanup_dead_worker(pid, _exitcode, **_kwargs):
    """Remove mmap files for exited worker processes to keep the registry clean."""
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        multiprocess.mark_process_dead(pid)
