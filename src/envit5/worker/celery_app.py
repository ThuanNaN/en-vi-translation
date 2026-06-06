"""Celery application factory."""
from __future__ import annotations

from celery import Celery
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
