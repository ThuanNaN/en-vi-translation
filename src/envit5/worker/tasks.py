"""Celery task: translate text via Triton with Redis result caching and long-text chunking."""
from __future__ import annotations

import hashlib
import time
import redis
from celery.exceptions import Retry

from envit5.core.metrics import (
    active_translations,
    cache_hits_total,
    cache_misses_total,
    text_chunks_histogram,
    translation_duration_seconds,
    translations_total,
)
from envit5.core.settings import get_settings
from envit5.worker.celery_app import celery_app
from envit5.worker.chunker import chunk_text, reassemble
from envit5.worker.triton_client import translate_via_triton


def _cache_key(src: str, tgt: str, text: str) -> str:
    digest = hashlib.sha256(f"{src}-{tgt}:{text}".encode()).hexdigest()
    return f"trans:{digest}"


@celery_app.task(name="envit5.translate", bind=True, max_retries=3, default_retry_delay=5)
def translate_task(self, text: str, src: str, tgt: str) -> str:  # pylint: disable=too-many-locals
    settings = get_settings()
    model_name = settings.model_name_for(src, tgt)
    direction = f"{src}-{tgt}"

    r = redis.from_url(settings.redis_url)
    key = _cache_key(src, tgt, text)

    cached = r.get(key)
    if cached is not None:
        cache_hits_total.labels(direction=direction).inc()
        return cached.decode("utf-8")

    cache_misses_total.labels(direction=direction).inc()
    chunks, is_para_start = chunk_text(text)
    text_chunks_histogram.labels(direction=direction).observe(len(chunks))

    active_translations.labels(direction=direction).inc()
    start = time.perf_counter()
    try:
        translated: list[str] = []
        for chunk in chunks:
            try:
                translated.append(translate_via_triton(chunk, model_name))
            except Exception as exc:
                translations_total.labels(direction=direction, status="retry").inc()
                raise self.retry(exc=exc)

        translation = reassemble(translated, is_para_start)
        r.setex(key, settings.cache_ttl_seconds, translation.encode("utf-8"))
        translations_total.labels(direction=direction, status="success").inc()
        return translation
    except Retry:
        raise
    except Exception:
        translations_total.labels(direction=direction, status="failure").inc()
        raise
    finally:
        translation_duration_seconds.labels(direction=direction).observe(
            time.perf_counter() - start
        )
        active_translations.labels(direction=direction).dec()
