"""Celery task: translate text via Triton with Redis result caching and long-text chunking."""
from __future__ import annotations

import hashlib
import redis
from envit5.core.settings import get_settings
from envit5.worker.celery_app import celery_app
from envit5.worker.chunker import chunk_text, reassemble
from envit5.worker.triton_client import translate_via_triton


def _cache_key(src: str, tgt: str, text: str) -> str:
    digest = hashlib.sha256(f"{src}-{tgt}:{text}".encode()).hexdigest()
    return f"trans:{digest}"


@celery_app.task(name="envit5.translate", bind=True, max_retries=3, default_retry_delay=5)
def translate_task(self, text: str, src: str, tgt: str) -> str:
    settings = get_settings()
    model_name = settings.model_name_for(src, tgt)

    r = redis.from_url(settings.redis_url)
    key = _cache_key(src, tgt, text)

    cached = r.get(key)
    if cached is not None:
        return cached.decode("utf-8")

    chunks, is_para_start = chunk_text(text)
    translated: list[str] = []
    for chunk in chunks:
        try:
            translated.append(translate_via_triton(chunk, model_name))
        except Exception as exc:
            raise self.retry(exc=exc)

    translation = reassemble(translated, is_para_start)
    r.setex(key, settings.cache_ttl_seconds, translation.encode("utf-8"))
    return translation
