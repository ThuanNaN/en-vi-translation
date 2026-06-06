"""FastAPI application: POST /translate and GET /jobs/{job_id}."""
from __future__ import annotations

from typing import Annotated
from celery.result import AsyncResult
from fastapi import Depends, FastAPI, Header, HTTPException, status
from envit5.api.models import JobResponse, TranslateRequest, TranslateResponse
from envit5.core.settings import get_settings
from envit5.worker.tasks import translate_task

app = FastAPI(title="EN↔VI Translation API", version="0.1.0")


def _require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> str:
    settings = get_settings()
    if x_api_key is None or not settings.api_keys or x_api_key not in settings.api_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
    return x_api_key


def _detect_direction(text: str) -> tuple[str, str]:
    import py3langid  # imported lazily to avoid loading startup overhead when unused

    lang, _ = py3langid.classify(text)
    if lang == "en":
        return "en", "vi"
    if lang == "vi":
        return "vi", "en"
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Auto-detected language '{lang}' is not supported. Only 'en' and 'vi' are served.",
    )


@app.post("/translate", response_model=TranslateResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_translation(
    req: TranslateRequest,
    _: str = Depends(_require_api_key),
) -> TranslateResponse:
    src, tgt = req.source, req.target
    if src is None:
        src, tgt = _detect_direction(req.text)

    task = translate_task.delay(req.text, src, tgt)
    return TranslateResponse(job_id=task.id)


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    _: str = Depends(_require_api_key),
) -> JobResponse:
    result = AsyncResult(job_id)

    # Celery returns PENDING + result=None for unknown IDs — treat as 404.
    if result.status == "PENDING" and result.result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    if result.status in ("PENDING", "RETRY"):
        return JobResponse(job_id=job_id, status="pending")

    if result.status == "STARTED":
        return JobResponse(job_id=job_id, status="started")

    if result.status == "SUCCESS":
        return JobResponse(job_id=job_id, status="done", translation=result.result)

    error = str(result.result) if result.result else "Unknown error"
    return JobResponse(job_id=job_id, status="failed", error=error)
