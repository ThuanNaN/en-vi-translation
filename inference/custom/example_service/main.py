"""Minimal custom inference service — replace the stub logic with your real model."""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Custom Inference Service")


class TranslateRequest(BaseModel):
    text: str
    src: str = ""
    tgt: str = ""
    model: str = ""


class TranslateResponse(BaseModel):
    translation: str


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest) -> TranslateResponse:
    # TODO: replace with a real model call, e.g.:
    #   result = my_model.predict(req.text, src=req.src, tgt=req.tgt)
    stub = f"[{req.src}→{req.tgt}] {req.text}"
    return TranslateResponse(translation=stub)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
