# Requirements & Design — en-vi-translation

EN↔VI Neural Machine Translation served on **Triton Inference Server** behind a
**Celery/Redis** queue, exposed through a **FastAPI** API. This document is the agreed
spec; it was captured from a requirements Q&A and drives the phased build.

## Decisions

| Area | Decision |
| --- | --- |
| Model source | Pretrained, off-the-shelf (no training/fine-tuning) |
| Models | **Two single-direction models** — `en→vi`: `Helsinki-NLP/opus-mt-en-vi`, `vi→en`: `Helsinki-NLP/opus-mt-vi-en` (both Marian) |
| Serving | Triton **Python backend** loading 🤗 Optimum `ORTModelForSeq2SeqLM` (+ tokenizer), calling `.generate()`. ONNX Runtime under the hood. |
| Why not pure ONNX backend | The `onnxruntime` backend runs a single forward pass and cannot drive the autoregressive decode loop a seq2seq model needs. |
| Hardware | **No GPU** → ONNX Runtime on CPU (TensorRT dropped). |
| Queue | **Celery + Redis** (broker + result backend). |
| API | **FastAPI**, **async-only**: submit returns a `job_id`, then poll. |
| Direction routing | Explicit `source`/`target` (or `direction`) param; **auto-detect source as fallback** when omitted. |
| Caching | Redis result cache, keyed by `sha256(direction:text)`. |
| Auth | API key per request via `X-API-Key`. |
| Monitoring | Prometheus metrics (API + worker + Triton `:8002`) → Grafana. |
| Deploy | docker-compose (Triton + FastAPI + Celery worker(s) + Redis + Prometheus/Grafana). |
| Out of scope (for now) | Rate limiting; sync translate endpoint; gRPC API; fine-tuning. |

## Architecture

```
client ──POST /translate──▶ FastAPI ──enqueue──▶ Redis (broker) ──▶ Celery worker
   ▲                          │ (returns job_id)                        │ tritonclient
   └──GET /jobs/{id}──────────┘                                         ▼
                  Redis (result cache + Celery backend) ◀──────────── Triton
                                                          (translator_en_vi,
   Prometheus ◀─ metrics (api, worker, triton:8002) ─▶ Grafana         translator_vi_en)
```

**Request flow:** API resolves direction (explicit param, else auto-detect source
language) → enqueues `translate_task(text, src, tgt)` → worker checks the Redis cache →
on miss calls the matching Triton model over HTTP/gRPC → caches the result → API returns
it when the client polls `GET /jobs/{id}`. Every endpoint requires a valid `X-API-Key`.

## Triton model contract

Both models are string-in / string-out (tokenization lives inside the model):

- input  `INPUT_TEXT`  : `TYPE_STRING`, dims `[1]`
- output `OUTPUT_TEXT` : `TYPE_STRING`, dims `[1]`
- `max_batch_size: 16` with `dynamic_batching` so concurrent requests are coalesced.
- `parameters`: `max_new_tokens`, `num_beams` (read in `model.py`).

Layout:

```
model_repository/
  translator_en_vi/  config.pbtxt  1/{model.py, onnx/}
  translator_vi_en/  config.pbtxt  1/{model.py, onnx/}
```

`onnx/` is produced by `scripts/export_models.py` and is git-ignored (regenerate, don't commit).

## Phased roadmap

1. **Scaffold & config** — package layout, `pyproject.toml`, `Settings`, `.env.example`. ✅
2. **Model + Triton (core)** — export script, two Python-backend models, Triton in compose; verify both directions. ✅ *(this scaffold; needs `make build/export/up` to actually run)*
3. **Queue + async API** — Celery task → Triton; `POST /translate` + `GET /jobs/{id}`; direction param + auto-detect. ⬜
4. **Cross-cutting** — Redis result cache, API-key auth, Prometheus/Grafana, health checks. ⬜
5. **Hardening** — tests, retries/timeouts, structured errors. ⬜

## Open items / notes

- `Helsinki-NLP/opus-mt-en-vi`'s model card mentions a `>>id<<` target-language token used
  by multi-target OPUS models. This pair is single-target, so it is typically not needed;
  if en→vi output looks wrong, prepend a `>>vie<<` token in `model.py` preprocessing.
- Triton base image tag (`TRITON_VERSION`) must point at a tag published on nvcr.io.
- Auto-detect language library to be chosen in Phase 3 (leaning `py3langid`: offline, deterministic).
