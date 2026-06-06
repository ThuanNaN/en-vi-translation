# en-vi-translation

English ↔ Vietnamese Neural Machine Translation served on **Triton Inference Server**, with async job dispatch via **Celery + Redis** and a **FastAPI** REST API.

## Architecture

```
client ──POST /translate──▶ FastAPI ──enqueue──▶ Redis (broker) ──▶ Celery worker
   ▲                          │ (returns job_id)                        │ tritonclient
   └──GET /jobs/{id}──────────┘                                         ▼
                  Redis (result cache + Celery backend) ◀──────────── Triton
                                                          (translator_en_vi,
   Prometheus ◀─ metrics (api:8080, worker:9091, triton:8002) ─▶ Grafana    translator_vi_en)
```

- **Two single-direction models** — `Helsinki-NLP/opus-mt-en-vi` and `Helsinki-NLP/opus-mt-vi-en`, exported to ONNX and loaded via the Triton Python backend with Optimum `ORTModelForSeq2SeqLM`.
- **Async API** — `POST /translate` enqueues a job and returns a `job_id`; `GET /jobs/{id}` polls for the result. Auth via `X-API-Key` header.
- **Direction routing** — explicit `source`/`target` (or `direction: "en-vi"`), with automatic source-language detection as fallback.
- **Redis caching** — results are cached by `sha256(direction:text)` for 24 h.
- **Prometheus metrics** — HTTP request rate/latency, translation throughput/latency, cache hit rate, Triton inference latency, and active-job gauge. Grafana dashboard auto-provisioned as `EN↔VI Translation Service`.

## Quick start

```bash
# 1. Build images and export models (one-time; downloads ~500 MB of HF weights)
make build
make build-app
make export

# 2. Start all services
make up         # Triton + Redis
make api-up     # FastAPI + Celery worker
make observe    # Prometheus + Grafana + Loki stack

# 3. Smoke-test
pip install -e '.[client]'
make smoke      # translates a sample sentence each direction

# 4. Open Grafana at http://localhost:3000  (admin / admin)
#    Open API docs at http://localhost:8080/docs
```

## API

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/translate` | POST | `X-API-Key` | Submit translation job, returns `{ job_id }` |
| `/jobs/{job_id}` | GET | `X-API-Key` | Poll job status: `pending` / `started` / `done` / `failed` |
| `/metrics` | GET | — | Prometheus metrics |

**Request body for `POST /translate`:**
```json
{ "text": "Hello world", "source": "en", "target": "vi" }
```
Or use the `direction` shorthand (`"en-vi"` / `"vi-en"`), or omit both for auto-detection.

## Development

```bash
pip install -e '.[dev]'   # ruff + pylint + pytest

pytest                    # run all tests (18 tests)
pytest tests/api/test_translate.py::test_submit_en_vi  # single test

ruff check . && ruff format .   # lint + format
make lint                       # ruff + pylint
```

## Makefile targets

| Target | Description |
|---|---|
| `make build` | Build Triton image (installs torch/transformers/optimum) |
| `make build-app` | Build API + worker image |
| `make export` | Export HF checkpoints to ONNX (runs inside Triton container) |
| `make up` / `make down` | Start / stop Triton + Redis |
| `make api-up` / `make api-down` | Start / stop API + worker |
| `make observe` | Start Prometheus, Grafana, Loki, Promtail, exporters |
| `make smoke` | Quick end-to-end translation test |
| `make stress` | Latency/throughput benchmark against Triton |
| `make eval` | BLEU/chrF quality eval on 5000 HF samples |
| `make app-stress` | Benchmark through the FastAPI layer (`API_KEY=mykey`) |
| `make logs` / `make api-logs` / `make worker-logs` | Tail service logs |
| `make clean` | Delete exported ONNX artifacts |
| `make remove` | Remove containers, volumes, images, and networks |

## Ports

| Service | Port |
|---|---|
| Triton HTTP | 8000 |
| Triton gRPC | 8001 |
| Triton metrics | 8002 |
| FastAPI | 8080 |
| Worker metrics | 9091 |
| Prometheus | 9090 |
| Grafana | 3000 |
| Loki | 3100 |

## Configuration

All settings use the `ENVIT5_` prefix (pydantic-settings, can also be set via `.env`):

| Variable | Default | Description |
|---|---|---|
| `ENVIT5_API_KEYS` | — | Comma-separated API keys (required) |
| `ENVIT5_TRITON_HTTP_URL` | `localhost:8000` | Triton HTTP endpoint |
| `ENVIT5_REDIS_URL` | `redis://localhost:6379/0` | Result cache URL |
| `ENVIT5_CACHE_TTL_SECONDS` | `86400` | Translation cache TTL |
| `ENVIT5_MAX_NEW_TOKENS` | `512` | Generation limit |
| `ENVIT5_NUM_BEAMS` | `1` | Beam search width |
| `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD` | `admin` / `admin` | Grafana credentials |
