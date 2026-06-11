# en-vi-translation

English ↔ Vietnamese Neural Machine Translation served on **Triton Inference Server**, with async job dispatch via **Celery + Redis**, a **FastAPI** REST API, a **Traefik** API gateway, and a **Gradio** demo UI.

## Architecture

```
                    Traefik :80
                    ┌──────────────────────────────────────┐
client ─POST /translate─▶ FastAPI ──enqueue──▶ Redis (broker) ──▶ Celery worker
   ▲                         │ (returns job_id)                        │ tritonclient
   └──GET /jobs/{id}─────────┘                                         ▼
                 Redis (result cache + Celery backend) ◀──────────── Triton
                                                         (translator_en_vi,
  Prometheus ◀─ metrics (api:8080, worker:9091, triton:8002)   translator_vi_en)
       └──▶ Grafana :3000
```

- **Two single-direction models** — `Helsinki-NLP/opus-mt-en-vi` and `Helsinki-NLP/opus-mt-vi-en`, exported to ONNX and loaded via the Triton Python backend with Optimum `ORTModelForSeq2SeqLM`.
- **Async API** — `POST /translate` enqueues a job and returns a `job_id`; `GET /jobs/{id}` polls for the result. Auth via `X-API-Key` header.
- **Direction routing** — explicit `source`/`target` (or `direction: "en-vi"`), with automatic source-language detection as fallback.
- **Long-text chunking** — input is split at sentence boundaries into ≤ 1 500-char chunks and reassembled after translation.
- **Redis caching** — results are cached by `sha256(direction:text)` for 24 h, short-circuiting Triton entirely on cache hits.
- **Traefik gateway** — rate-limited (100 req/s, burst 50) and gzip-compressed. Routes `/translate`, `/jobs`, and `/health` to the API; routes `/` to the Gradio UI.
- **Prometheus metrics** — HTTP request rate/latency, translation throughput/latency, cache hit rate, and active-job gauge. Grafana dashboard auto-provisioned as `EN↔VI Translation Service`.

## Quick start

```bash
# 1. Build images and export models (one-time; downloads ~500 MB of HF weights)
make build        # Triton image
make build-app    # API + worker image
make export       # export HF checkpoints to ONNX (runs inside Triton container)

# 2. Start all services
make up           # Triton + Redis
make gateway-up   # Traefik API gateway
make api-up       # FastAPI + Celery worker
make ui-up        # Gradio demo UI (optional)
make observe      # Prometheus + Grafana + Loki stack (optional)

# 3. Smoke-test
pip install -e '.[client]'
make smoke        # translates a sample sentence each direction

# 4. Open
#    API docs  → http://localhost/docs
#    Gradio UI → http://localhost/
#    Grafana   → http://localhost:3000  (admin / admin)
#    Traefik   → http://localhost:8888/dashboard/
```

Or start everything in one shot (skips Triton build and ONNX export):

```bash
make all
```

## Production deploy

Uses pre-built images from GHCR — no local build required:

```bash
make prod-pull IMAGE_TAG=v0.1.0   # pull images
make prod-up   IMAGE_TAG=v0.1.0   # start all services
make prod-down                     # stop and remove
```

## API

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/translate` | POST | `X-API-Key` | Submit translation job, returns `{ job_id }` |
| `/jobs/{job_id}` | GET | `X-API-Key` | Poll job status: `pending` / `started` / `done` / `failed` |
| `/health` | GET | — | Liveness check |
| `/metrics` | GET | — | Prometheus metrics (not routed through Traefik) |

**Request body for `POST /translate`:**
```json
{ "text": "Hello world", "source": "en", "target": "vi" }
```
Or use the `direction` shorthand (`"en-vi"` / `"vi-en"`), or omit both for auto-detection.

## Development

```bash
pip install -e '.[dev]'   # ruff + pylint + pytest

pytest                    # run all tests
pytest tests/api/test_translate.py::test_submit_en_vi  # single test

ruff check . && ruff format .   # lint + format
make lint                       # ruff + pylint
```

## Makefile targets

| Target | Description |
|---|---|
| `make build` | Build Triton image (installs torch/transformers/optimum) |
| `make build-app` | Build API + worker image |
| `make build-ui` | Build Gradio UI image |
| `make export` | Export HF checkpoints to ONNX (runs inside Triton container) |
| `make up` / `make down` | Start / stop Triton + Redis |
| `make gateway-up` / `make gateway-down` | Start / stop Traefik gateway |
| `make api-up` / `make api-down` | Start / stop API + worker |
| `make ui-up` / `make ui-down` | Start / stop Gradio UI |
| `make observe` | Start Prometheus, Grafana, Loki, Promtail, exporters |
| `make all` | Build app/UI images and start all services |
| `make smoke` | Quick end-to-end translation test |
| `make stress` | Latency/throughput benchmark against Triton directly |
| `make eval` | BLEU/chrF quality eval on 5000 HF samples (Triton) |
| `make app-stress` | Benchmark through the FastAPI layer (`API_KEY=mykey`) |
| `make app-eval` | BLEU/chrF quality eval through FastAPI layer |
| `make logs` / `make api-logs` / `make worker-logs` / `make traefik-logs` | Tail service logs |
| `make prod-pull` / `make prod-up` / `make prod-down` | Production deploy from GHCR |
| `make clean` | Delete exported ONNX artifacts |
| `make remove` | Remove containers, volumes, images, and networks |

## Ports

| Service | Host port |
|---|---|
| Traefik HTTP (all external traffic) | 80 |
| Traefik dashboard | 8888 |
| Triton HTTP | 8000 |
| Triton gRPC | 8001 |
| Triton metrics | 8002 |
| Worker Prometheus metrics | 9091 |
| Prometheus | 9090 |
| Grafana | 3000 |
| Loki | 3100 |

The API (`8080`) has no host binding — traffic enters via Traefik on port 80.

## Configuration

All settings use the `ENVIT5_` prefix (pydantic-settings, can also be set via `.env`). Copy `.env.example` to `.env` to get started.

| Variable | Default | Description |
|---|---|---|
| `ENVIT5_API_KEYS` | — | Comma-separated API keys (required) |
| `ENVIT5_TRITON_HTTP_URL` | `localhost:8000` | Triton HTTP endpoint |
| `ENVIT5_REDIS_URL` | `redis://localhost:6379/0` | Result cache URL |
| `ENVIT5_CACHE_TTL_SECONDS` | `86400` | Translation cache TTL |
| `ENVIT5_MAX_NEW_TOKENS` | `512` | Generation limit |
| `ENVIT5_NUM_BEAMS` | `1` | Beam search width (1 = greedy) |
| `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD` | `admin` / `admin` | Grafana credentials |
