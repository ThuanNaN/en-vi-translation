# Polyglot

Multilingual Neural Machine Translation platform — pluggable serving backends, async job dispatch, and a production-ready API gateway.

```
                      Traefik :80
                      ┌──────────────────────────────────────────────┐
client ─POST /translate─▶ FastAPI ──enqueue──▶ Redis (broker) ──▶ Celery worker
   ▲                         │ (returns job_id)                         │
   └──GET /jobs/{id}─────────┘                                          ▼
                 Redis (result cache + Celery backend)         BackendClient
                                                              ┌────────────────┐
  Prometheus ◀─ /metrics (api:8080, worker:9091, triton:8002) │ Triton  :8000  │
       └──▶ Grafana :3000                                     │ vLLM    :8000  │ ← future
                                                              │ HF      :8000  │ ← future
                                                              └────────────────┘
```

## Repo layout

```
services/
├── gateway/          # FastAPI + Celery worker (polyglot_gateway package)
├── ui/               # Gradio demo frontend (polyglot_ui package)
└── serving/
    └── triton/       # Triton Inference Server + ONNX model_repository

infra/
├── compose/          # docker-compose.yml + docker-compose.prod.yml
├── traefik/          # reverse proxy config
└── observability/    # Prometheus, Grafana, Loki, Promtail
```

## Key design decisions

- **Backend registry** — `BACKENDS` (JSON env var) maps each language pair to a serving backend. Adding a language pair or swapping backends requires no code change:

  ```bash
  BACKENDS='{"en-vi":{"type":"triton","url":"triton:8000","model_name":"translator_en_vi"},
             "vi-en":{"type":"vllm","url":"vllm:8000","model_name":"Helsinki-NLP/opus-mt-vi-en"}}'
  ```

- **Async API** — `POST /translate` enqueues a job and returns a `job_id`; `GET /jobs/{id}` polls for the result. Auth via `X-API-Key` header.

- **Long-text chunking** — input is split at sentence boundaries into ≤ 1 500-char chunks and reassembled after translation.

- **Redis caching** — results are cached by `sha256(direction:text)` for 24 h, short-circuiting the backend entirely on hits.

- **Traefik gateway** — rate-limited (100 req/s, burst 50) and gzip-compressed.

- **Prometheus metrics** — HTTP rate/latency, translation throughput/latency, cache hit rate, active-job gauge. Grafana dashboards auto-provisioned.

## Quick start

```bash
# 1. Build images and export ONNX models (one-time; downloads ~500 MB of HF weights)
make build        # Triton image (~30 min, installs torch + optimum)
make build-app    # gateway image (API + worker)
make export       # export HF checkpoints to ONNX (runs inside Triton container)

# 2. Start all services
make up           # Triton + Redis
make gateway-up   # Traefik
make api-up       # FastAPI + Celery worker
make ui-up        # Gradio demo (optional)
make observe      # Prometheus + Grafana + Loki (optional)

# 3. Smoke-test
make smoke        # translates a sample sentence via Triton directly

# 4. Open
#   API docs  → http://localhost/docs
#   Gradio UI → http://localhost/
#   Grafana   → http://localhost:3000  (admin / admin)
#   Traefik   → http://localhost:8888/dashboard/
```

Start everything in one shot (skips Triton build and ONNX export):

```bash
make all
```

## Production deploy

Uses pre-built GHCR images — each service repo versions independently:

```bash
make prod-pull GATEWAY_TAG=v0.2.0 UI_TAG=v0.1.1 TRITON_TAG=v0.1.0
make prod-up   GATEWAY_TAG=v0.2.0 UI_TAG=v0.1.1 TRITON_TAG=v0.1.0
make prod-down
```

## API

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/translate` | POST | `X-API-Key` | Submit translation job, returns `{ job_id }` |
| `/jobs/{job_id}` | GET | `X-API-Key` | Poll status: `pending` / `started` / `done` / `failed` |
| `/health` | GET | — | Liveness check |
| `/metrics` | GET | — | Prometheus metrics (not routed through Traefik) |

**Request body for `POST /translate`:**
```json
{ "text": "Hello world", "source": "en", "target": "vi" }
```
Or use the `direction` shorthand (`"en-vi"`), or omit both for auto-detection.

## Development

```bash
# From services/gateway/
pip install -e '.[dev]'    # ruff + pylint + pytest + api + worker deps

pytest                                                      # all tests
pytest tests/api/test_translate.py::test_submit_en_vi      # single test

ruff check . && ruff format .                               # lint + format
make lint                                                   # ruff + pylint via Makefile
```

## Configuration

Copy `.env.example` to `.env` to get started.

| Variable | Default | Description |
|---|---|---|
| `API_KEYS` | — | Comma-separated API keys (**required**; empty = all requests rejected) |
| `BACKENDS` | Triton en↔vi | JSON map of language pair → backend config |
| `REDIS_URL` | `redis://localhost:6379/0` | Result cache |
| `CELERY_BROKER_URL` | `redis://localhost:6379/1` | Task queue |
| `CACHE_TTL_SECONDS` | `86400` | Translation cache TTL (24 h) |
| `MAX_NEW_TOKENS` | `512` | Generation token limit |
| `NUM_BEAMS` | `1` | Beam search width (1 = greedy) |
| `GRAFANA_ADMIN_PASSWORD` | `admin` | Grafana password |

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

The API (`8080`) has no host binding — all traffic enters via Traefik on port 80.

## Adding a new language pair or backend

1. Deploy a serving container (Triton, vLLM, or HuggingFace Inference).
2. Add an entry to `BACKENDS` — no gateway code change needed.
3. (Triton only) Add model directory under `services/serving/triton/model_repository/` and run `make export`.
