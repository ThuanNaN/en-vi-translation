# PolyglotHub — Model- and engine-agnostic translation/serving hub

Model- and engine-agnostic translation/serving hub — pluggable serving backends, async job dispatch, and a production-ready API gateway.

```
                              ┌─────────────────────────────────────────────────────────────────┐
                              │                       Traefik  :80                               │
                              └─────────────────────────┬───────────────────────────────────────┘
                                                         │
                    POST /translate                       ▼                     enqueue
client ────────────────────────────────────────► FastAPI (api:8080) ──────────────────────────► Redis :6379
   ▲                                                     │                                      │  ├─ db/1  broker
   │  GET /jobs/{id}                                     │ 202 job_id                           │  ├─ db/2  result backend
   └─────────────────────────────────────────────────────┘                                      │  └─ db/0  trans cache + job markers
                                                                                                 │
                                                                                                 ▼
                                                                                      Celery worker (worker:9091)
                                                                                                 │
                                                                                  cache hit?      │
                                                                          Redis ◄────────────────┤
                                                                                                 │ cache miss
                                                                                                 ▼
                                                                                         BackendClient
                                                                              ┌──────────────────────────────┐
                                                                              │  Triton  :8000   NMT  (ONNX) │
                                                                              │  vLLM    :8000   LLM  (Qwen) │
                                                                              │  Custom  :9000   any engine  │
                                                                              │  HF      —       (future)    │
                                                                              └──────────────────────────────┘

Observability
  Prometheus :9090 ◄── /metrics ── api:8080 · worker:9091 · triton:8002
       └──► Grafana :3000
       └──► Loki    :3100 ◄── Promtail (log aggregation)
```

## Repo layout

```
apps/
├── api/              # FastAPI + Celery worker (polyglot_gateway package)
│   └── src/polyglot_gateway/
│       ├── api/      # FastAPI routes and request/response models
│       ├── inference/  # BackendClient adapters: triton, vllm, custom, ...
│       ├── worker/   # Celery app, tasks, long-text chunker
│       └── core/     # Settings (pydantic), Prometheus metrics
└── ui/               # Gradio demo frontend (polyglot_ui package)

inference/
├── triton/           # Triton Inference Server + ONNX model_repository
├── vllm/config/      # vLLM configuration docs
└── custom/
    └── example_service/  # Minimal FastAPI reference implementation

infra/
├── docker-compose.yml     # main Compose file
├── docker-compose.prod.yml
├── traefik/          # reverse proxy config
└── observability/    # Prometheus, Grafana, Loki, Promtail

configs/
├── models.yaml       # backend registry (loaded when BACKENDS env is not set)
└── services.yaml     # service endpoint reference
```

## Key design decisions

- **Backend registry** — add a new backend by editing `configs/models.yaml`. No code change
  needed for new language pairs. The `BACKENDS` env var overrides the YAML file for production.

  ```yaml
  # configs/models.yaml
  backends:
    en-vi:
      type: triton
      url: triton:8000
      model_name: translator_en_vi
    llm:
      type: vllm
      url: vllm:8000
      model_name: Qwen/Qwen3.5-0.8B
  ```

  Lookup order for a request with `model="llm"`: `"src-tgt:llm"` → `"src-tgt"` → `"llm"`.

- **Pluggable adapters** — `apps/api/src/polyglot_gateway/inference/` holds one file per engine
  type. Each implements the three-line `BackendClient` Protocol. Adding a new engine type means
  adding one file + one case in `registry.py` (see [Adding a new engine](#adding-a-new-model-or-engine)).

- **Async API** — `POST /translate` enqueues a job and returns a `job_id`; `GET /jobs/{id}` polls for the result. Auth via `X-API-Key` header.

- **Long-text chunking** — input is split at sentence boundaries into ≤ 1 500-char chunks and reassembled after translation.

- **Redis caching** — results are cached by `sha256(direction:text)` for 24 h, short-circuiting the backend entirely on hits.

- **Traefik gateway** — rate-limited (100 req/s, burst 50) and gzip-compressed.

- **Prometheus metrics** — HTTP rate/latency, translation throughput/latency, cache hit rate, active-job gauge. Grafana dashboards auto-provisioned.

## Quick start

```bash
# 1. Build images and export ONNX models (one-time; downloads ~500 MB of HF weights)
make build-triton   # Triton image (~30 min, installs torch + optimum)
make build-app      # API + worker image
make export         # export HF checkpoints → ONNX inside Triton container

# 2. Start infrastructure + inference engines
make infra-up       # Redis + Traefik
make triton-up      # Triton Inference Server
make vllm-up        # vLLM / Qwen3.5-0.8B (optional; needs GPU)

# 3. Start backend API
make api-up         # FastAPI + Celery worker

# 4. Start optional services
make ui-up          # Gradio demo at http://localhost/
make observe        # Prometheus + Grafana + Loki + Promtail

# 5. Smoke-test and stress-test
make triton-ready          # probe Triton readiness
make smoke                 # translate a sample sentence via Triton directly
make app-stress-all API_KEY=changeme   # full app stress: NMT + LLM combined report

# 6. Open
#   API docs  → http://localhost/docs
#   Gradio UI → http://localhost/
#   Grafana   → http://localhost:3000  (admin / admin)
#   Traefik   → http://localhost:8888/dashboard/
```

Start everything in one shot (skips Triton build and ONNX export):

```bash
make all
```

Run `make help` for a full list of targets grouped by section.

## Production deploy

Uses pre-built GHCR images — each service versions independently:

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

| Field | Required | Description |
|---|---|---|
| `text` | yes | Text to translate |
| `source` + `target` | no† | ISO language codes (`"en"`, `"vi"`, …) |
| `direction` | no† | Shorthand: `"en-vi"` (mutually exclusive with `source`/`target`) |
| `model` | no | Backend selector — omit for NMT (Triton), `"llm"` for Qwen/vLLM |

† If both are omitted, the source language is auto-detected.

## Stress testing

`scripts/app_stresstest.py` exercises the FastAPI app against both backends in a single run and prints a combined latency/throughput report.

```bash
# Both backends — NMT (Triton) + LLM (vLLM) — requires: api-up + vllm-up
make app-stress-all API_KEY=mykey

# Individual backend targets
make app-stress      API_KEY=mykey   # NMT only   (stresstest.py --target app)
make app-stress-llm  API_KEY=mykey   # LLM only   (stresstest.py --target app --model llm)
make vllm-stress                     # vLLM directly via OpenAI API

# Direct Triton (bypasses FastAPI/Celery)
make stress                          # latency/throughput report
make bench                           # GPU saturation: 5000 req, 64 workers, long texts

# Tune load for app-stress-all
python scripts/app_stresstest.py \
    --api-url http://localhost:80 --api-key mykey \
    --nmt-requests 200 --nmt-concurrency 16 \
    --llm-requests  50 --llm-concurrency  4 \
    --text-size medium \
    --results-file results/app_stress.json
```

`app_stresstest.py` uses higher concurrency for NMT (fast ONNX seq2seq) and lower for LLM (autoregressive). Run `python scripts/app_stresstest.py --help` for all options.

## Development

```bash
# From apps/api/
pip install -e '.[dev]'    # ruff + pylint + pytest + api + worker deps

pytest                                                      # all tests
pytest tests/api/test_translate.py::test_submit_en_vi      # single test

ruff check . && ruff format .                               # lint + format
make lint                                                   # ruff check + pylint via Makefile
```

## Configuration

Copy `.env.example` to `.env` to get started.

| Variable | Default | Description |
|---|---|---|
| `API_KEYS` | — | Comma-separated API keys (**required**; empty = all requests rejected) |
| `BACKENDS` | _(see models.yaml)_ | JSON map of route key → backend config (overrides YAML) |
| `BACKENDS_CONFIG` | `configs/models.yaml` | Path to YAML backend registry |
| `REDIS_URL` | `redis://localhost:6379/0` | Result cache |
| `CELERY_BROKER_URL` | `redis://localhost:6379/1` | Task queue |
| `CACHE_TTL_SECONDS` | `86400` | Translation cache TTL (24 h) |
| `MAX_NEW_TOKENS` | `512` | Generation token limit (Triton) |
| `NUM_BEAMS` | `1` | Beam search width (1 = greedy) |
| `HF_TOKEN` | — | HuggingFace token (needed if model is gated) |
| `GRAFANA_ADMIN_PASSWORD` | `admin` | Grafana password |

## Ports

| Service | Host port |
|---|---|
| Traefik HTTP (all external traffic) | 80 |
| Traefik dashboard | 8888 |
| Triton HTTP | 8000 |
| Triton gRPC | 8001 |
| Triton metrics | 8002 |
| vLLM OpenAI-compatible API | 8010 |
| Worker Prometheus metrics | 9091 |
| Prometheus | 9090 |
| Grafana | 3000 |
| Loki | 3100 |

The API (`8080`) has no host binding — all traffic enters via Traefik on port 80.

## Adding a new model or engine

### Option A: New language pair on an existing engine

1. Edit `configs/models.yaml` — add a new `backends:` entry:
   ```yaml
   en-fr:
     type: triton
     url: triton:8000
     model_name: translator_en_fr
   ```
2. If Triton: add the model directory under `inference/triton/model_repository/` and re-export (`make export`).
3. Rebuild and restart: `make build-app && make api-up`.

### Option B: New direction on vLLM (no config change needed)

The existing `"llm"` backend in `configs/models.yaml` is direction-agnostic — it handles any
`src`/`tgt` pair. Just POST with `"model": "llm"` and the source/target you need.

### Option C: New inference engine type

1. **Add the server** — put it in `inference/<engine-name>/`. For a custom FastAPI service,
   copy `inference/custom/example_service/` as a starting point. It must expose:
   ```
   POST /translate  {"text":…, "src":…, "tgt":…}  → {"translation":…}
   GET  /health                                     → {"status":"ok"}
   ```

2. **Add the adapter** — create `apps/api/src/polyglot_gateway/inference/<engine>.py`:
   ```python
   class MyEngineBackendClient:
       def __init__(self, url: str) -> None: ...
       def translate(self, text: str, model_name: str, src: str = "", tgt: str = "") -> str: ...
   ```

3. **Register it** — add a case in `apps/api/src/polyglot_gateway/inference/registry.py`:
   ```python
   if config.type == "my-engine":
       return MyEngineBackendClient(url=config.url)
   ```

4. **Add the type** — extend `BackendConfig.type` in `apps/api/src/polyglot_gateway/core/settings.py`.

5. **Wire in Compose** — add a service block to `infra/docker-compose.yml` (see the commented
   `custom-service` example at the bottom), add it to `configs/models.yaml`, and rebuild:
   ```bash
   make build-app && make build-custom && make api-up && make custom-up
   ```
