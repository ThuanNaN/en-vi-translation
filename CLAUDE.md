# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multi-language Neural Machine Translation platform served on **Triton Inference Server** (ONNX
Runtime + Python backend), fronted by a **Celery/Redis** queue and a **FastAPI** API. The
architecture is designed to scale across serving backends (Triton, vLLM, HuggingFace) and
language pairs without code changes.

## Repo layout (3-layer + infra)

```
services/
├── gateway/          # Layer 2 — FastAPI + Celery worker + backend abstraction
│   ├── src/polyglot_gateway/
│   │   ├── api/      # FastAPI app, request/response models
│   │   ├── worker/   # Celery app, tasks, chunker, backend clients
│   │   │   └── backends/  # BackendClient abstraction (triton, vllm, ...)
│   │   └── core/     # Settings, Prometheus metrics
│   └── tests/
├── ui/               # Layer 1 — Gradio demo frontend
│   └── src/polyglot_ui/
└── serving/
    └── triton/       # Layer 3 — Triton Inference Server + model_repository
        ├── model_repository/   # translator_en_vi/, translator_vi_en/
        ├── scripts/            # export_models.py, smoke_test.py
        └── Dockerfile

infra/
├── compose/          # docker-compose.yml + docker-compose.prod.yml
├── traefik/          # traefik.yml + dynamic/middlewares.yml
├── observability/    # prometheus.yml, loki.yml, promtail.yml, grafana/
└── k8s/              # (future Kubernetes manifests)

scripts/              # stresstest.py (end-to-end, targets running stack)
Makefile              # top-level orchestration (COMPOSE points to infra/compose/)
```

## Architecture (the parts that span files)

- **Backend registry** — the gateway maps route keys to serving backends via `BACKENDS` (JSON).
  Key format: `"src-tgt"` (default NMT) or `"src-tgt:model"` for named variants (e.g. `"en-vi:llm"`).
  `Settings.backend_for(src, tgt, model=None)` resolves the key; the worker calls
  `make_backend_client(config)` to get the right `BackendClient` implementation.
  Adding a language pair or backend variant = env var change, no code.

  ```bash
  # Triton NMT (default) + one vLLM entry covering all directions:
  BACKENDS='{
    "en-vi": {"type":"triton","url":"triton:8000","model_name":"translator_en_vi"},
    "vi-en": {"type":"triton","url":"triton:8000","model_name":"translator_vi_en"},
    "llm":   {"type":"vllm","url":"vllm:8000","model_name":"Qwen/Qwen3.5-0.8B"}
  }'
  ```

  Lookup order for `backend_for(src, tgt, model)`: `"src-tgt:model"` → `"src-tgt"` → `"model"`.
  The `"llm"` key acts as a direction-agnostic fallback — `src`/`tgt` are passed directly to
  `VLLMBackendClient.translate()` where they are resolved to language names for the prompt.

- **BackendClient Protocol** — `services/gateway/src/polyglot_gateway/worker/backends/base.py`
  defines `BackendClient(Protocol)` with one method: `translate(text, model_name) -> str`.
  `backends/triton.py` implements it with `tritonclient.http`; `backends/vllm.py` implements it
  with a plain `urllib.request` POST to `/v1/chat/completions` (no `openai` package needed).
  `BackendClient.translate(text, model_name, src, tgt)` — `src`/`tgt` language codes are
  passed through from the task so `VLLMBackendClient` can build a direction-aware prompt
  without any per-direction config; Triton ignores `src`/`tgt`.
  `BackendConfig.type` also accepts `"hf"` but `registry.py:make_backend_client()` raises
  `ValueError` for it — reserved for a future HuggingFace direct backend. Add new backends by
  adding a file here and a case in `backends/registry.py:make_backend_client()`.

- **Config is centralized** in `services/gateway/src/polyglot_gateway/core/settings.py`
  (pydantic-settings, no env prefix). Use `get_settings()` (the `@lru_cache` singleton)
  everywhere — never instantiate `Settings()` directly. Supported language pairs come from
  `settings.backends` keys — there is no separate `_SUPPORTED_PAIRS` list.

- **Metrics are centralized** in `services/gateway/src/polyglot_gateway/core/metrics.py`.
  All `Counter`/`Histogram`/`Gauge` objects are defined there and imported wherever incremented.
  The API mounts `make_metrics_asgi_app()` at `/metrics`; the worker starts a Prometheus HTTP
  server on port 9091 (`METRICS_PORT` env var — the one exception to the Settings rule,
  read via `os.environ` in `celery_app.py` before full initialization). Prometheus multiprocess
  mode: `PROMETHEUS_MULTIPROC_DIR=/tmp/prom_multiproc` must be set for worker containers.

- **Task autodiscovery** — `celery_app.py` calls `autodiscover_tasks(["polyglot_gateway.worker"])`
  *after* `celery_app` is assigned as a module-level name (circular-import safety). Always
  verify `[tasks]` in the worker startup log is non-empty after structural changes.

- **Celery task name is `polyglot.translate`** — do not rename; in-flight jobs in Redis would break.

- **API contract:**
  - `GET /health` — unauthenticated liveness probe, always returns `{"status": "ok"}`.
  - `POST /translate` — returns `{"job_id": "..."}` (202 Accepted). Auth via `X-API-Key` header.
    Direction via `source`/`target` fields or the `direction` shorthand (`"en-vi"`); source
    auto-detected when both omitted (uses `py3langid`). `direction` and `source`/`target` are
    mutually exclusive (422 if both). Optional `model` field selects a backend variant
    (`"llm"` → vLLM, omit → default NMT). Validation calls `settings.backend_for()` — unknown
    language-pair + model combinations return 422.
  - `GET /jobs/{id}` — poll for result. Status values: `pending`, `started`, `done`, `failed`.

- **Long-text chunking** — `services/gateway/src/polyglot_gateway/worker/chunker.py` splits input
  at sentence boundaries into chunks ≤ 1 500 chars. `chunk_text()` / `reassemble()` are the
  only place this logic lives.

- **Translation result caching** — SHA-256 of `{src}-{tgt}:{text}` → Redis key `trans:<digest>`,
  TTL `CACHE_TTL_SECONDS` (default 24 h). Cache hits short-circuit the backend entirely.

- **Job-submitted marker** — `POST /translate` writes `jobtrack:<id>` to Redis.
  `GET /jobs/{id}` uses it to distinguish "queued" (Celery PENDING + marker present) from
  "unknown job" (Celery PENDING + marker absent → 404).

- **Triton models** — `services/serving/triton/model_repository/` holds Python backends that
  run seq2seq via Optimum + ONNX Runtime. Dynamic batching: `max_queue_delay_microseconds:
  100000`, `max_batch_size: 16`. ONNX is a build artifact — export with `make export` before
  first run.

- **Gradio UI** — `services/ui/src/polyglot_ui/app.py` is a thin polling client over the REST
  API, configured via `API_URL` and `API_KEY`.

- **Task time limits coupling** — `celery_app.py` sets `task_soft_time_limit=25` and
  `task_time_limit=30`; `settings.py` has `request_timeout_seconds=30.0`. These are
  intentionally matched. If you change `request_timeout_seconds`, update `celery_app.py` too.

## Build / run loop

Triton requires ONNX models exported first — run `make export` once before `make up`. All
`make` commands use `--project-directory .` so paths in `infra/compose/docker-compose.yml`
are always relative to the repo root.

```bash
make build      # build the Triton image (large: installs torch/transformers/optimum)
make build-app  # build the gateway image (API + worker)
make build-ui   # build the Gradio UI image
make export     # export HF checkpoints to ONNX into services/serving/triton/model_repository/*/1/onnx/
make up         # start Triton + Redis
make gateway-up # start Traefik API gateway
make api-up     # start API + Celery worker (requires: make up && make gateway-up)
make ui-up      # start Gradio demo UI at http://localhost/ (requires: make api-up)
make ready      # probe Triton readiness
make smoke      # translate a sample both ways
make stress     # stress-test both models (latency/throughput)
make bench      # GPU-saturation benchmark: async, long texts, high concurrency
make eval       # BLEU/chrF quality eval on 5000 HF samples (needs: pip install -e '.[eval]')
make app-stress # stress-test through FastAPI (needs: make api-up; override with API_KEY=mykey)
make app-eval   # quality eval through FastAPI
make observe    # start Prometheus + Grafana + Loki + Promtail + exporters
make logs       # tail Triton logs
make api-logs   # tail API logs
make worker-logs # tail Celery worker logs
make ui-logs    # tail Gradio UI logs
make traefik-logs # tail Traefik logs
make ps         # show service status
make api-down   # stop API + worker only
make gateway-down # stop Traefik
make ui-down    # stop Gradio UI
make vllm-up    # start vLLM (Qwen/Qwen3.5-0.8B) on host port 8010 (requires GPU)
make vllm-down  # stop vLLM
make vllm-logs  # tail vLLM logs
make down       # stop and remove containers
make remove     # stop + remove containers, volumes, networks, and images
make clean      # delete ONNX artifacts (runs inside Docker — root-owned files)
make all        # build-app + build-ui + up + gateway-up + api-up + ui-up + observe
```

**Production deploy** (separate image tags per service repo):

```bash
make prod-pull GATEWAY_TAG=v0.2.0 UI_TAG=v0.1.1 TRITON_TAG=v0.1.0
make prod-up   GATEWAY_TAG=v0.2.0 UI_TAG=v0.1.1 TRITON_TAG=v0.1.0
make prod-down
```

`make export` runs `scripts/export_models.py` *inside* the Triton container. HuggingFace
checkpoints are cached in the `hf-cache` Docker volume so repeat exports skip the download.
Bump `TRITON_VERSION` if the base image tag isn't on nvcr.io.

## Dev setup

Copy `.env.example` to `.env` and fill in secrets before first run.

**Gateway** (`services/gateway/`):

```bash
pip install -e '.[dev]'    # ruff + pylint + pytest (includes api + worker deps)
pip install -e '.[worker]' # tritonclient + numpy + celery (for direct Triton calls)
```

**UI** (`services/ui/`):

```bash
pip install -e .           # gradio + requests
```

Run tests (from `services/gateway/`):

```bash
pytest                                              # all tests
pytest tests/api/                                   # API tests only
pytest tests/api/test_translate.py::test_submit_en_vi  # single test
```

Lint (`make lint` runs `ruff check` + `pylint` only; run `ruff format` separately):

```bash
ruff check .               # from services/gateway/, or: make lint
pylint src/ tests/
ruff format .              # not included in make lint — run manually
```

For `make stress` / `make bench` (direct Triton calls): `pip install -e '.[worker]'` from
`services/gateway/` satisfies the `tritonclient[http]` + `numpy` requirement.
For `make eval` / `make app-eval`: additionally `pip install datasets sacrebleu`.

## Conventions / gotchas

- **Exported ONNX is a build artifact**, git-ignored (`services/serving/triton/**/onnx/`).
  `make clean` must run inside Docker because files are written by a root-owned container.
- **`BACKENDS` JSON** is the single source of truth for which language pairs are live and
  which backend serves each. The API's 422 validation reads from this at request time.
- **Gateway ports:** Traefik `80` (HTTP entry), `8888` (dashboard). Static config:
  `infra/traefik/traefik.yml`; rate-limit + gzip: `infra/traefik/dynamic/middlewares.yml`.
  `/metrics` is NOT routed through Traefik — Prometheus scrapes services directly.
- **Triton ports:** `8000` HTTP, `8001` gRPC, `8002` Prometheus metrics.
- **App ports:** API `8080` (internal, Traefik-routed), worker metrics `9091`.
- **Observability ports:** Prometheus `9090`, Grafana `3000`, Loki `3100`. Dashboards are
  auto-provisioned from `infra/observability/grafana/provisioning/`. Grafana credentials:
  `admin/admin` (override via `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD`).
- **Redis DB layout:** `db/0` = result cache + job markers, `db/1` = Celery broker,
  `db/2` = Celery result backend. Redis key prefixes `trans:` and `jobtrack:` must not change.
- `API_KEYS` accepts a comma-separated string (`key1,key2`) or a JSON list. Empty list
  = **all requests rejected**.
- **Test isolation:** `services/gateway/tests/conftest.py` patches `BACKENDS` (as JSON)
  and Redis URLs, then calls `get_settings.cache_clear()` before/after each test.
  Tests mock `translate_task.delay` — no broker needed.
- **GPU support** requires NVIDIA driver + Container Toolkit. Compose passes `count: 1` GPU to
  Triton; `config.pbtxt` uses `KIND_GPU`. CPU-only mode: change `KIND_GPU` to `KIND_CPU`.
- **GitHub Actions:** `.github/workflows/build.yml` builds gateway/UI/Triton images → GHCR;
  `.github/workflows/tag.yml` auto-tags on push to `main`.
- **`triton_infer_duration_seconds` metric** is defined in `core/metrics.py` but currently
  has no incrementer — `triton_client.py` was removed during the backend-abstraction refactor.
  Wire it up in `backends/triton.py` if per-chunk Triton latency tracking is needed.
