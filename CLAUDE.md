# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

English↔Vietnamese Neural Machine Translation served on **Triton Inference Server** (ONNX
Runtime + Python backend), fronted by a **Celery/Redis** queue and a **FastAPI** API. Full spec
and decisions live in [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) — read it before making
design changes.

**Build status:** Phases 1–4 are complete. `src/envit5/worker/` and `src/envit5/api/` are
implemented and tested. Prometheus metrics (Phase 4) are live: `src/envit5/core/metrics.py`
centralises all metric objects; the API exposes them at `/metrics` and the Celery worker
serves them on port 9091 via a background HTTP server. Still pending from Phase 4: structured
error responses, per-request retries/timeouts at the API layer.

## Architecture (the parts that span files)

- **One Triton Inference Server** hosts both translation models (`translator_en_vi` and
  `translator_vi_en`) from `model_repository/`. Each model is a Python backend that runs
  seq2seq generation via Optimum + ONNX Runtime. Adding a language pair = adding a new model
  directory under `model_repository/` and a new `triton_model_*` field in `Settings`.
- **Triton is called via `tritonclient.http`** in `src/envit5/worker/triton_client.py`.
  `translate_via_triton(text, model_name)` sends one chunk as a `BYTES` tensor and returns
  the decoded output string.
- **Dynamic batching** is configured per model in `config.pbtxt`
  (`max_queue_delay_microseconds: 100000`). Triton coalesces concurrent requests from all
  four Celery workers into batches up to `max_batch_size: 16`.
- **ONNX models are a build artifact** — export with `make export` before first run.
  `make clean` removes them (must run inside Docker; files are root-owned by the container).
- **Config is centralized** in `src/envit5/core/settings.py` (pydantic-settings, env prefix
  `ENVIT5_`). Don't read env vars ad hoc elsewhere; add a field there. Use `get_settings()`
  (the `@lru_cache` singleton) everywhere in application code — never instantiate `Settings()`
  directly.
- **Metrics are centralized** in `src/envit5/core/metrics.py`. All `Counter`/`Histogram`/`Gauge`
  objects are defined there and imported wherever they're incremented. The API mounts
  `make_metrics_asgi_app()` at `/metrics`; in the worker, `celery_app.py`'s `worker_init`
  signal starts `start_http_server(9091)`. The worker uses prometheus_client **multiprocess
  mode**: `PROMETHEUS_MULTIPROC_DIR=/tmp/prom_multiproc` is set in docker-compose so all four
  prefork children write mmap files that the main process aggregates on each scrape. The
  `worker_process_shutdown` signal calls `mark_process_dead` to clean up stale files. Do not
  skip `PROMETHEUS_MULTIPROC_DIR` when adding worker containers; without it, prefork children's
  metrics are invisible.
- **Task autodiscovery** — `celery_app.py` calls `celery_app.autodiscover_tasks(["envit5.worker"])`
  *after* `celery_app` is assigned as a module-level name. The call must come after assignment
  (not inside `_make_celery()`) because the lazy signal Celery fires imports `tasks.py`, which
  in turn does `from envit5.worker.celery_app import celery_app` — if `celery_app` isn't yet
  assigned in the module namespace, that import fails. If you add new task modules in other
  packages, add them to the `autodiscover_tasks` list. After any structural change here, always
  verify `[tasks]` in the worker startup log is non-empty.
- **API contract:** `POST /translate` (returns `job_id`) + `GET /jobs/{id}` (poll for result).
  Auth via `X-API-Key` header. Direction via `source`/`target` fields or the `direction`
  shorthand (`"en-vi"`); source auto-detected when both omitted (uses `py3langid`; only `en`
  and `vi` are accepted). Request schema lives in `src/envit5/api/models.py` — `direction` and
  `source`/`target` are mutually exclusive; providing both raises a 422.
- **Long-text chunking** — `src/envit5/worker/chunker.py` splits input at sentence boundaries
  into chunks ≤ 1 500 chars (≈ 350–420 tokens, safely under the model's 512-token ceiling).
  Paragraph structure (blank-line gaps) is preserved and reassembled after translation. The
  `chunk_text()` / `reassemble()` functions are the only place this logic lives — don't do
  ad-hoc splitting elsewhere.
- **Translation result caching** — the worker hashes `{src}-{tgt}:{text}` with SHA-256 and
  caches the result in Redis db/0 under key `trans:<digest>` for `ENVIT5_CACHE_TTL_SECONDS`
  (default 86 400 s = 24 h). Cache hits short-circuit Triton entirely.
- **Job-submitted marker** — on `POST /translate` the API writes a short-lived `jobtrack:<id>`
  key to Redis. `GET /jobs/{id}` uses this to distinguish "queued but not started" (Celery
  PENDING + marker present) from "unknown job" (Celery PENDING + marker absent, returns 404).
- **Gradio UI** — `src/envit5/ui/app.py` is a thin polling client over the REST API.
  Configured via `ENVIT5_API_URL` (default `http://localhost:8080`) and `ENVIT5_API_KEY`
  (default `changeme`). Start with `make ui-up`; runs on port 7860.
- **Generation parameters** — `ENVIT5_MAX_NEW_TOKENS` (default 512) and `ENVIT5_NUM_BEAMS`
  (default 1 = greedy) are passed through `Settings` and used inside `model.py` `.generate()`.
  Beam search can be enabled by bumping `ENVIT5_NUM_BEAMS`.

## Build / run loop

Triton requires ONNX models to be exported first — run `make export` once before `make up`.
Models are cached in the `hf-cache` Docker volume so repeat exports skip the HuggingFace download.

```bash
make build      # build the Triton image (large: installs torch-cpu/transformers/optimum)
make build-app  # build the API + worker image
make build-ui   # build the Gradio demo UI image
make export     # download HF checkpoints, export to ONNX into model_repository/*/1/onnx/
make up         # start Triton Inference Server + Redis in the background
make gateway-up # start Traefik API gateway (requires: make up)
make api-up     # start API + Celery worker (requires: make up && make gateway-up)
make ui-up      # start Gradio demo UI at http://localhost/ via Traefik (requires: make api-up)
make ready      # probe http://localhost:8000/v2/health/ready
make smoke      # translate a sample both ways (needs: pip install -e '.[client]')
make stress     # stress-test both models and print latency/throughput report
make eval       # BLEU/chrF quality eval on 5000 HF dataset samples (needs: pip install -e '.[eval]')
make app-stress # stress-test the FastAPI app (needs: make api-up; override key with API_KEY=mykey)
make app-eval   # quality eval through FastAPI app
make observe    # start full observability stack (Prometheus, Grafana, Loki, Promtail, node/DCGM exporters)
make logs        # tail Triton logs
make api-logs    # tail API logs
make worker-logs # tail Celery worker logs
make ui-logs     # tail Gradio UI logs
make traefik-logs # tail Traefik gateway logs
make ps         # show service status
make api-down   # stop API + worker only
make ui-down    # stop Gradio UI only
make down       # stop and remove containers
make remove     # stop and remove containers, volumes, images, and networks
make clean      # delete exported ONNX artifacts — runs inside Docker because files are root-owned
make all        # build-app + build-ui + up + gateway-up + api-up + ui-up + observe  (does NOT build Triton image or export ONNX models)
```

**Production deploy** (uses pre-built GHCR images, no local build needed):

```bash
make prod-pull IMAGE_TAG=v0.1.0  # pull triton/api/worker/ui images from GHCR
make prod-up   IMAGE_TAG=v0.1.0  # start all services (overlays docker-compose.prod.yml)
make prod-down                   # stop and remove prod containers
```

`make export` runs `scripts/export_models.py` *inside* the Triton image (so you don't need
torch locally). HuggingFace checkpoints are cached in a Docker named volume (`hf-cache`), so
repeat exports skip the download. To export only one direction: pass `--only translator_en_vi`
(or `translator_vi_en`) to `scripts/export_models.py` directly. Override model ids via
`ENVIT5_HF_MODEL_EN_VI` / `ENVIT5_HF_MODEL_VI_EN` (see `.env.example`). Bump
`TRITON_VERSION` if the default base image tag isn't on nvcr.io.

## Dev setup

```bash
pip install -e '.[dev]'      # ruff + pylint + pytest (includes api + worker + client)
pip install -e '.[client]'   # tritonclient + numpy (for smoke_test.py)
pip install -e '.[export]'   # ML deps — only on Python 3.10–3.12
pip install -e '.[eval]'     # datasets + sacrebleu + tritonclient (for stresstest.py --eval-dataset)
```

Run tests:

```bash
pytest                        # run all tests
pytest tests/api/             # run only api tests
pytest tests/api/test_translate.py::test_submit_en_vi  # run a single test
```

Lint and format:

```bash
ruff check .      # lint
ruff format .     # auto-format
pylint src/ tests/  # static analysis (or: make lint to run both)
```

## Conventions / gotchas

- **Exported ONNX is a build artifact**, git-ignored (`model_repository/**/onnx/`). Never commit
  it; regenerate with `make export`. `model.py` + `config.pbtxt` are the tracked source.
  `make clean` must run inside Docker because the files are written by a root-owned container.
- **Gateway ports:** Traefik `80` (HTTP entry), `8888` (dashboard — dev only at `http://localhost:8888/dashboard/`).
  Static config: `docker/traefik/traefik.yml`; middleware definitions: `docker/traefik/dynamic/middlewares.yml`.
  Rate limiting (100 req/s burst 50) and gzip compression are applied on all API routes.
  `X-API-Key` auth stays in FastAPI — it is application-level, not a gateway concern.
  `/metrics` is NOT routed through Traefik; Prometheus scrapes services directly over the Docker network.
  Traefik exposes its own metrics at `traefik:8899` (internal, not host-mapped).
- **Triton ports:** `8000` HTTP, `8001` gRPC, `8002` Prometheus metrics.
- **App ports:** API `8080` (internal only — no host binding), worker Prometheus metrics `9091`.
- **Observability ports:** Prometheus `9090`, Grafana `3000`, Loki `3100`. Grafana default
  credentials `admin/admin` (override via `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD`).
  The `envit5-app` Grafana dashboard (uid `envit5-app`) is auto-provisioned from
  `docker/grafana/provisioning/dashboards/envit5-app.json`.
- GPU support requires the **NVIDIA driver + NVIDIA Container Toolkit** on the host. The
  compose file passes `count: 1` GPU to Triton; both `config.pbtxt` files use `KIND_GPU`.
  `onnxruntime-gpu` (in the Dockerfile) picks up `CUDAExecutionProvider` automatically.
  CUDA torch (`cu124` wheel) is required for Optimum IO binding (GPU output buffer allocation).
- Host Python is 3.14, but ML deps run in containers; the export venv (if run locally instead of
  via Docker) should be Python 3.10–3.12 where torch/transformers wheels are reliable.
- **Redis DB layout:** `db/0` = result cache, `db/1` = Celery broker, `db/2` = Celery result
  backend. All three URLs are configurable via `ENVIT5_REDIS_URL` / `ENVIT5_CELERY_*`.
- `ENVIT5_API_KEYS` accepts a comma-separated string (e.g. `key1,key2`) or a JSON list. If the
  env var is unset the list is empty and **all requests are rejected** (the auth check treats an
  empty key list as "no valid keys configured" rather than "auth disabled").
- **Test isolation:** `tests/conftest.py` patches all service URLs to safe defaults and calls
  `get_settings.cache_clear()` before and after every test — required because `get_settings` is
  `@lru_cache`. Tests mock `translate_task.delay` so they never need a running broker.
- **`make stress` vs `make eval`:** `stress` measures latency/throughput with synthetic load;
  `eval` runs `stresstest.py --eval-dataset` against the HuggingFace dataset and reports BLEU/chrF.
  Both have `app-*` counterparts that drive the FastAPI layer instead of Triton directly.
- **GitHub Actions:** `.github/workflows/build.yml` builds and pushes Docker images to GHCR on
  merge to `main`; `.github/workflows/tag.yml` auto-tags the release on `main`.
