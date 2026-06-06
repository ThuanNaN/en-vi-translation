# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

English↔Vietnamese Neural Machine Translation served on **Triton Inference Server**,
fronted by a **Celery/Redis** queue and a **FastAPI** API. Full spec and decisions live in
[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) — read it before making design changes.

**Build status:** Phases 1–4 are complete. `src/envit5/worker/` and `src/envit5/api/` are
implemented; all 18 API tests pass. Prometheus metrics (Phase 4) are live: `src/envit5/core/metrics.py`
centralises all metric objects; the API exposes them at `/metrics` and the Celery worker
serves them on port 9091 via a background HTTP server. Still pending from Phase 4: structured
error responses, per-request retries/timeouts at the API layer.

## Architecture (the parts that span files)

- **Two single-direction Triton models**, not one bidirectional model:
  `translator_en_vi` ← `Helsinki-NLP/opus-mt-en-vi`, `translator_vi_en` ← `Helsinki-NLP/opus-mt-vi-en`.
  Direction is selected by *routing to the right model*, so adding a language pair = adding a
  model dir + its HF id (in `scripts/export_models.py` and `Settings.model_name_for`).
- **Serving is the Triton Python backend, not the onnxruntime backend.** Each
  `model_repository/<name>/1/model.py` loads Optimum `ORTModelForSeq2SeqLM` and calls
  `.generate()`. This is deliberate: the plain onnxruntime backend does a single forward pass
  and can't run seq2seq autoregressive decoding. Heavy deps (torch-cpu, transformers, optimum,
  sentencepiece) therefore live in **`docker/triton.Dockerfile`**, not in the host env.
- **Models are string-in / string-out** (`INPUT_TEXT`/`OUTPUT_TEXT`, `TYPE_STRING`).
  Tokenization and decoding happen inside `model.py`, so callers only pass plain text. If you
  change the I/O contract, update `model.py`, both `config.pbtxt`, and `scripts/smoke_test.py`
  together.
- **The two `model.py` files are intentionally identical** — the sibling `onnx/` directory is
  what makes an instance translate a given direction. Keep them in sync (or refactor to a shared
  module mounted into both) if you edit one.
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
  shorthand (`"en-vi"`); source auto-detected when both omitted.

## Build / run loop

Order matters — export must populate `model_repository/*/1/onnx/` before Triton can load:

```bash
make build      # build the Triton image (large: installs torch-cpu/transformers/optimum)
make build-app  # build the API + worker image
make export     # download HF checkpoints, export to ONNX into model_repository/*/1/onnx/
make up         # start Triton + Redis in the background
make api-up     # start API + Celery worker (requires: make up)
make ready      # probe http://localhost:8000/v2/health/ready
make smoke      # translate a sample both ways (needs: pip install -e '.[client]')
make stress     # stress-test both models and print latency/throughput report
make eval       # BLEU/chrF quality eval on 5000 HF dataset samples (needs: pip install -e '.[eval]')
make app-stress # stress-test the FastAPI app (needs: make api-up; override key with API_KEY=mykey)
make app-eval   # quality eval through FastAPI app
make observe    # start full observability stack (Prometheus, Grafana, Loki, Promtail, node/DCGM exporters)
make logs       # tail Triton logs
make api-logs   # tail API logs
make worker-logs # tail Celery worker logs
make ps         # show service status
make api-down   # stop API + worker only
make down       # stop and remove containers
make remove     # stop and remove containers, volumes, images, and networks
make clean      # delete exported ONNX artifacts — runs inside Docker because files are root-owned
make all        # build + build-app + export + up + api-up + observe
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
- **Triton ports:** `8000` HTTP, `8001` gRPC, `8002` Prometheus metrics.
- **App ports:** API `8080`, worker Prometheus metrics `9091`.
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
- `ENVIT5_API_KEYS` accepts a comma-separated string (e.g. `key1,key2`) or a JSON list.
- **Test isolation:** `tests/conftest.py` patches all service URLs to safe defaults and calls
  `get_settings.cache_clear()` before and after every test — required because `get_settings` is
  `@lru_cache`. Tests mock `translate_task.delay` so they never need a running broker.
- **`make stress` vs `make eval`:** `stress` measures latency/throughput with synthetic load;
  `eval` runs `stresstest.py --eval-dataset` against the HuggingFace dataset and reports BLEU/chrF.
  Both have `app-*` counterparts that drive the FastAPI layer instead of Triton directly.
