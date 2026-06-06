# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

English↔Vietnamese Neural Machine Translation served on **Triton Inference Server**,
fronted by a **Celery/Redis** queue and a **FastAPI** API. Full spec and decisions live in
[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) — read it before making design changes.

**Build status:** Phase 2 (model + Triton serving) is scaffolded. The queue (Celery) and
API (FastAPI) layers — `src/envit5/worker/` and `src/envit5/api/` — are **not implemented
yet** (see the roadmap in REQUIREMENTS.md). `src/envit5/core/settings.py` already carries
config for those later phases. There are **no tests yet** (Phase 5).

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
  Tokenization and decoding happen inside `model.py`, so callers (and the future worker) only
  pass plain text. If you change the I/O contract, update `model.py`, both `config.pbtxt`, and
  `scripts/smoke_test.py` together.
- **The two `model.py` files are intentionally identical** — the sibling `onnx/` directory is
  what makes an instance translate a given direction. Keep them in sync (or refactor to a shared
  module mounted into both) if you edit one.
- **Config is centralized** in `src/envit5/core/settings.py` (pydantic-settings, env prefix
  `ENVIT5_`). Don't read env vars ad hoc elsewhere; add a field there. Use `get_settings()`
  (the `@lru_cache` singleton) everywhere in application code — never instantiate `Settings()`
  directly.

## Build / run loop

Order matters — export must populate `model_repository/*/1/onnx/` before Triton can load:

```bash
make build      # build the Triton image (large: installs torch-cpu/transformers/optimum)
make export     # download HF checkpoints, export to ONNX into model_repository/*/1/onnx/
make up         # start Triton + Redis
make ready      # probe http://localhost:8000/v2/health/ready
make smoke      # translate a sample both ways (needs: pip install -e '.[client]')
make stress     # stress-test both models and print latency/throughput report
make logs       # tail Triton logs
make ps         # show service status
make down       # stop
make clean      # delete exported ONNX artifacts (model_repository/*/1/onnx/)
make all        # build + export + up
```

`make export` runs `scripts/export_models.py` *inside* the Triton image (so you don't need
torch locally). HuggingFace checkpoints are cached in a Docker named volume (`hf-cache`), so
repeat exports skip the download. To export only one direction: pass `--only translator_en_vi`
(or `translator_vi_en`) to `scripts/export_models.py` directly. Override model ids via
`ENVIT5_HF_MODEL_EN_VI` / `ENVIT5_HF_MODEL_VI_EN` (see `.env.example`). Bump
`TRITON_VERSION` if the default base image tag isn't on nvcr.io.

## Dev setup

```bash
pip install -e '.[dev]'      # ruff + pytest
pip install -e '.[client]'   # tritonclient + numpy (for smoke_test.py)
pip install -e '.[export]'   # ML deps — only on Python 3.10–3.12
```

Lint and format:

```bash
ruff check .      # lint
ruff format .     # auto-format
```

## Conventions / gotchas

- **Exported ONNX is a build artifact**, git-ignored (`model_repository/**/onnx/`). Never commit
  it; regenerate with `make export`. `model.py` + `config.pbtxt` are the tracked source.
- Triton ports: **8000** HTTP, **8001** gRPC, **8002** Prometheus metrics.
- GPU support requires the **NVIDIA driver + NVIDIA Container Toolkit** on the host. The
  compose file passes `count: 1` GPU to Triton; both `config.pbtxt` files use `KIND_GPU`.
  `onnxruntime-gpu` (in the Dockerfile) picks up `CUDAExecutionProvider` automatically.
  CUDA torch (`cu124` wheel) is required for Optimum IO binding (GPU output buffer allocation).
- Host Python is 3.14, but ML deps run in containers; the export venv (if run locally instead of
  via Docker) should be Python 3.10–3.12 where torch/transformers wheels are reliable.
- Redis DB layout: **db/0** = result cache, **db/1** = Celery broker, **db/2** = Celery result
  backend. All three URLs are configurable via `ENVIT5_REDIS_URL` / `ENVIT5_CELERY_*`.
- `ENVIT5_API_KEYS` accepts a comma-separated string (e.g. `key1,key2`) or a JSON list.
