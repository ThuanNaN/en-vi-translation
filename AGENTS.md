# AGENTS.md — Guidance for AI coding agents

Purpose: help AI coding agents be immediately productive in this repository. Keep this file minimal and link-first; follow links for details.

- **Project:** PolyglotHub — Model- and engine-agnostic translation/serving hub (formerly `en-vi-translation`).
	PolyglotHub is designed to accept any model and route it to any inference engine via a pluggable backend registry.
- **Primary commands:** `make export`, `make build-app`, `make up`, `make api-up`, `make smoke`, `make test`.
- **Where to read full details:** See [CLAUDE.md](CLAUDE.md) and [README.md](README.md) (PolyglotHub) for architecture, run loop, and gotchas.

Key files & quick notes (actionable):
- **Config:** `services/gateway/src/polyglot_gateway/core/settings.py`: use `get_settings()` (cached) everywhere — do not instantiate `Settings()` directly. See [settings.py](services/gateway/src/polyglot_gateway/core/settings.py#L1-L200).
- **Backend registry:** `BACKENDS` env JSON drives routing. See [CLAUDE.md](CLAUDE.md#what-this-is) and [services/gateway/src/polyglot_gateway/worker/backends/registry.py](services/gateway/src/polyglot_gateway/worker/backends/registry.py#L1-L200).
- **Celery:** `services/gateway/src/polyglot_gateway/worker/celery_app.py` — `task_soft_time_limit`/`task_time_limit` must match `request_timeout_seconds` in settings. Do not rename task `polyglot.translate`.
- **Chunking:** `services/gateway/src/polyglot_gateway/worker/chunker.py` contains the chunking and reassembly logic — change here if altering chunk sizes.
- **Cache & job markers:** Redis keys `trans:<digest>` and `jobtrack:<id>` are part of runtime contracts — do not change prefixes.
- **Triton models / export:** ONNX artifacts are build outputs (gitignored). Run `make export` before starting Triton. See `services/serving/triton/`.
- **Tests:** Tests mock `BACKENDS` and Redis; use `pytest` from `services/gateway/` for unit tests. See `services/gateway/tests/conftest.py` for patterns.

Conventions & gotchas (short):
- Link, don't duplicate: prefer referencing `CLAUDE.md` or source files for details.
- Timeouts: keep Celery timeouts and `request_timeout_seconds` in sync.
- Metrics: worker exposes Prometheus on port `9091` (set `PROMETHEUS_MULTIPROC_DIR` when running tests/worker).
- ONNX artifacts are produced by `make export` inside Triton container (root-owned files).

If you'd like, I can also add a minimal `.github/copilot-instructions.md` bridging to this file and calling out quick-edit guidelines (e.g., where to add backends, how to wire a new Triton model). Next, I can generate suggested agent skills for automated tasks (tests, export, smoke tests).
