COMPOSE      := docker compose --project-directory . -f infra/docker-compose.yml
PROD_COMPOSE := docker compose --project-directory . -f infra/docker-compose.yml -f infra/docker-compose.prod.yml

API_KEY     ?= changeme
GATEWAY_TAG ?= v0.1.0
UI_TAG      ?= v0.1.0
TRITON_TAG  ?= v0.1.0

.DEFAULT_GOAL := help

.PHONY: \
	help \
	build build-triton build-app build-ui build-custom export clean \
	infra-up infra-down gateway-up gateway-down ps down remove \
	triton-up triton-down triton-logs triton-ready \
	vllm-up vllm-down vllm-logs \
	custom-up custom-down custom-logs \
	api-up api-down api-logs worker-logs \
	ui-up ui-down ui-logs \
	observe observe-down prom-logs grafana-logs \
	smoke stress bench \
	app-stress app-stress-llm app-stress-all vllm-stress \
	lint \
	all prod-pull prod-up prod-down


# ── Help ──────────────────────────────────────────────────────────────────────

help: ## Show available targets grouped by section
	@awk 'BEGIN{FS=":.*?## "; section=""} \
		/^## / { printf "\n\033[1m%s\033[0m\n", substr($$0,4) } \
		/^[a-zA-Z_-]+:.*?## / { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' \
		$(MAKEFILE_LIST)


# ── Build ─────────────────────────────────────────────────────────────────────
## Build

build-triton: ## Build the Triton image (large: installs torch + optimum, ~30 min)
	$(COMPOSE) build triton

build-app: ## Build the API + Celery worker image
	$(COMPOSE) build api

build-ui: ## Build the Gradio UI image
	$(COMPOSE) build ui

build-custom: ## Build the example custom inference service image
	$(COMPOSE) build custom-service

build: build-triton build-app build-ui ## Build all images (Triton + app + UI)

export: ## Export HF checkpoints → ONNX inside the Triton container (run once before triton-up)
	$(COMPOSE) run --rm --no-deps -w /workspace triton \
		python3 scripts/export_models.py --model-repository /models

clean: ## Delete exported ONNX artifacts from inference/triton/model_repository/ (runs inside Docker)
	$(COMPOSE) run --rm --no-deps -w /workspace triton \
		sh -c 'rm -rf /models/*/1/onnx'


# ── Infrastructure ────────────────────────────────────────────────────────────
## Infrastructure

infra-up: ## Start core infra: Redis + Traefik
	$(COMPOSE) up -d redis traefik

infra-down: ## Stop Redis + Traefik
	$(COMPOSE) stop redis traefik

gateway-up: ## Start Traefik reverse proxy / API gateway
	$(COMPOSE) up -d traefik

gateway-down: ## Stop Traefik
	$(COMPOSE) stop traefik

ps: ## Show status of all running containers
	$(COMPOSE) ps

down: ## Stop and remove all containers
	$(COMPOSE) down

remove: ## Stop and remove containers, volumes, networks, and images
	$(COMPOSE) down -v --rmi all --remove-orphans


# ── Inference engines ─────────────────────────────────────────────────────────
## Inference engines

triton-up: ## Start Triton Inference Server (ONNX NMT models; requires: make export)
	$(COMPOSE) up -d triton

triton-down: ## Stop Triton
	$(COMPOSE) stop triton

triton-logs: ## Tail Triton logs
	$(COMPOSE) logs -f triton

triton-ready: ## Probe Triton readiness endpoint
	@curl -fsS localhost:8000/v2/health/ready >/dev/null \
		&& echo "triton: READY" || echo "triton: NOT READY"

vllm-up: ## Start vLLM (Qwen/Qwen3.5-0.8B, OpenAI-compatible API on :8010; requires GPU)
	$(COMPOSE) up -d vllm

vllm-down: ## Stop vLLM
	$(COMPOSE) stop vllm

vllm-logs: ## Tail vLLM logs
	$(COMPOSE) logs -f vllm

custom-up: ## Start the example custom inference service (inference/custom/example_service/)
	$(COMPOSE) up -d custom-service

custom-down: ## Stop the example custom inference service
	$(COMPOSE) stop custom-service

custom-logs: ## Tail custom inference service logs
	$(COMPOSE) logs -f custom-service


# ── Backend API ───────────────────────────────────────────────────────────────
## Backend API

api-up: ## Start FastAPI + Celery worker (requires: infra-up + triton-up)
	$(COMPOSE) up -d api worker

api-down: ## Stop FastAPI + Celery worker
	$(COMPOSE) stop api worker

api-logs: ## Tail FastAPI logs
	$(COMPOSE) logs -f api

worker-logs: ## Tail Celery worker logs
	$(COMPOSE) logs -f worker


# ── Frontend UI ───────────────────────────────────────────────────────────────
## Frontend UI

ui-up: ## Start Gradio demo UI at http://localhost/ (requires: api-up)
	$(COMPOSE) up -d ui

ui-down: ## Stop Gradio UI
	$(COMPOSE) stop ui

ui-logs: ## Tail Gradio UI logs
	$(COMPOSE) logs -f ui


# ── Observability ─────────────────────────────────────────────────────────────
## Observability

observe: ## Start full observability stack: Prometheus, Grafana, Loki, Promtail, node/DCGM exporters
	$(COMPOSE) up -d node-exporter dcgm-exporter prometheus grafana loki promtail

observe-down: ## Stop observability stack
	$(COMPOSE) stop node-exporter dcgm-exporter prometheus grafana loki promtail

prom-logs: ## Tail Prometheus logs
	$(COMPOSE) logs -f prometheus

grafana-logs: ## Tail Grafana logs
	$(COMPOSE) logs -f grafana


# ── Testing & evaluation ──────────────────────────────────────────────────────
## Testing & evaluation

smoke: ## Smoke-test Triton directly (translate a sample both ways)
	python inference/triton/scripts/smoke_test.py --url localhost:8000

stress: ## Stress-test Triton directly — latency/throughput report
	python scripts/stresstest.py --url localhost:8000

bench: ## GPU saturation benchmark via Triton (async, long texts, high concurrency)
	python scripts/stresstest.py --url localhost:8000 \
		--async --requests 5000 --concurrency 64 --text-size long

app-stress: ## Stress-test the FastAPI app via Triton NMT backend (needs: api-up; API_KEY=mykey)
	python scripts/stresstest.py --target app --api-url http://localhost:80 --api-key $(API_KEY)

vllm-stress: ## Stress-test vLLM directly via OpenAI API (needs: vllm-up)
	python scripts/stresstest.py --target vllm --vllm-url localhost:8010 \
		--requests 100 --concurrency 8 --text-size medium

app-stress-llm: ## Stress-test FastAPI routed through vLLM (needs: api-up + vllm-up)
	python scripts/stresstest.py --target app --api-url http://localhost:80 --api-key $(API_KEY) \
		--model llm --requests 50 --concurrency 4 --text-size medium

app-stress-all: ## Stress-test FastAPI with BOTH backends: NMT (Triton) + LLM (vLLM) combined report (needs: api-up + vllm-up)
	python scripts/app_stresstest.py \
		--api-url http://localhost:80 --api-key $(API_KEY) \
		--nmt-requests 100 --nmt-concurrency 4 \
		--llm-requests 20  --llm-concurrency 4 \
		--text-size short


# ── Dev ───────────────────────────────────────────────────────────────────────
## Dev

lint: ## Run ruff check + pylint over apps/api/
	cd apps/api && ruff check . && pylint src/ tests/


# ── Shortcuts ─────────────────────────────────────────────────────────────────
## Shortcuts

all: build-app build-ui infra-up triton-up api-up ui-up observe ## Build and start everything (Triton must be pre-built)

prod-pull: ## Pull pre-built images from GHCR (GATEWAY_TAG=, UI_TAG=, TRITON_TAG=)
	GATEWAY_TAG=$(GATEWAY_TAG) UI_TAG=$(UI_TAG) TRITON_TAG=$(TRITON_TAG) \
		$(PROD_COMPOSE) pull triton api worker ui

prod-up: ## Start all services from GHCR images without building
	GATEWAY_TAG=$(GATEWAY_TAG) UI_TAG=$(UI_TAG) TRITON_TAG=$(TRITON_TAG) \
		$(PROD_COMPOSE) up -d --no-build

prod-down: ## Stop and remove prod containers
	$(PROD_COMPOSE) down
