.DEFAULT_GOAL := help
COMPOSE := docker compose --project-directory . -f infra/compose/docker-compose.yml

.PHONY: help build build-app build-ui export up api-up api-down ui-up ui-down gateway-up gateway-down down remove logs api-logs worker-logs ui-logs traefik-logs ps ready smoke stress bench eval app-stress app-eval observe clean lint all prod-pull prod-up prod-down

API_KEY     ?= changeme
GATEWAY_TAG ?= v0.1.0
UI_TAG      ?= v0.1.0
TRITON_TAG  ?= v0.1.0
PROD_COMPOSE := docker compose --project-directory . -f infra/compose/docker-compose.yml -f infra/compose/docker-compose.prod.yml

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build the Triton serving image (large: installs torch + optimum, ~30 min)
	$(COMPOSE) build triton

build-app: ## Build the API + worker image
	$(COMPOSE) build api

build-ui: ## Build the Gradio UI image
	$(COMPOSE) build ui

export: ## Export HF checkpoints to ONNX into services/serving/triton/model_repository/ (run once before `up`)
	$(COMPOSE) run --rm --no-deps -w /workspace triton \
		python3 scripts/export_models.py --model-repository /models

up: ## Start Triton Inference Server + Redis in the background
	$(COMPOSE) up -d triton redis

gateway-up: ## Start Traefik API gateway
	$(COMPOSE) up -d traefik

gateway-down: ## Stop Traefik API gateway
	$(COMPOSE) stop traefik

api-up: ## Start API + Celery worker (requires: make up && make gateway-up)
	$(COMPOSE) up -d api worker

api-down: ## Stop API + worker only
	$(COMPOSE) stop api worker

ui-up: ## Start Gradio demo UI at http://localhost/ (requires: make api-up)
	$(COMPOSE) up -d ui

ui-down: ## Stop Gradio demo UI
	$(COMPOSE) stop ui

down: ## Stop and remove containers
	$(COMPOSE) down

remove: ## Stop and remove containers, volumes, and networks
	$(COMPOSE) down -v --rmi all --remove-orphans

logs: ## Tail Triton logs
	$(COMPOSE) logs -f triton

api-logs: ## Tail API logs
	$(COMPOSE) logs -f api

worker-logs: ## Tail Celery worker logs
	$(COMPOSE) logs -f worker

ui-logs: ## Tail Gradio UI logs
	$(COMPOSE) logs -f ui

traefik-logs: ## Tail Traefik gateway logs
	$(COMPOSE) logs -f traefik

ps: ## Show service status
	$(COMPOSE) ps

ready: ## Probe Triton readiness endpoint
	@curl -fsS localhost:8000/v2/health/ready >/dev/null \
		&& echo "triton: READY" || echo "triton: NOT READY"

smoke: ## Translate a sample both ways (needs: pip install -e '.[client]' in services/serving/triton)
	python services/serving/triton/scripts/smoke_test.py --url localhost:8000

stress: ## Stress-test both models and print latency report (needs: pip install -e '.[client]')
	python scripts/stresstest.py --url localhost:8000

bench: ## GPU-saturation benchmark: async mode, long texts, high concurrency (needs: pip install -e '.[client]')
	python scripts/stresstest.py --url localhost:8000 \
		--async --requests 5000 --concurrency 64 --text-size long

eval: ## Evaluate translation quality with BLEU/chrF on 5000 HF samples (needs: pip install -e '.[eval]')
	python scripts/stresstest.py --url localhost:8000 \
		--eval-dataset talmp/en-vi-translation \
		--eval-samples 5000 \
		--concurrency 20

app-stress: ## Stress-test the FastAPI app via gateway (needs: make api-up; override key with API_KEY=mykey)
	python scripts/stresstest.py --target app --api-url http://localhost:80 --api-key $(API_KEY)

app-eval: ## Evaluate FastAPI app quality with BLEU/chrF on 5000 HF samples (needs: make api-up, pip install -e '.[eval]')
	python scripts/stresstest.py --target app --api-url http://localhost:80 --api-key $(API_KEY) \
		--eval-dataset talmp/en-vi-translation \
		--eval-samples 5000 \
		--concurrency 20 \
		--results-file results/app_eval_en_vi.json

observe: ## Start full observability stack (Prometheus, Grafana, Loki, Promtail, node/DCGM exporters)
	$(COMPOSE) up -d node-exporter dcgm-exporter prometheus grafana loki promtail

lint: ## Run ruff + pylint over gateway src/ and tests/
	cd services/gateway && ruff check . && pylint src/ tests/

clean: ## Delete exported ONNX artifacts from services/serving/triton/model_repository/ (uses Docker)
	$(COMPOSE) run --rm --no-deps -w /workspace triton \
		sh -c 'rm -rf /models/*/1/onnx'

all: build-app build-ui up gateway-up api-up ui-up observe ## Build app/UI images, start all services

prod-pull: ## Pull all pre-built images from GHCR (GATEWAY_TAG=, UI_TAG=, TRITON_TAG=)
	GATEWAY_TAG=$(GATEWAY_TAG) UI_TAG=$(UI_TAG) TRITON_TAG=$(TRITON_TAG) $(PROD_COMPOSE) pull triton api worker ui

prod-up: ## Start all services from GHCR images without building
	GATEWAY_TAG=$(GATEWAY_TAG) UI_TAG=$(UI_TAG) TRITON_TAG=$(TRITON_TAG) $(PROD_COMPOSE) up -d --no-build

prod-down: ## Stop and remove prod containers
	$(PROD_COMPOSE) down
