.DEFAULT_GOAL := help
COMPOSE := docker compose

.PHONY: help build build-app build-ui export up api-up api-down ui-up ui-down down remove logs api-logs worker-logs ui-logs ps ready smoke stress eval app-stress app-eval observe clean lint all prod-pull prod-up prod-down

API_KEY   ?= changeme
IMAGE_TAG ?= v0.1.0
PROD_COMPOSE := docker compose -f docker-compose.yml -f docker-compose.prod.yml

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

build: ## Build the Triton image (installs transformers/optimum/torch-cpu)
	$(COMPOSE) build triton

build-app: ## Build the API + worker image
	$(COMPOSE) build api

build-ui: ## Build the Gradio UI image
	$(COMPOSE) build ui

export: ## Export HF checkpoints to ONNX into model_repository/ (run once before `up`)
	$(COMPOSE) run --rm --no-deps -w /workspace triton \
		python3 scripts/export_models.py --model-repository /models

up: ## Start Triton + Redis in the background
	$(COMPOSE) up -d triton redis

api-up: ## Start API + Celery worker (requires: make up)
	$(COMPOSE) up -d api worker

api-down: ## Stop API + worker only
	$(COMPOSE) stop api worker

ui-up: ## Start Gradio demo UI (requires: make api-up)
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

ps: ## Show service status
	$(COMPOSE) ps

ready: ## Probe Triton readiness endpoint
	@curl -fsS localhost:8000/v2/health/ready >/dev/null \
		&& echo "triton: READY" || echo "triton: NOT READY"

smoke: ## Translate a sample both ways (needs: pip install -e '.[client]')
	python scripts/smoke_test.py --url localhost:8000

stress: ## Stress-test both models and print latency report (needs: pip install -e '.[client]')
	python scripts/stresstest.py --url localhost:8000

eval: ## Evaluate translation quality with BLEU/chrF on 5000 HF samples (needs: pip install -e '.[eval]')
	python scripts/stresstest.py --url localhost:8000 \
		--eval-dataset talmp/en-vi-translation \
		--eval-samples 5000 \
		--concurrency 20

app-stress: ## Stress-test the FastAPI app (needs: make api-up; override key with API_KEY=mykey)
	python scripts/stresstest.py --target app --api-url http://localhost:8080 --api-key $(API_KEY)

app-eval: ## Evaluate FastAPI app quality with BLEU/chrF on 5000 HF samples (needs: make api-up, pip install -e '.[eval]')
	python scripts/stresstest.py --target app --api-url http://localhost:8080 --api-key $(API_KEY) \
		--eval-dataset talmp/en-vi-translation \
		--eval-samples 5000 \
		--concurrency 20 \
		--results-file results/app_eval_en_vi.json

observe: ## Start full observability stack (Prometheus, Grafana, Loki, Promtail, node/DCGM exporters)
	$(COMPOSE) up -d node-exporter dcgm-exporter prometheus grafana loki promtail

lint: ## Run ruff + pylint over src/ and tests/
	ruff check .
	pylint src/ tests/

clean: ## Delete exported ONNX artifacts (uses Docker to handle root-owned files)
	$(COMPOSE) run --rm --no-deps -w /workspace triton \
		sh -c 'rm -rf /models/*/1/onnx'

all: build build-app export up api-up observe ## Build all images, export models, start all services

prod-pull: ## Pull all pre-built images from GHCR (IMAGE_TAG=v0.1.0)
	IMAGE_TAG=$(IMAGE_TAG) $(PROD_COMPOSE) pull triton api worker ui

prod-up: ## Start all services from GHCR images without building (IMAGE_TAG=v0.1.0)
	IMAGE_TAG=$(IMAGE_TAG) $(PROD_COMPOSE) up -d --no-build

prod-down: ## Stop and remove prod containers
	$(PROD_COMPOSE) down
