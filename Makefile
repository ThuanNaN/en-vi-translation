.DEFAULT_GOAL := help
COMPOSE := docker compose

.PHONY: help build export up down logs ps ready smoke stress clean all

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

build: ## Build the Triton image (installs transformers/optimum/torch-cpu)
	$(COMPOSE) build triton

export: ## Export HF checkpoints to ONNX into model_repository/ (run once before `up`)
	$(COMPOSE) run --rm --no-deps -w /workspace triton \
		python3 scripts/export_models.py --model-repository /models

up: ## Start Triton + Redis in the background
	$(COMPOSE) up -d triton redis

down: ## Stop and remove containers
	$(COMPOSE) down

logs: ## Tail Triton logs
	$(COMPOSE) logs -f triton

ps: ## Show service status
	$(COMPOSE) ps

ready: ## Probe Triton readiness endpoint
	@curl -fsS localhost:8000/v2/health/ready >/dev/null \
		&& echo "triton: READY" || echo "triton: NOT READY"

smoke: ## Translate a sample both ways (needs: pip install -e '.[client]')
	python scripts/smoke_test.py --url localhost:8000

stress: ## Stress-test both models and print latency report (needs: pip install -e '.[client]')
	python scripts/stresstest.py --url localhost:8000

clean: ## Delete exported ONNX artifacts
	rm -rf model_repository/*/1/onnx

all: build export up ## Build image, export models, then start services
