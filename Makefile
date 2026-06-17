# EthicLens developer entrypoints. `make help` lists targets.
.DEFAULT_GOAL := help
PY := .venv/Scripts/python.exe

.PHONY: help install lint format type test cov audit-golden demo up down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create venv and install all workspace packages + dev tools
	uv venv
	uv pip install -e "packages/fairness-core[validation,viz,cli]" --group dev

lint: ## Ruff lint
	uv run ruff check .

format: ## Ruff format
	uv run ruff format .

type: ## mypy strict on the engine
	uv run mypy packages/fairness-core/src

test: ## Run the full test suite
	uv run pytest

cov: ## Tests with coverage gate (>= 85%)
	uv run pytest --cov=fairness_core --cov-report=term-missing --cov-fail-under=85

audit-golden: ## Reproduce the golden-reference audit (DI ~ 0.55)
	$(PY) -m ml.training.train_calibrated_bias_model --verify

demo: ## Print a Fairness Scorecard for a freshly trained biased model
	uv run ethiclens-audit demo

up: ## Start the full stack (api, worker, web, postgres, redis, mlflow)
	docker compose up --build

down: ## Stop the stack
	docker compose down -v

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
