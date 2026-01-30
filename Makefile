.PHONY: help install dev test test-unit test-integration lint format typecheck clean run-local docker-build docker-up docker-down deploy frontend-install frontend-dev frontend-build frontend-lint

# Default Python version
PYTHON := python3

# Project paths
SRC := src/dova
TESTS := tests

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ======================
# Development Setup
# ======================

install: ## Install production dependencies
	$(PYTHON) -m pip install -e .

dev: ## Install development dependencies
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

# ======================
# Testing
# ======================

test: ## Run all tests
	pytest $(TESTS) -v --cov=$(SRC) --cov-report=term-missing

test-unit: ## Run unit tests only
	pytest $(TESTS)/unit -v

test-integration: ## Run integration tests only
	pytest $(TESTS)/integration -v -m integration

test-fast: ## Run fast tests (no slow markers)
	pytest $(TESTS) -v -m "not slow"

# ======================
# Code Quality
# ======================

lint: ## Run linting checks
	ruff check $(SRC) $(TESTS)

format: ## Format code with ruff
	ruff format $(SRC) $(TESTS)
	ruff check --fix $(SRC) $(TESTS)

typecheck: ## Run type checking with mypy
	mypy $(SRC)

check: lint typecheck ## Run all checks (lint + typecheck)

# ======================
# Local Development
# ======================

run-local: ## Run API server locally
	uvicorn dova.api.main:app --host 0.0.0.0 --port 8000 --reload

run-cli: ## Run CLI in interactive mode
	$(PYTHON) -m dova.cli

# ======================
# Frontend
# ======================

frontend-install: ## Install frontend dependencies
	cd frontend && npm install

frontend-dev: ## Run frontend dev server
	cd frontend && npm run dev

frontend-build: ## Build frontend for production
	cd frontend && npm run build

frontend-lint: ## Lint frontend code
	cd frontend && npm run lint

# ======================
# Docker
# ======================

docker-build: ## Build Docker image
	docker build -t dova:latest .

docker-up: ## Start Docker Compose services
	docker-compose up -d

docker-down: ## Stop Docker Compose services
	docker-compose down

docker-logs: ## View Docker Compose logs
	docker-compose logs -f

# ======================
# Infrastructure
# ======================

cdk-install: ## Install CDK dependencies
	cd infra && npm install

cdk-synth: ## Synthesize CDK stack
	cd infra && npx cdk synth

cdk-deploy: ## Deploy CDK stack
	cd infra && npx cdk deploy --require-approval never

cdk-destroy: ## Destroy CDK stack
	cd infra && npx cdk destroy --force

deploy: docker-build cdk-deploy ## Full deployment (build + CDK deploy)

# ======================
# Utilities
# ======================

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

seed-data: ## Seed test data
	$(PYTHON) scripts/seed_data.py

create-memory: ## Create AgentCore memory
	$(PYTHON) scripts/create_memory.py

# ======================
# Documentation
# ======================

docs-serve: ## Serve documentation locally
	mkdocs serve

docs-build: ## Build documentation
	mkdocs build
