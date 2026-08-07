.DEFAULT_GOAL := help
.PHONY: help install dev lint format types test test-all cov check doctor run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	pip install -e ".[tools,app]"

dev: ## Install everything and set up hooks
	pip install -e ".[dev,tools,app]" && pre-commit install

lint: ## Lint
	ruff check .

format: ## Format
	ruff format .

types: ## Type check (strict)
	mypy

test: ## Unit tests
	pytest -m unit

test-all: ## All tests, including live integration
	pytest

cov: ## Unit tests with coverage
	pytest -m unit --cov --cov-report=term-missing --cov-report=html

check: lint types test ## Everything CI runs
	ruff format --check .

doctor: ## Show resolved configuration
	agent doctor

run: ## Answer three benchmark tasks
	agent run --limit 3

clean: ## Remove build and cache artefacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage site
	find . -type d -name __pycache__ -not -path "./.git/*" -exec rm -rf {} +
