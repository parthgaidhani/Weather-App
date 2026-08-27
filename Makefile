.PHONY: help install install-dev lint format test-imports test clean run-api run-ui run-train docker-build docker-run

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PIP) install -r requirements.txt

install-dev:  ## Install runtime + dev dependencies
	$(PIP) install -r requirements.txt -r requirements-dev.txt

lint:  ## Run all linters
	ruff check src/
	black --check --line-length 100 src/
	isort --check-only --profile black src/

format:  ## Auto-format code
	black --line-length 100 src/
	isort --profile black src/

test-imports:  ## Smoke-test that every module parses
	$(PYTHON) -c "import ast, pathlib, sys; \
errs = [f'{p}: {e}' for p in pathlib.Path('src').rglob('*.py') \
try: ast.parse(p.read_text(encoding='utf-8')) \
except SyntaxError as e]; \
print('\n'.join(errs) or 'All modules parse cleanly.'); sys.exit(1 if errs else 0)"

test:  ## Run pytest
	pytest tests/ -v --cov=src --cov-report=term-missing

run-api:  ## Launch FastAPI backend (port 8000)
	uvicorn src.api.main:app --reload --port 8000

run-ui:  ## Launch Streamlit dashboard (port 8501)
	streamlit run src/ui/app.py

run-train:  ## Run full training pipeline
	$(PYTHON) train.py --steps data preprocess ml

docker-build:  ## Build Docker image
	docker build -t weather-app:latest .

docker-run:  ## Run docker-compose stack
	docker-compose up --build

clean:  ## Remove caches & build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ dist/ build/ *.egg-info
