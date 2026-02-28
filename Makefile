# SPX Algo — Makefile
# Usage:  make build | make test | make test-leakage | make lint | make coverage | make signal | make run

.PHONY: build test test-leakage test-integration lint coverage signal run clean

# ── Docker ────────────────────────────────────────────────────────────────
build:
	docker compose build

run:
	docker compose up scheduler

# ── Tests ─────────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v --tb=short -q

test-leakage:
	@echo "=== Running leakage gate ==="
	python -m pytest tests/test_no_leakage.py -v --tb=short
	@echo "=== Leakage gate PASSED ==="

test-integration:
	python -m pytest tests/test_integration.py -v --tb=short -m integration

test-phase4:
	python -m pytest tests/test_backtest.py -v --tb=short

test-all: test-leakage test
	@echo "=== Full test suite complete ==="

# ── Coverage ──────────────────────────────────────────────────────────────
coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing \
	    --cov-fail-under=80 -q

# ── Lint ──────────────────────────────────────────────────────────────────
lint:
	@command -v ruff >/dev/null 2>&1 && ruff check src/ tests/ || \
	    python -m flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503

# ── Signal generation ─────────────────────────────────────────────────────
signal:
	python -m src.pipeline.signal_generator

# ── Live data fetch ───────────────────────────────────────────────────────
fetch:
	python -c "from src.data.live_fetcher import run_daily_fetch; import pprint; pprint.pprint(run_daily_fetch())"

# ── Clean ─────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
