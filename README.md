# SPX Iron-Condor Algo

A production-grade algorithmic trading system for SPX iron-condor options, built in 6 phases.

## Architecture

```
data/raw/          ← Daily OHLCV + FRED macro (fetched by live_fetcher.py)
data/processed/    ← Feature matrix (built by features/builder.py)
output/signals/    ← Daily signal JSON (generated at 4:05 PM ET)
output/trades/     ← Paper-trade log CSV
output/monitoring/ ← Drift log + retrain flags
output/reports/    ← Coverage, sweep, and digest reports
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download historical data (needs FRED_API_KEY in .env)
python -m src.data.fetcher

# Run full test suite
make test                    # 380 passed, 6 skipped, 0 failed

# Verify conformal coverage gate
python -m src.calibration.conformal_verification

# Generate today's signal
python -m src.pipeline.signal_generator

# Run smoke test (all gates)
bash scripts/smoke_test.sh --skip-docker

# Start the daily scheduler (4:05 PM ET signal + 10:00 AM reconciliation)
python -m src.pipeline.scheduler
```

## Phase Summary

| Phase | Deliverables | Tests |
|-------|-------------|-------|
| 1 — Data & Features | Live fetcher, 20+ features, leakage gate | 120 |
| 2 — Targets & Models | XGBoost/LightGBM/CatBoost + Huber loss | 66 |
| 3 — Calibration | Conformal prediction (MAPIE/ICP), regime HMM | 50 |
| 4 — Backtest | Iron-condor engine, $0.40/condor cost, regime gating | 50 |
| 5 — Pipeline | Signal generator, scheduler, dead-man's switch, Docker | 48 |
| 6 — Monitoring | Drift detector, paper logger, reconciler, Optuna sweep | 52 |
| **Total** | | **380 passed · 0 failed** |

## Configuration

Copy `.env.example` to `.env` and fill in:

```
FRED_API_KEY=your_fred_key          # https://fred.stlouisfed.org/docs/api/api_key.html
IBKR_PORT=7497                      # TWS Paper Trading port
EXECUTION_MODE=PAPER                # Never change to LIVE until ready
DISCORD_WEBHOOK_URL=                # Optional alerts
```

## Docker

```bash
make build           # Build image
make run             # Start scheduler container
docker compose run --rm algo python -m src.pipeline.signal_generator
```

## Paper Trading Launch

See [`docs/PAPER_TRADE_RUNBOOK.md`](docs/PAPER_TRADE_RUNBOOK.md) for the full step-by-step guide.

**Go/No-Go criteria (after 60 trading days):**
- MAE < 0.5% | Directional accuracy > 58% | Condor win rate > 65%
- Conformal 68% coverage: 60–76% | No extended DEGRADED drift

## CI/CD

GitHub Actions runs on every push to `main`/`develop`:
- Leakage gate (24 tests)
- Full test suite with coverage ≥ 80%

## License

MIT
