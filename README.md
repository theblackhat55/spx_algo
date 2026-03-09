# SPX Iron-Condor Algo

A production-grade algorithmic trading system that generates daily SPX iron-condor signals using ML-based range prediction and conformal prediction intervals.

## Performance (Jan 2 – Feb 26, 2026 replay)

| Metric | Value |
|---|---|
| Condor win rate | 86.8% (33/38) |
| Total P&L | $29,135 |
| Avg win | $864 |
| Avg loss | $122 |
| Max drawdown | -$216 |
| Sharpe ratio | 4.53 |
| 90% conformal coverage | 85.5% |

## How It Works

The system predicts the next-day SPX high and low as percentage deviations from the prior close, wraps them in conformal prediction intervals, and places iron-condor strikes at the 90% interval bounds. A regime detector (VIX z-score + ATR expansion + HMM) gates trades: GREEN = full size, YELLOW = half size, RED = skip.

**Models:** LightGBM regressors trained via 265-fold walk-forward validation on 76 features (volatility, technical, calendar, proximity, options, lagged targets) across 6,379 trading days (2000–2026).

**Wing width:** 10 points ($1,000 max risk per contract).

## Architecture

```
data/raw/            ← Daily SPX + VIX OHLCV (fetched via yfinance)
data/processed/      ← Feature matrix (76 features, built by features/builder.py)
output/models/       ← Trained model artifacts (.pkl)
output/signals/      ← Daily signal JSON (generated at 4:30 PM ET)
output/trades/       ← Paper-trade log CSV
output/monitoring/   ← Drift log + retrain flags
output/reports/      ← Backtest, coverage, and digest reports
```

## Quick Start

### Prerequisites

- Python 3.11+
- Ubuntu 22.04+ (tested on Hetzner Cloud CX22)

### Installation

```bash
git clone https://github.com/theblackhat55/spx_algo.git
cd spx_algo
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set FRED_API_KEY
```

### Download Data

```bash
python3.11 << 'PYEOF'
import yfinance as yf, pandas as pd
for ticker, name in [('^GSPC', 'spx_daily'), ('^VIX', 'vix_daily')]:
    df = yf.download(ticker, start='2000-01-01', auto_adjust=True, progress=False)
    df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df.to_parquet(f'data/raw/{name}.parquet')
    print(f'{name}: {len(df)} rows through {df.index[-1].date()}')
PYEOF
```

### Build Features

```bash
python3.11 -m src.features.builder
```

### Run Tests

```bash
pytest tests/ -v --tb=short    # 386 passed, 0 failed
```

### Train Models (walk-forward, ~40 min)

```bash
python3.11 scripts/walk_forward_train.py --folds 265 --start-year 2010
```

This trains 6 regressors (XGBoost, LightGBM, CatBoost, Huber-XGBoost, Huber-LightGBM, Ridge) for both high and low targets, selects the best by tail-252 MAE, and saves artifacts to `output/models/`.

### Generate a Signal

```bash
python3.11 -c "
import sys; sys.path.insert(0, '.')
from src.pipeline.signal_generator import SignalGenerator
sig = SignalGenerator().generate(mode='paper', save=True)
print(f'Date: {sig.signal_date} | Regime: {sig.regime} | Direction: {sig.direction}')
print(f'Predicted high: {sig.predicted_high:.2f} | Predicted low: {sig.predicted_low:.2f}')
print(f'Short call: {sig.ic_short_call:.2f} | Short put: {sig.ic_short_put:.2f}')
print(f'Model: {sig.model_versions}')
"
```

### Run a Backtest

```bash
python3.11 << 'PYEOF'
import sys; sys.path.insert(0, '.')
from src.backtest.engine import IronCondorEngine, PositionConfig

config = PositionConfig(wing_width_pts=10)
engine = IronCondorEngine(config=config)
# See output/reports/backtest_10pt_wings_v2.csv for full results
PYEOF
```

### Replay Historical Period

Generate signals as-of each trading day and compare against actuals:

```bash
python3.11 << 'PYEOF'
import sys; sys.path.insert(0, '.')
from src.pipeline.signal_generator import SignalGenerator
import pandas as pd

spx = pd.read_parquet("data/raw/spx_daily.parquet")
spx.index = pd.to_datetime(spx.index)
gen = SignalGenerator()

target_date = pd.Timestamp("2026-02-26")
prior_idx = spx.index.get_loc(target_date) - 1
as_of = spx.index[prior_idx].strftime("%Y-%m-%d")

sig = gen.generate(mode="replay", as_of_date=as_of, save=False)
print(f"Prediction for {target_date.date()}:")
print(f"  High: {sig.predicted_high:.2f}, Low: {sig.predicted_low:.2f}")
print(f"  Actual: High={spx.loc[target_date, 'High']:.2f}, Low={spx.loc[target_date, 'Low']:.2f}")
PYEOF
```

## Daily Operations (Paper Trading)

### Automated Scheduling (cron)

The system runs two cron jobs on weekdays:

| Time (UTC) | Time (ET) | Job | Script |
|---|---|---|---|
| 21:30 | 4:30 PM | Data refresh + signal generation | `scripts/daily_cron.sh` |
| 15:00 | 10:00 AM | Reconciliation + drift check | `scripts/morning_reconcile.sh` |

Install with:

```bash
crontab -e
# Add:
30 21 * * 1-5 /root/spx_algo/scripts/daily_cron.sh >> /var/log/spx-algo-cron.log 2>&1
0 15 * * 1-5 /root/spx_algo/scripts/morning_reconcile.sh >> /var/log/spx-algo-recon.log 2>&1
```

### Monitoring

```bash
# Latest signal
cat output/signals/latest_signal.json | python3.11 -m json.tool

# Paper trade log
cat output/trades/paper_trade_log.csv

# Cron logs
tail -50 /var/log/spx-algo-cron.log
tail -50 /var/log/spx-algo-recon.log

# Drift status
ls -lt output/reports/drift_*.json | head -5
```

### Retraining

Retrain when the drift detector flags DEGRADED for consecutive days, or monthly:

```bash
# 1. Refresh data
python3.11 << 'PYEOF'
import yfinance as yf, pandas as pd
for ticker, name in [('^GSPC', 'spx_daily'), ('^VIX', 'vix_daily')]:
    df = yf.download(ticker, start='2000-01-01', auto_adjust=True, progress=False)
    df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df.to_parquet(f'data/raw/{name}.parquet')
PYEOF

# 2. Rebuild features
python3.11 -m src.features.builder

# 3. Walk-forward train (~40 min)
python3.11 scripts/walk_forward_train.py --folds 265

# 4. Verify
pytest tests/ --tb=short -q
python3.11 -c "
import sys; sys.path.insert(0, '.')
from src.pipeline.signal_generator import SignalGenerator
sig = SignalGenerator().generate(mode='paper', save=False)
print(f'Model: {sig.model_versions}')
"
```

## Phase Summary

| Phase | Deliverables | Tests |
|---|---|---|
| 1 — Data & Features | Live fetcher, 76 features (volatility, technical, calendar, proximity, options, lagged), leakage gate | 120 |
| 2 — Targets & Models | XGBoost, LightGBM, CatBoost + Huber-loss variants, Ridge, walk-forward training | 66 |
| 3 — Calibration | Conformal prediction (MAPIE/residual-ICP), regime HMM + VIX + ATR, 63-day calibration window | 50 |
| 4 — Backtest | Iron-condor engine, intrusion-depth P&L, $0.40/condor friction, regime gating | 50 |
| 5 — Pipeline | Signal generator, scheduler, daily orchestrator, Docker support | 48 |
| 6 — Monitoring | Drift detector, paper logger, reconciler, Optuna hyperparameter sweep, conformal verification | 52 |
| **Total** | | **386 passed · 0 failed** |

## Configuration

Copy `.env.example` to `.env` and set:

```
FRED_API_KEY=your_fred_key
IBKR_PORT=7497
EXECUTION_MODE=PAPER
DISCORD_WEBHOOK_URL=           # Optional alerts
```

Key parameters in code:

| Parameter | Value | Location |
|---|---|---|
| Wing width | 10 pts ($1,000 max risk) | `signal_generator.py`, `engine.py`, `paper_logger.py` |
| Calibration window | 63 days (~3 months) | `signal_generator.py` CALIBRATION_TAIL |
| Regime skip | RED days skipped | `engine.py` PositionConfig |
| Friction | $0.40/condor ($0.10/leg × 4) | `engine.py` PositionConfig |
| Walk-forward folds | 265 | `walk_forward_train.py` |


## Experimental Range-Forecast Workflow

The `feature/phase-6c-es-overnight` branch adds an experimental Databento-enhanced forecasting workflow for SPX daily OHLC/range research.

Key scripts:
- `scripts/walkforward_backtest_base_ohlc.py` — base walk-forward OHLC benchmark
- `scripts/walkforward_backtest_gap_augmented_hybrid.py` — Databento gap-augmented hybrid walk-forward test
- `scripts/compare_walkforward_base_vs_gap_hybrid.py` — overlap comparison of base vs hybrid
- `scripts/evaluate_hybrid_plus_range_model.py` — holdout evaluation of hybrid + direct range overlay
- `scripts/evaluate_hybrid_range_and_skew_model.py` — holdout evaluation of hybrid + range + skew overlay
- `scripts/generate_gap_augmented_range_skew_forecast.py` — generate latest experimental range+skew forecast
- `scripts/run_gap_augmented_range_skew_forecast_step.py` — archive latest experimental forecast
- `scripts/compare_daily_hybrid_vs_range_skew_actuals.py` — compare archived forecasts against Yahoo SPX actuals
- `scripts/run_daily_forecast_comparison_step.py` — archived comparison runner
- `scripts/print_latest_expected_range.py` — quick operational summary for latest forecast

**Important:** These range/range+skew overlay scripts are research utilities and should be treated as experimental until validated over a sufficiently long forward period.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/daily_cron.sh` | Automated daily pipeline: data refresh → features → signal → log |
| `scripts/morning_reconcile.sh` | Morning reconciliation: fetch actuals → compute errors → drift check |
| `scripts/walk_forward_train.py` | Walk-forward model training (regressors + classifiers) |
| `scripts/daily_orchestrator.py` | Python-based daily orchestrator with notifications |
| `scripts/daily_pipeline.sh` | Bash-based daily pipeline |
| `scripts/resume_from_step5.py` | Resume pipeline from signal generation step |
| `scripts/smoke_test.sh` | End-to-end smoke test |

## Go/No-Go Criteria (after 60 trading days)

| Criterion | Target | Jan-Feb 2026 Result |
|---|---|---|
| Condor win rate | > 65% | 86.8% ✓ |
| Total P&L | > $0 | $29,135 ✓ |
| Max drawdown | < $5,000 | -$216 ✓ |
| 90% conformal coverage | > 80% | 85.5% ✓ |
| Drift status | No extended DEGRADED | Healthy ✓ |

## Docker

```bash
make build
make run
docker compose run --rm algo python -m src.pipeline.signal_generator
```

## License

MIT
