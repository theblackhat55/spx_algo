Below is a ready-to-save runbook you can place at `docs/RUNBOOK.md`.

```md
# SPX Algo Operational Runbook

_Last updated: 2026-03-08_

## 1. Purpose

This runbook explains how to operate the `spx_algo` system day to day.

Current production-recommended forecast path:

- **Gap / open-sensitive component**: Databento-augmented gap model
- **High / low / close components**: existing OHLC component models
- **Primary forecast artifact**: `output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json`

This runbook is for:
- running tomorrow’s forecast
- running the daily orchestrator
- reconciling forecasts after market data is available
- building scorecards
- troubleshooting common failures
- knowing what to commit vs not commit

---

## 2. Current production architecture

### 2.1 Forecast architecture
The current best-performing architecture is:

- `target_gap_ret` → **Databento gap model**
- `target_high_from_open` → **existing OHLC model**
- `target_low_from_open` → **existing OHLC model**
- `target_close_from_open` → **existing OHLC model**

### 2.2 Key data sources
- **SPX daily**: `data/raw/spx_daily.parquet`
- **Main feature matrix**: built from SPX / VIX / calendar / options / ES daily
- **Databento ES 1-minute**: `data/raw/es_databento_1m.parquet`
- **Databento overnight features**: `data/processed/es_databento_overnight_features.parquet`

### 2.3 Main forecast output
Primary forecast artifact:
- `output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json`

Archive location:
- `output/forecasts/archive/`

---

## 3. Important scripts

### 3.1 Core forecast scripts
- `scripts/generate_gap_augmented_hybrid_forecast.py`
- `scripts/run_gap_augmented_hybrid_forecast_step.py`

### 3.2 Training / data prep
- `scripts/fetch_es_databento_1m.py`
- `scripts/build_es_databento_overnight_features.py`
- `scripts/train_databento_gap_model.py`
- `scripts/train_ohlc_models.py`

### 3.3 Evaluation / experiments
- `scripts/evaluate_databento_overnight_experiment.py`
- `scripts/evaluate_databento_gap_ablation.py`
- `scripts/evaluate_ohlc_hybrid.py`
- `scripts/evaluate_ohlc_vs_baseline.py`

### 3.4 Workflow / orchestration
- `scripts/daily_orchestrator.py`

### 3.5 Reconciliation / scorecard
- `scripts/reconcile_hybrid_forecast.py`
- `scripts/build_forecast_scorecard.py`

---

## 4. Required environment

Activate virtualenv first:

```bash
source .venv/bin/activate
```

Most commands should be run with:

```bash
PYTHONPATH=.
```

### 4.1 Required credentials / env vars
Stored in `.env`.

Important ones include:
- `DATABENTO_API_KEY`
- `FRED_API_KEY` if relevant to the rest of the system
- optional alerting vars:
  - `TELEGRAM_TOKEN`
  - `TELEGRAM_CHAT_ID`
  - `SLACK_WEBHOOK_URL`

Optional IBKR vars may exist but are not required for the Databento gap-augmented forecast path.

---

## 5. Standard daily operating procedure

## 5.1 Before market / evening before
Goal: generate tomorrow’s forecast.

### Recommended command
```bash
PYTHONPATH=. python scripts/generate_gap_augmented_hybrid_forecast.py
```

### Operator/archive version
```bash
PYTHONPATH=. python scripts/run_gap_augmented_hybrid_forecast_step.py
```

This will:
- create/update `output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json`
- archive the result under `output/forecasts/archive/YYYY-MM-DD_gap_augmented_hybrid_ohlc_forecast.json`

### Quick inspect
```bash
python -m json.tool output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json
```

---

## 5.2 Full daily orchestrator
If using the full workflow:

```bash
PYTHONPATH=. python scripts/daily_orchestrator.py
```

Current orchestrator behavior:
- runs the **gap-augmented hybrid forecast step** non-fatally
- runs the older hybrid forecast step non-fatally
- continues legacy orchestration tasks

If the new forecast step fails, orchestration should continue.

---

## 5.3 After market close / after new SPX daily data is available
Goal: reconcile prediction vs actual.

Run:
```bash
PYTHONPATH=. python scripts/reconcile_hybrid_forecast.py
```

If actual SPX OHLC for the forecast date is not available yet, reconciliation may fail or exit early depending on current script behavior.

Expected outputs:
- `output/reports/forecast_reconcile_<date>.json`
- `output/reports/forecast_history.csv`

---

## 5.4 Build rolling scorecard
After forecast history exists:

```bash
PYTHONPATH=. python scripts/build_forecast_scorecard.py
```

Expected output:
- `output/reports/forecast_scorecard.json`

---

## 6. Databento data maintenance

## 6.1 Fetch Databento ES 1-minute data
Use this to refresh or backfill raw ES intraday data:

```bash
PYTHONPATH=. python scripts/fetch_es_databento_1m.py --start 2024-01-01 --end 2026-03-08 --out data/raw/es_databento_1m.parquet
```

Adjust start/end as needed.

### Notes
- This uses `DATABENTO_API_KEY`
- Current validated symbol path uses continuous ES 1-minute bars
- Data quality was good enough for overnight/pre-open feature generation

---

## 6.2 Build Databento overnight features
After fetching or refreshing Databento ES 1-minute bars:

```bash
PYTHONPATH=. python scripts/build_es_databento_overnight_features.py
```

Expected output:
- `data/processed/es_databento_overnight_features.parquet`

Quick inspect:
```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/processed/es_databento_overnight_features.parquet")
print(df.tail())
print(len(df), df.index.min(), df.index.max())
PY
```

---

## 7. Model training procedures

## 7.1 Train base OHLC component models
Run:
```bash
PYTHONPATH=. python scripts/train_ohlc_models.py
```

Expected outputs:
- `output/models/ohlc/*.joblib`
- `output/reports/ohlc_metrics.json`

These models drive:
- high-from-open
- low-from-open
- close-from-open
- and legacy component forecasts

---

## 7.2 Train Databento gap model
Run:
```bash
PYTHONPATH=. python scripts/train_databento_gap_model.py
```

Expected outputs:
- `output/models/ohlc/gap_databento_model.joblib`
- `output/reports/gap_databento_model_report.json`
- `output/reports/gap_databento_feature_importance.csv`

This model is currently the preferred source for:
- `target_gap_ret`

---

## 8. Recommended operator command set

## 8.1 Minimal forecast-only command
```bash
PYTHONPATH=. python scripts/generate_gap_augmented_hybrid_forecast.py
```

## 8.2 Forecast + archive
```bash
PYTHONPATH=. python scripts/run_gap_augmented_hybrid_forecast_step.py
```

## 8.3 Full orchestration
```bash
PYTHONPATH=. python scripts/daily_orchestrator.py
```

## 8.4 Reconcile after actuals are available
```bash
PYTHONPATH=. python scripts/reconcile_hybrid_forecast.py
```

## 8.5 Build scorecard
```bash
PYTHONPATH=. python scripts/build_forecast_scorecard.py
```

---

## 9. Forecast artifacts and interpretation

## 9.1 Main forecast JSON
File:
- `output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json`

Typical fields:
- `forecast_for_date`
- `generated_from_feature_date`
- `prev_close`
- `component_source_selection`
- `predicted_components`
- `predicted_ohlc`
- `model_artifacts`

### Example interpretation
If `component_source_selection` is:
```json
{
  "target_gap_ret": "databento_gap_model",
  "target_high_from_open": "model",
  "target_low_from_open": "model",
  "target_close_from_open": "model"
}
```

then:
- next open/gap is informed by Databento overnight futures context
- the rest of the bar is built from the existing OHLC component models

---

## 10. Validation and smoke checks

## 10.1 Python compile check
```bash
python -m py_compile scripts/daily_orchestrator.py
```

## 10.2 Run quick forecast smoke test
```bash
PYTHONPATH=. python scripts/generate_gap_augmented_hybrid_forecast.py
```

## 10.3 Validate output exists
```bash
ls -lah output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json
```

## 10.4 Pretty-print forecast
```bash
python -m json.tool output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json
```

---

## 11. Common failure modes and fixes

## 11.1 `DATABENTO_API_KEY` missing
### Symptom
Fetch scripts fail with auth error or missing key error.

### Fix
Ensure `.env` contains:
```env
DATABENTO_API_KEY=db-...
```

Then rerun.

---

## 11.2 Databento raw file missing
### Symptom
Overnight feature build fails because `data/raw/es_databento_1m.parquet` is missing.

### Fix
Run:
```bash
PYTHONPATH=. python scripts/fetch_es_databento_1m.py --start 2024-01-01 --end 2026-03-08 --out data/raw/es_databento_1m.parquet
```

Then:
```bash
PYTHONPATH=. python scripts/build_es_databento_overnight_features.py
```

---

## 11.3 Gap model file missing
### Symptom
Gap-augmented forecast generation fails because `gap_databento_model.joblib` is missing.

### Fix
Retrain:
```bash
PYTHONPATH=. python scripts/train_databento_gap_model.py
```

---

## 11.4 Base OHLC models missing
### Symptom
Forecast generator fails loading OHLC component models.

### Fix
Retrain:
```bash
PYTHONPATH=. python scripts/train_ohlc_models.py
```

---

## 11.5 Reconciliation says no actual SPX data yet
### Symptom
RuntimeError or message indicating no actual SPX OHLC found for forecast date.

### Fix
Wait until `data/raw/spx_daily.parquet` includes that date, then rerun:
```bash
PYTHONPATH=. python scripts/reconcile_hybrid_forecast.py
```

---

## 11.6 LightGBM warnings
Typical warning:
```text
No further splits with positive gain, best gain: -inf
```

### Meaning
This often happens in small or narrow-signal samples and does not automatically mean the run failed.

### Action
- if output files are created and metrics are produced, treat as warning not fatal
- review metrics instead of reacting to the warning alone

---

## 12. Why Databento is used narrowly

Databento full-feature integration was tested and was **not** promoted globally because:

- full Databento feature merge improved open somewhat
- but degraded several other OHLC metrics
- especially range behavior

The successful pattern was:

- use a **small subset** of Databento overnight/pre-open features
- apply them **only to the gap target**

Winning Databento gap features:
- `es_overnight_ret`
- `es_preopen_ret_last_60m`
- `es_preopen_ret_last_30m`
- `es_overnight_range_pct`

This is the current recommended operational policy.

---

## 13. Files that should and should not be committed

## 13.1 Commit these
Source code, scripts, tests, docs:
- `src/...`
- `scripts/...`
- `tests/...`
- `docs/...`

## 13.2 Do NOT commit these runtime artifacts
- `output/models/...`
- `output/reports/...`
- `output/forecasts/...`
- `output/analysis/...`

Unless you are intentionally versioning a report snapshot, these should stay uncommitted.

---

## 14. Suggested pre-push checklist

Run:

```bash
python -m py_compile scripts/daily_orchestrator.py
PYTHONPATH=. pytest -q
PYTHONPATH=. python scripts/generate_gap_augmented_hybrid_forecast.py
python -m json.tool output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json >/dev/null
git status
```

Review:
- compile passes
- tests pass
- forecast script works
- output JSON is valid
- no accidental runtime artifacts are staged

---

## 15. Suggested daily operator workflow

### Before market / night before
```bash
source .venv/bin/activate
cd ~/spx_algo
PYTHONPATH=. python scripts/run_gap_augmented_hybrid_forecast_step.py
python -m json.tool output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json
```

### After market / after SPX daily data refresh
```bash
source .venv/bin/activate
cd ~/spx_algo
PYTHONPATH=. python scripts/reconcile_hybrid_forecast.py
PYTHONPATH=. python scripts/build_forecast_scorecard.py
```

---

## 16. Resume context for future work

When resuming this project later, remember:

### Confirmed best architecture
- gap from Databento gap model
- high/low/close from existing OHLC models

### Confirmed findings
- full Databento feature merge was too broad and hurt several metrics
- targeted Databento gap augmentation improved the system
- Databento is useful, but only when applied narrowly to gap/open-sensitive forecasting

### Best next future enhancements
- reconcile and score the **gap-augmented** forecast separately
- eventually make the gap-augmented forecast the sole default if live performance confirms the offline gains
- optionally standardize JSON output keys from `pred_open`/`pred_high` to `open`/`high`

---

## 17. One-command quick reference

### Generate tomorrow’s gap-augmented forecast
```bash
PYTHONPATH=. python scripts/generate_gap_augmented_hybrid_forecast.py
```

### Generate + archive forecast
```bash
PYTHONPATH=. python scripts/run_gap_augmented_hybrid_forecast_step.py
```

### Run full orchestrator
```bash
PYTHONPATH=. python scripts/daily_orchestrator.py
```

### Reconcile after actuals exist
```bash
PYTHONPATH=. python scripts/reconcile_hybrid_forecast.py
```

### Build scorecard
```bash
PYTHONPATH=. python scripts/build_forecast_scorecard.py
```

### Retrain base OHLC models
```bash
PYTHONPATH=. python scripts/train_ohlc_models.py
```

### Retrain Databento gap model
```bash
PYTHONPATH=. python scripts/train_databento_gap_model.py
```

### Refresh Databento intraday data
```bash
PYTHONPATH=. python scripts/fetch_es_databento_1m.py --start 2024-01-01 --end 2026-03-08 --out data/raw/es_databento_1m.parquet
```

### Rebuild Databento overnight features
```bash
PYTHONPATH=. python scripts/build_es_databento_overnight_features.py
```

---

## 18. Owner notes
This repo now contains both:
- legacy hybrid forecast workflow
- Databento gap-augmented hybrid forecast workflow

Operationally, the Databento gap-augmented path is the preferred one.
The legacy path is still kept as fallback / comparison until enough live reconciled observations accumulate.
```

If you want, I can also provide:
1. a shorter `docs/OPERATOR_CHECKLIST.md`, or  
2. the exact terminal commands to create `docs/RUNBOOK.md`, commit it, and push.
