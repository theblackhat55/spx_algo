```markdown
# SPX Algo

Production-grade SPX forecasting and signal-generation system for daily monitoring, forecasting, comparison, and dashboard publishing.

This repo currently supports two related workflows:

1. Daily SPX signal pipeline
2. Daily SPX forecast pipeline

The live system now includes:
- daily signal generation
- gap-augmented hybrid OHLC forecast generation
- range+skew overlay forecast generation
- archived forecast comparison versus actual SPX OHLC
- Databento ES 1-minute / overnight feature pipeline
- dashboard integration
- health-check workflow

---

## Current production outputs

### Signals
- `output/signals/latest_signal.json`

### Forecasts
- `output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json`
- `output/forecasts/latest_gap_augmented_range_skew_forecast.json`
- `output/forecasts/archive/<DATE>_gap_augmented_hybrid_ohlc_forecast.json`
- `output/forecasts/archive/<DATE>_gap_augmented_range_skew_forecast.json`

### Comparison / reports
- `output/reports/daily_forecast_comparison/daily_hybrid_vs_range_skew_scorecard.csv`
- `output/reports/daily_forecast_comparison/<DATE>_hybrid_vs_range_skew_actuals.json`

### Logs
- `logs/daily_signal.log`
- `logs/forecast_cron.log`
- `logs/comparison_cron.log`

---

## Current data stack

### Raw data
- `data/raw/spx_daily.parquet`
- `data/raw/vix_daily.parquet`
- `data/raw/es_daily.parquet`
- `data/raw/es_5m_recent.parquet`
- `data/raw/es_databento_1m.parquet`

### Processed data
- `data/processed/features.parquet`
- `data/processed/es_overnight_features.parquet`
- `data/processed/es_databento_overnight_features.parquet`

### Feature matrix
The current operational feature matrix is **96 features**.

---

## Repo structure

```text
data/raw/                      SPX / VIX / ES raw market data
data/processed/                feature and overnight processed artifacts

output/models/                 trained model artifacts
output/signals/                latest signal and historical signal files
output/forecasts/              latest forecast JSONs
output/forecasts/archive/      archived dated forecasts
output/reports/                comparisons, diagnostics, evaluation outputs
logs/                          daily pipeline logs

scripts/                       operational wrappers, fetchers, trainers, evaluation scripts
src/                           core feature, model, and pipeline code
tests/                         automated tests
docs/                          repo documentation / runbooks
```

---

## Setup

### Prerequisites
- Python 3.11+
- Linux recommended
- Databento access required for the hybrid forecast path

### Installation
```bash
git clone https://github.com/theblackhat55/spx_algo.git
cd spx_algo
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Update `.env` with required credentials.

---

## Main operational commands

### Rebuild base features
```bash
python3.11 -m src.features.builder
```

### Run daily signal pipeline
```bash
sudo /root/spx_algo/scripts/cron_run_daily_signal.sh
```

### Run forecast pipeline
```bash
sudo /root/spx_algo/scripts/cron_run_forecast.sh
```

### Run daily comparison pipeline
```bash
sudo /root/spx_algo/scripts/cron_run_comparison.sh
```

---

## Forecast pipeline overview

The current morning forecast path is:

1. refresh SPX / VIX raw data
2. refresh ES raw data
3. refresh Databento ES 1-minute data
4. rebuild ES overnight features
5. rebuild Databento overnight features
6. rebuild base feature matrix
7. generate hybrid OHLC forecast
8. generate range+skew overlay forecast
9. archive latest forecast outputs

Key scripts:
- `scripts/fetch_es_daily.py`
- `scripts/fetch_es_intraday_recent.py`
- `scripts/fetch_es_databento_1m.py`
- `scripts/build_es_overnight_features.py`
- `scripts/build_es_databento_overnight_features.py`
- `scripts/run_gap_augmented_hybrid_forecast_step.py`
- `scripts/run_gap_augmented_range_skew_forecast_step.py`
- `scripts/cron_run_forecast.sh`

---

## Databento notes

The Databento ES workflow is now part of the live forecast path.

Current behavior:
- `fetch_es_databento_1m.py` supports delta-append
- processed Databento overnight features are written to:
  - `data/processed/es_databento_overnight_features.parquet`
- the hybrid forecast uses the latest overlapping date between base features and Databento overnight features
- the Databento premarket feature window was aligned for morning forecast use

---

## Daily expected state

### After morning forecast run
Expected:
- forecast JSONs updated
- `generated_from_feature_date = current trading day`
- `forecast_for_date = next trading day`

### After daily signal run
Expected:
- `latest_signal.json` updated
- `signal_date = current trading day`

### After comparison run
Expected:
- scorecard CSV updated
- dated comparison JSON written
- archived forecast matched to actuals

---

## Health checks

### Full shell health check
```bash
/root/spx_algo/scripts/health_check_spx.sh
```

### Check latest forecast dates
```bash
python3.11 - <<'PY'
import json
for p in [
    "/root/spx_algo/output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json",
    "/root/spx_algo/output/forecasts/latest_gap_augmented_range_skew_forecast.json",
]:
    x = json.load(open(p))
    print("\nFILE:", p)
    print("forecast_for_date:", x.get("forecast_for_date"))
    print("generated_from_feature_date:", x.get("generated_from_feature_date"))
PY
```

### Check latest signal
```bash
python3.11 - <<'PY'
import json
p = "/root/spx_algo/output/signals/latest_signal.json"
x = json.load(open(p))
print(x.get("signal_date"), x.get("regime"), x.get("direction"), x.get("tradeable"))
PY
```

### Check comparison freshness
```bash
stat /root/spx_algo/output/reports/daily_forecast_comparison/daily_hybrid_vs_range_skew_scorecard.csv
tail -n 50 /root/spx_algo/logs/comparison_cron.log
```

---

## Training

### Databento gap model
```bash
python3.11 scripts/train_databento_gap_model.py
```

Model artifact:
- `output/models/ohlc/gap_databento_model.joblib`

### Tests
```bash
pytest tests/ -v --tb=short
```

---

## Dashboard integration

The shared trading dashboard reads canonical latest files such as:
- `output/signals/latest_signal.json`
- `output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json`
- `output/forecasts/latest_gap_augmented_range_skew_forecast.json`
- `output/reports/daily_forecast_comparison/daily_hybrid_vs_range_skew_scorecard.csv`

If dashboard data appears stale:
1. verify upstream scripts succeeded
2. verify canonical latest files updated
3. restart dashboard if needed

---

## Additional docs

See:
- `docs/DAILY_RUNBOOK.md`

---

## License

MIT
```
