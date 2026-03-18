```markdown
# SPX Daily Runbook

## Daily flow

### Morning forecast run
Wrapper:
- `scripts/cron_run_forecast.sh`

Expected outputs:
- `output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json`
- `output/forecasts/latest_gap_augmented_range_skew_forecast.json`

Expected fields:
- `generated_from_feature_date = current trading day`
- `forecast_for_date = next trading day`

### Post-close signal run
Wrapper:
- `scripts/cron_run_daily_signal.sh`

Expected output:
- `output/signals/latest_signal.json`

Expected field:
- `signal_date = current trading day`

### Comparison run
Wrapper:
- `scripts/cron_run_comparison.sh`

Expected outputs:
- dated actual-vs-forecast JSON report
- updated `daily_hybrid_vs_range_skew_scorecard.csv`

---

## One-minute checks

### Forecast dates
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

### Signal date
```bash
python3.11 - <<'PY'
import json
p = "/root/spx_algo/output/signals/latest_signal.json"
x = json.load(open(p))
print(x.get("signal_date"), x.get("regime"), x.get("direction"), x.get("tradeable"))
PY
```

### Scorecard freshness
```bash
stat /root/spx_algo/output/reports/daily_forecast_comparison/daily_hybrid_vs_range_skew_scorecard.csv
```

### Full health check
```bash
/root/spx_algo/scripts/health_check_spx.sh
```

---

## Troubleshooting

### Forecast stuck on old date
Check:
```bash
python3.11 - <<'PY'
import pandas as pd
for p in [
    "/root/spx_algo/data/processed/features.parquet",
    "/root/spx_algo/data/processed/es_overnight_features.parquet",
    "/root/spx_algo/data/processed/es_databento_overnight_features.parquet",
]:
    df = pd.read_parquet(p)
    print("\nFILE:", p)
    print("index max:", df.index.max())
PY
```

Then inspect:
```bash
tail -n 120 /root/spx_algo/logs/forecast_cron.log
```

### Comparison failing for missing archive
Check:
```bash
ls -lt /root/spx_algo/output/forecasts/archive | head -30
tail -n 120 /root/spx_algo/logs/comparison_cron.log
```

### Dashboard stale
```bash
systemctl restart trading-dashboard
journalctl -u trading-dashboard -n 30 --no-pager
```

---

## Useful scripts

- `scripts/cron_run_forecast.sh`
- `scripts/cron_run_daily_signal.sh`
- `scripts/cron_run_comparison.sh`
- `scripts/train_databento_gap_model.py`
- `scripts/fetch_es_databento_1m.py`
- `scripts/build_es_databento_overnight_features.py`
```
