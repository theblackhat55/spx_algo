#!/usr/bin/env bash
set -euo pipefail

echo "================ SPX DAILY HEALTH CHECK ================"
echo "Host: $(hostname)"
echo "Now : $(date -Is)"
echo

python3.11 - <<'PY'
from pathlib import Path
import json, time
import pandas as pd

def stat_line(path_str):
    p = Path(path_str)
    print(f"\nFILE: {p}")
    if not p.exists() and not p.is_symlink():
        print("  MISSING")
        return
    try:
        st = p.stat()
        print("  mtime:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)))
        print("  size :", st.st_size)
    except Exception as e:
        print("  stat error:", e)

def json_dates(path_str, keys):
    p = Path(path_str)
    print(f"\nJSON: {p}")
    if not p.exists():
        print("  MISSING")
        return
    try:
        x = json.load(open(p))
        for k in keys:
            if k in x:
                print(f"  {k}: {x[k]}")
    except Exception as e:
        print("  read error:", e)

def parquet_max(path_str):
    p = Path(path_str)
    print(f"\nPARQUET: {p}")
    if not p.exists():
        print("  MISSING")
        return
    try:
        df = pd.read_parquet(p)
        print("  shape:", df.shape)
        if "date" in df.columns:
            print("  max(date):", df["date"].max())
        else:
            print("  index max:", df.index.max())
    except Exception as e:
        print("  read error:", e)

# Canonical outputs
for p in [
    "/root/spx_algo/output/signals/latest_signal.json",
    "/root/spx_algo/output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json",
    "/root/spx_algo/output/forecasts/latest_gap_augmented_range_skew_forecast.json",
    "/root/spx_algo/output/reports/daily_forecast_comparison/daily_hybrid_vs_range_skew_scorecard.csv",
]:
    stat_line(p)

# JSON content dates
json_dates(
    "/root/spx_algo/output/signals/latest_signal.json",
    ["signal_date", "regime", "direction", "tradeable"],
)
json_dates(
    "/root/spx_algo/output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json",
    ["forecast_for_date", "generated_from_feature_date"],
)
json_dates(
    "/root/spx_algo/output/forecasts/latest_gap_augmented_range_skew_forecast.json",
    ["forecast_for_date", "generated_from_feature_date"],
)

# Upstream freshness
for p in [
    "/root/spx_algo/data/raw/spx_daily.parquet",
    "/root/spx_algo/data/raw/vix_daily.parquet",
    "/root/spx_algo/data/raw/es_daily.parquet",
    "/root/spx_algo/data/raw/es_5m_recent.parquet",
    "/root/spx_algo/data/raw/es_databento_1m.parquet",
    "/root/spx_algo/data/processed/es_overnight_features.parquet",
    "/root/spx_algo/data/processed/es_databento_overnight_features.parquet",
    "/root/spx_algo/data/processed/features.parquet",
]:
    parquet_max(p)
PY

echo
echo "---------------- Recent logs ----------------"
for f in \
  /root/spx_algo/logs/daily_signal.log \
  /root/spx_algo/logs/forecast_cron.log \
  /root/spx_algo/logs/comparison_cron.log
do
  echo
  echo "LOG: $f"
  if [[ -f "$f" ]]; then
    tail -n 20 "$f"
  else
    echo "MISSING"
  fi
done

echo
echo "---------------- Quick verdict guide ----------------"
cat <<'TXT'
Healthy expected state:
- latest_signal.json exists and signal_date is current trading day
- latest forecast JSONs exist and:
    generated_from_feature_date = current trading day
    forecast_for_date = next trading day
- scorecard CSV mtime updates after comparison job
- features.parquet and overnight feature parquets are current
- log tails end with successful completion, not traceback/errors
TXT

echo
echo "================ END HEALTH CHECK ================"
