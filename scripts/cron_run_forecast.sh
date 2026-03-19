#!/bin/bash
set -euo pipefail
cd /root/spx_algo
mkdir -p /root/spx_algo/logs
exec >> /root/spx_algo/logs/forecast_cron.log 2>&1

echo "[$(date -Is)] START forecast"
. /root/spx_algo/.venv/bin/activate
export PYTHONPATH=/root/spx_algo

echo "[$(date -Is)] STEP refresh SPX/VIX raw data"
python3.11 << 'PYEOF'
import yfinance as yf
import pandas as pd

for ticker, name in [('^GSPC', 'spx_daily'), ('^VIX', 'vix_daily')]:
    df = yf.download(ticker, start='2000-01-01', auto_adjust=True, progress=False)
    df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df.to_parquet(f'data/raw/{name}.parquet')
    print(f'{name}: {len(df)} rows through {df.index[-1].date()}')
PYEOF

echo "[$(date -Is)] STEP rebuild overnight features"
python3.11 /root/spx_algo/scripts/build_es_databento_overnight_features.py
python3.11 /root/spx_algo/scripts/build_es_overnight_features.py || true

echo "[$(date -Is)] STEP rebuild base features"
python3.11 -m src.features.builder

echo "[$(date -Is)] STEP hybrid forecast"
python3.11 /root/spx_algo/scripts/run_gap_augmented_hybrid_forecast_step.py

echo "[$(date -Is)] STEP range+skew forecast"
python3.11 /root/spx_algo/scripts/run_gap_augmented_range_skew_forecast_step.py

echo "[$(date -Is)] END forecast"

python3.11 scripts/build_health_snapshot.py
python3.11 scripts/render_ops_summary.py
