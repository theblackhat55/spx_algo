#!/bin/bash
set -euo pipefail

cd /root/spx_algo
source .venv/bin/activate

LOG="output/reports/daily_$(date +%Y%m%d).log"
exec > >(tee -a "$LOG") 2>&1

echo "===== $(date) ===== DAILY PIPELINE START ====="

# 1. Refresh market data
python3.11 << 'PYEOF'
import yfinance as yf
import pandas as pd

for ticker, name in [('^GSPC', 'spx_daily'), ('^VIX', 'vix_daily')]:
    df = yf.download(ticker, start='2000-01-01', auto_adjust=True, progress=False)
    df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df.to_parquet(f'data/raw/{name}.parquet')
    print(f'{name}: {len(df)} rows through {df.index[-1].date()}')
PYEOF

# 2. Rebuild features
python3.11 -m src.features.builder

# 3. Generate signal
python3.11 << 'PYEOF'
import sys; sys.path.insert(0, '.')
from src.pipeline.signal_generator import SignalGenerator
sig = SignalGenerator().generate(mode='paper', save=True)
print(f'Signal: {sig.signal_date} | Regime: {sig.regime} | Direction: {sig.direction} | Tradeable: {sig.tradeable}')
PYEOF

# 4. Log to paper trade CSV
python3.11 << 'PYEOF'
import sys, json; sys.path.insert(0, '.')
from src.execution.paper_logger import PaperTradeLogger
with open('output/signals/latest_signal.json') as f:
    sig = json.load(f)
logger = PaperTradeLogger()
logger.log_signal(sig)
print(f'Logged signal for {sig["signal_date"]}')
PYEOF

echo "===== $(date) ===== DAILY PIPELINE COMPLETE ====="
