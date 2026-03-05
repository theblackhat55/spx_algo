#!/bin/bash
set -euo pipefail

cd /root/spx_algo
source .venv/bin/activate

LOG="output/reports/reconcile_$(date +%Y%m%d).log"
exec > >(tee -a "$LOG") 2>&1

echo "===== $(date) ===== RECONCILIATION START ====="

python3 -c "
import sys; sys.path.insert(0, '.')
from src.pipeline.reconciler import Reconciler
from datetime import datetime, timedelta

recon = Reconciler()
# Get last trading date (handle weekends)
today = datetime.now()
if today.weekday() == 0:  # Monday
    last_trade = today - timedelta(days=3)
elif today.weekday() == 6:  # Sunday
    last_trade = today - timedelta(days=2)
else:
    last_trade = today - timedelta(days=1)

trade_date = last_trade.strftime('%Y-%m-%d')
print(f'Running reconciliation for: {trade_date}')
result = recon.reconcile(trade_date)
print(f'Reconciliation: {result}')
"

echo ""
echo "===== ERROR CORRECTION UPDATE ====="
python3 scripts/update_error_correction.py

echo ""
echo "===== $(date) ===== RECONCILIATION COMPLETE ====="
