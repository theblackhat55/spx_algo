#!/bin/bash
set -euo pipefail
cd /root/spx_algo
mkdir -p /root/spx_algo/logs
exec >> /root/spx_algo/logs/comparison_cron.log 2>&1
echo "[$(date -Is)] START comparison"
. /root/spx_algo/.venv/bin/activate
export PYTHONPATH=/root/spx_algo
python /root/spx_algo/scripts/run_daily_forecast_comparison_step.py --date "$(date +%F)" || true
echo "[$(date -Is)] END comparison"

python3.11 scripts/build_forecast_monitor_snapshot.py
python3.11 scripts/build_retraining_recommendation.py
python3.11 scripts/render_ops_summary.py
