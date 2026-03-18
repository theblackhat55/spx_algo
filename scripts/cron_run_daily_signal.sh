#!/bin/bash
set -euo pipefail
cd /root/spx_algo
mkdir -p /root/spx_algo/logs
exec >> /root/spx_algo/logs/daily_signal.log 2>&1
echo "[$(date -Is)] START daily signal"
. /root/spx_algo/.venv/bin/activate
export PYTHONPATH=/root/spx_algo
bash /root/spx_algo/scripts/daily_cron.sh
echo "[$(date -Is)] END daily signal"
