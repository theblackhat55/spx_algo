#!/usr/bin/env bash
set -euo pipefail

cd /root/spx_algo
. /root/spx_algo/.venv/bin/activate
export PYTHONPATH=/root/spx_algo

mkdir -p /root/spx_algo/logs

echo "[$(date -Is)] starting daily signal job"
bash /root/spx_algo/scripts/daily_cron.sh
echo "[$(date -Is)] finished daily signal job"
