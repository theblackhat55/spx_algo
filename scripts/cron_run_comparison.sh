#!/usr/bin/env bash
set -euo pipefail

cd /root/spx_algo
. /root/spx_algo/.venv/bin/activate
export PYTHONPATH=/root/spx_algo

mkdir -p /root/spx_algo/logs

RUN_DATE="${1:-$(date +%F)}"

echo "[$(date -Is)] starting comparison job for ${RUN_DATE}"
python scripts/run_daily_forecast_comparison_step.py --date "${RUN_DATE}" || {
  status=$?
  echo "[$(date -Is)] comparison job returned non-zero status ${status} for ${RUN_DATE}"
  exit "${status}"
}
echo "[$(date -Is)] finished comparison job for ${RUN_DATE}"
