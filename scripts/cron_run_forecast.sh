#!/usr/bin/env bash
set -euo pipefail

cd /root/spx_algo
. /root/spx_algo/.venv/bin/activate
export PYTHONPATH=/root/spx_algo

mkdir -p /root/spx_algo/logs

echo "[$(date -Is)] starting forecast job"
python scripts/run_gap_augmented_range_skew_forecast_step.py
echo "[$(date -Is)] finished forecast job"
