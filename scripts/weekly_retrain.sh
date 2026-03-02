#!/bin/bash
set -euo pipefail
cd /root/spx_algo
source .venv/bin/activate
python3 scripts/weekly_retrain.py 2>&1
