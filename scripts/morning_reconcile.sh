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

recon = Reconciler()
result = recon.run()
print(f'Reconciliation: {result}')
"

echo ""
echo "===== ERROR CORRECTION UPDATE ====="
python3 scripts/update_error_correction.py

echo ""
echo "===== $(date) ===== RECONCILIATION COMPLETE ====="
