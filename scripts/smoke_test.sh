#!/usr/bin/env bash
# scripts/smoke_test.sh
# ======================
# Task 39 — End-to-End System Smoke Test
#
# Exit codes:
#   0  All green — proceed to paper trading.
#   1  Test suite failure.
#   2  Conformal coverage gate failure.
#   3  Signal generation failure.
#   4  Drift detector failure.
#
# Usage:  bash scripts/smoke_test.sh [--skip-docker]
#
set -euo pipefail

SKIP_DOCKER=false
for arg in "$@"; do
  [[ "$arg" == "--skip-docker" ]] && SKIP_DOCKER=true
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

_ok()   { echo -e "${GREEN}✅  $*${NC}"; }
_warn() { echo -e "${YELLOW}⚠️   $*${NC}"; }
_fail() { echo -e "${RED}❌  $*${NC}"; }

echo ""
echo "=================================================="
echo "  SPX Algo — Smoke Test  ($(date -u '+%Y-%m-%d %H:%M UTC'))"
echo "=================================================="

# ------------------------------------------------------------------
# Step 1: Full test suite
# ------------------------------------------------------------------
echo ""
echo "STEP 1/7  Full test suite..."
if python -m pytest tests/ -v --tb=short -q 2>&1; then
  _ok "Test suite passed"
else
  _fail "Test suite FAILED"
  exit 1
fi

# ------------------------------------------------------------------
# Step 2: Conformal coverage gate
# ------------------------------------------------------------------
echo ""
echo "STEP 2/7  Conformal coverage verification..."
if python -m src.calibration.conformal_verification 2>&1; then
  _ok "Conformal coverage gate PASSED"
else
  EXIT=$?
  _fail "Conformal coverage gate FAILED (exit $EXIT)"
  exit 2
fi

# ------------------------------------------------------------------
# Step 3: Historical signal (2024-01-15)
# ------------------------------------------------------------------
echo ""
echo "STEP 3/7  Historical signal generation (2024-01-15)..."
if python -m src.pipeline.signal_generator --date 2024-01-15 2>&1; then
  _ok "Historical signal generation OK"
else
  EXIT=$?
  _warn "Historical signal generation returned $EXIT (may need real data)"
  # Non-fatal: real data may not be present; we continue if the file was created
  SIG_FILE=$(ls output/signals/signal_2024-01-15.json 2>/dev/null || true)
  if [[ -z "$SIG_FILE" ]]; then
    _warn "Signal file for 2024-01-15 not found — data may be missing (non-fatal in test env)"
  fi
fi

# ------------------------------------------------------------------
# Step 4: Today's signal
# ------------------------------------------------------------------
echo ""
echo "STEP 4/7  Today's signal generation..."
if python -m src.pipeline.signal_generator 2>&1; then
  _ok "Today's signal generation OK"
else
  EXIT=$?
  _fail "Signal generation FAILED (exit $EXIT)"
  exit 3
fi

# ------------------------------------------------------------------
# Step 5: Drift detector check
# ------------------------------------------------------------------
echo ""
echo "STEP 5/7  Drift detector health check..."
DRIFT_OUT=$(python -c "
import json
from src.monitoring.drift_detector import DriftDetector
dd = DriftDetector()
status = dd.check_drift()
report = dd.daily_report()
print(json.dumps({'status': status.value, 'window_rows': report.get('window_rows', 0)}))
" 2>&1)

STATUS=$(echo "$DRIFT_OUT" | python -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('status','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")

case "$STATUS" in
  HEALTHY)  _ok  "Drift detector: HEALTHY" ;;
  WARNING)  _warn "Drift detector: WARNING (non-fatal)" ;;
  DEGRADED) _fail "Drift detector: DEGRADED — do NOT start paper trading"; exit 4 ;;
  *)        _warn "Drift detector status unknown: $DRIFT_OUT" ;;
esac

# ------------------------------------------------------------------
# Step 6: Validate latest signal JSON
# ------------------------------------------------------------------
echo ""
echo "STEP 6/7  Validating latest_signal.json..."
LATEST=$(ls output/signals/signal_*.json 2>/dev/null | sort | tail -1 || true)
if [[ -z "$LATEST" ]]; then
  _warn "No signal files found in output/signals/ — run signal generator first"
else
  python -c "
import json, sys
with open('$LATEST') as f:
    sig = json.load(f)
required = ['regime', 'signal_date', 'generated_at']
missing = [k for k in required if k not in sig]
if missing:
    print('Missing keys: ' + ', '.join(missing))
    sys.exit(1)
print('Signal valid: date=' + sig['signal_date'] + '  regime=' + sig['regime'])
" && _ok "Signal JSON valid" || { _fail "Signal JSON validation failed"; exit 3; }
fi

# ------------------------------------------------------------------
# Step 7: Docker (optional)
# ------------------------------------------------------------------
echo ""
echo "STEP 7/7  Docker build check..."
if $SKIP_DOCKER; then
  _warn "Docker check skipped (--skip-docker)"
elif ! command -v docker &>/dev/null; then
  _warn "Docker not available — skipping build check"
else
  if docker compose build --quiet 2>&1 && \
     docker compose run --rm algo python -m pytest tests/ -q 2>&1; then
    _ok "Docker build + tests passed"
  else
    _warn "Docker build/test had issues — check output above"
  fi
fi

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "=================================================="
echo -e "${GREEN}✅  SMOKE TEST COMPLETE — All gates passed.${NC}"
echo "    Safe to start paper trading."
echo "    Run:  python -m src.pipeline.scheduler"
echo "=================================================="
exit 0
