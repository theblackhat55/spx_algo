#!/usr/bin/env bash
# =============================================================================
# scripts/daily_pipeline.sh
#
# Daily SPX Algo pipeline runner.
# Runs the full sequence:
#   1. Data fetch (live market data + VIX)
#   2. Feature engineering
#   3. Signal generation
#   4. Optional: paper-trade outcome logging (previous day)
#
# Designed to be invoked by cron, systemd, or daily_orchestrator.py.
#
# Usage:
#   ./scripts/daily_pipeline.sh [--mode paper|live] [--date YYYY-MM-DD]
#
# Environment variables (optional overrides):
#   EXECUTION_MODE  - "paper" (default) or "live"
#   SIGNAL_DATE     - Override signal date (YYYY-MM-DD)
#   SIGNAL_DIR      - Output directory for signals (default: output/signals)
#   LOG_DIR         - Directory for log files (default: output/logs)
# =============================================================================
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
EXECUTION_MODE="${EXECUTION_MODE:-paper}"
SIGNAL_DATE="${SIGNAL_DATE:-$(date +%Y-%m-%d)}"
SIGNAL_DIR="${SIGNAL_DIR:-$REPO_ROOT/output/signals}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/output/logs}"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)  EXECUTION_MODE="$2"; shift 2 ;;
        --date)  SIGNAL_DATE="$2";    shift 2 ;;
        *)       echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$SIGNAL_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_pipeline_${SIGNAL_DATE}.log"
PYTHONPATH="$REPO_ROOT"
export PYTHONPATH

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "SPX Algo Daily Pipeline"
echo "Date         : $SIGNAL_DATE"
echo "Mode         : $EXECUTION_MODE"
echo "Repo root    : $REPO_ROOT"
echo "Signal dir   : $SIGNAL_DIR"
echo "Log file     : $LOG_FILE"
echo "Started at   : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

cd "$REPO_ROOT"

# ── Step 1: Data fetch ────────────────────────────────────────────────────────
echo ""
echo "── Step 1: Data fetch ─────────────────────────────────────"
python3 -c "
from src.data.live_fetcher import run_daily_fetch
run_daily_fetch()
print('Data fetch complete.')
"

# ── Step 2: Feature engineering ───────────────────────────────────────────────
echo ""
echo "── Step 2: Feature engineering ────────────────────────────"
python3 -c "
import pandas as pd
from pathlib import Path
from src.features.engineer import build_features

spx = pd.read_parquet('data/raw/spx_daily.parquet')
feats = build_features(spx)
Path('data/processed').mkdir(parents=True, exist_ok=True)
feats.to_parquet('data/processed/features.parquet')
print(f'Features built: {feats.shape[0]} rows × {feats.shape[1]} cols')
"

# ── Step 3: Signal generation ─────────────────────────────────────────────────
echo ""
echo "── Step 3: Signal generation ───────────────────────────────"
python3 -c "
import json
from pathlib import Path
from src.pipeline.signal_generator import SignalGenerator

gen = SignalGenerator(mode='$EXECUTION_MODE')
sig = gen.generate(as_of_date='$SIGNAL_DATE')
if sig is None:
    raise RuntimeError('SignalGenerator returned None')

out_dir = Path('$SIGNAL_DIR')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f'signal_$SIGNAL_DATE.json'
out_path.write_text(sig.to_json(), encoding='utf-8')

latest = out_dir / 'latest_signal.json'
if latest.is_symlink(): latest.unlink()
latest.symlink_to(out_path.name)

print(f'Signal written: {out_path}')
print(f'Tradeable: {sig.tradeable}  Regime: {sig.regime}')
print(f'Notes: {sig.notes}')
"

echo ""
echo "============================================================"
echo "Daily pipeline COMPLETE at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
