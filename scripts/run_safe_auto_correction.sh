#!/usr/bin/env bash
set -euo pipefail

cd /root/spx_algo
. .venv/bin/activate
export PYTHONPATH=/root/spx_algo

ACTION="RUN_UPDATE_ERROR_CORRECTION"
START_REASON="${1:-OpenClaw skill spx_safe_auto_correction started}"
LOG_PATH="/root/spx_algo/output/monitoring/action_log.jsonl"
MAX_RECENT_MINUTES="${MAX_RECENT_MINUTES:-240}"

# ------------------------------------------------------------
# Recency guard: skip if same SUCCESS action ran recently
# ------------------------------------------------------------
if [[ -f "$LOG_PATH" ]]; then
  python3.11 - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone

log_path = Path("/root/spx_algo/output/monitoring/action_log.jsonl")
max_recent_minutes = int(__import__("os").environ.get("MAX_RECENT_MINUTES", "240"))

lines = [x for x in log_path.read_text().splitlines() if x.strip()]
entries = []
for line in lines:
    try:
        entries.append(json.loads(line))
    except Exception:
        pass

entries = [
    e for e in entries
    if e.get("action") == "RUN_UPDATE_ERROR_CORRECTION"
    and e.get("status") == "SUCCESS"
    and e.get("ts")
]

if not entries:
    raise SystemExit(0)

last = entries[-1]
ts = datetime.fromisoformat(last["ts"])
if ts.tzinfo is None:
    ts = ts.replace(tzinfo=timezone.utc)

now = datetime.now(timezone.utc)
age_min = (now - ts).total_seconds() / 60.0

if age_min < max_recent_minutes:
    print(
        f"SKIP: recent successful RUN_UPDATE_ERROR_CORRECTION already ran "
        f"{age_min:.1f} minutes ago at {last['ts']}"
    )
    raise SystemExit(99)
PY
  rc=$?
  if [[ $rc -eq 99 ]]; then
    exit 0
  elif [[ $rc -ne 0 ]]; then
    echo "WARN: recency guard check failed, continuing anyway"
  fi
fi

python3.11 scripts/log_ops_action.py \
  --action "$ACTION" \
  --status STARTED \
  --reason "$START_REASON"

if python3.11 scripts/update_error_correction.py; then
  python3.11 scripts/build_health_snapshot.py
  python3.11 scripts/build_forecast_monitor_snapshot.py
  python3.11 scripts/build_retraining_recommendation.py
  python3.11 scripts/render_ops_summary.py
  python3.11 scripts/log_ops_action.py \
    --action "$ACTION" \
    --status SUCCESS \
    --reason "OpenClaw skill spx_safe_auto_correction completed"
  python3.11 scripts/evaluate_last_action.py
else
  python3.11 scripts/log_ops_action.py \
    --action "$ACTION" \
    --status FAILED \
    --reason "OpenClaw skill spx_safe_auto_correction failed"
  exit 1
fi
