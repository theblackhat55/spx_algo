#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/root/spx_algo")
MON_DIR = ROOT / "output" / "monitoring"
HEALTH_PATH = MON_DIR / "health_snapshot.json"
MONITOR_PATH = MON_DIR / "forecast_monitor_snapshot.json"
RECO_PATH = MON_DIR / "retraining_recommendation.json"
ACTION_LOG_PATH = MON_DIR / "action_log.jsonl"
OUT_PATH = MON_DIR / "last_action_evaluation.json"

def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def _load_last_action() -> dict[str, Any] | None:
    if not ACTION_LOG_PATH.exists():
        return None
    lines = [x for x in ACTION_LOG_PATH.read_text().splitlines() if x.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except Exception:
        return None

def main() -> None:
    health = _load_json(HEALTH_PATH) or {}
    monitor = _load_json(MONITOR_PATH) or {}
    reco = _load_json(RECO_PATH) or {}
    last_action = _load_last_action() or {}

    payload = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "last_action": last_action,
        "health_overall_status": health.get("overall_status"),
        "forecast_status": health.get("forecast_status"),
        "drift_classification": monitor.get("classification"),
        "latest_drift_status": monitor.get("latest_drift_status"),
        "current_recommendation": reco.get("decision"),
        "priority": reco.get("priority"),
        "summary": {
            "health_ok": health.get("overall_status") == "OK",
            "forecast_ok": health.get("forecast_status") == "OK",
            "recommendation_cleared": reco.get("decision") in [None, "", "NO_ACTION"],
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print("Saved:", OUT_PATH)
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
