#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/spx_algo")
OUT = ROOT / "output" / "monitoring" / "retraining_recommendation.json"

HEALTH_PATH = ROOT / "output" / "monitoring" / "health_snapshot.json"
MONITOR_PATH = ROOT / "output" / "monitoring" / "forecast_monitor_snapshot.json"

GAP_MODEL_PATH = ROOT / "output" / "models" / "ohlc" / "gap_databento_model.joblib"
WEEKLY_RETRAIN_REPORT = ROOT / "output" / "reports" / "weekly_retrain_latest.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    health = load_json(HEALTH_PATH)
    monitor = load_json(MONITOR_PATH)

    reasons: list[str] = []
    decision = "NO_ACTION"
    priority = "LOW"
    recommended_script = None
    requires_manual_approval = False

    if not health:
        decision = "MANUAL_REVIEW"
        priority = "HIGH"
        reasons.append("health_snapshot.json missing")
    elif health.get("overall_status") == "BAD":
        decision = "MANUAL_REVIEW"
        priority = "HIGH"
        reasons.append("overall health is BAD")
    else:
        classification = monitor.get("classification", "STABLE")
        drift_status = ((monitor.get("drift_log") or {}).get("latest_drift_status"))

        if classification == "DRIFT_ALERT" or drift_status in {"WARNING", "DEGRADED"}:
            decision = "RUN_UPDATE_ERROR_CORRECTION"
            priority = "MEDIUM"
            recommended_script = "scripts/update_error_correction.py"
            reasons.append("legacy drift monitor is elevated")
        elif classification == "RANGE_DRIFT":
            decision = "RUN_WEEKLY_RETRAIN"
            priority = "MEDIUM"
            recommended_script = "scripts/weekly_retrain.py"
            reasons.append("rolling range MAE appears elevated")
        elif not GAP_MODEL_PATH.exists():
            decision = "RUN_GAP_MODEL_RETRAIN"
            priority = "MEDIUM"
            recommended_script = "scripts/train_databento_gap_model.py"
            reasons.append("gap_databento_model.joblib missing")
        elif not WEEKLY_RETRAIN_REPORT.exists():
            decision = "RUN_WEEKLY_RETRAIN"
            priority = "LOW"
            recommended_script = "scripts/weekly_retrain.py"
            reasons.append("weekly retrain report missing")
        else:
            reasons.append("health is acceptable and no strong retraining trigger detected")

    payload = {
        "as_of": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "priority": priority,
        "reasons": reasons,
        "recommended_script": recommended_script,
        "requires_manual_approval": requires_manual_approval,
        "inputs": {
            "health_snapshot_path": str(HEALTH_PATH),
            "forecast_monitor_snapshot_path": str(MONITOR_PATH),
            "gap_model_exists": GAP_MODEL_PATH.exists(),
            "weekly_retrain_report_exists": WEEKLY_RETRAIN_REPORT.exists(),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved:", OUT)
    print("decision:", payload["decision"])
    print("priority:", payload["priority"])


if __name__ == "__main__":
    main()
