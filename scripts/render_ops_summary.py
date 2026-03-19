#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/spx_algo")
HEALTH_PATH = ROOT / "output" / "monitoring" / "health_snapshot.json"
MONITOR_PATH = ROOT / "output" / "monitoring" / "forecast_monitor_snapshot.json"
RECO_PATH = ROOT / "output" / "monitoring" / "retraining_recommendation.json"

OUT_MD = ROOT / "output" / "monitoring" / "daily_ops_summary.md"
OUT_JSON = ROOT / "output" / "monitoring" / "daily_ops_summary.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    health = load_json(HEALTH_PATH)
    monitor = load_json(MONITOR_PATH)
    reco = load_json(RECO_PATH)

    lines = []
    lines.append(f"# SPX Daily Ops Summary")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().astimezone().isoformat()}")
    lines.append(f"- Overall health: {health.get('overall_status', 'UNKNOWN')}")
    lines.append(f"- Drift classification: {monitor.get('classification', 'UNKNOWN')}")
    lines.append(f"- Recommendation: {reco.get('decision', 'UNKNOWN')}")
    lines.append("")

    lines.append("## Health")
    lines.append(f"- Signal status: {health.get('signal_status', 'UNKNOWN')}")
    lines.append(f"- Forecast status: {health.get('forecast_status', 'UNKNOWN')}")
    lines.append(f"- Comparison status: {health.get('comparison_status', 'UNKNOWN')}")
    lines.append(f"- Signal date: {(health.get('signal') or {}).get('signal_date')}")
    lines.append(f"- Forecast for date: {(health.get('forecasts') or {}).get('forecast_for_date')}")
    lines.append(f"- Generated from feature date: {(health.get('forecasts') or {}).get('generated_from_feature_date')}")
    lines.append("")

    lines.append("## Forecast Monitoring")
    lines.append(f"- Latest comparison date: {(monitor.get('scorecard') or {}).get('latest_forecast_date')}")
    lines.append(f"- Latest drift status: {(monitor.get('drift_log') or {}).get('latest_drift_status')}")
    lines.append(f"- Evidence: {', '.join(monitor.get('evidence', [])) if monitor.get('evidence') else 'none'}")
    lines.append("")

    lines.append("## Recommendation")
    lines.append(f"- Decision: {reco.get('decision')}")
    lines.append(f"- Priority: {reco.get('priority')}")
    lines.append(f"- Recommended script: {reco.get('recommended_script')}")
    lines.append(f"- Manual approval required: {reco.get('requires_manual_approval')}")
    if reco.get("reasons"):
        lines.append("- Reasons:")
        for r in reco["reasons"]:
            lines.append(f"  - {r}")
    lines.append("")

    text = "\n".join(lines) + "\n"

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(text, encoding="utf-8")

    summary_json = {
        "as_of": datetime.now().astimezone().isoformat(),
        "overall_health": health.get("overall_status"),
        "drift_classification": monitor.get("classification"),
        "recommendation": reco.get("decision"),
        "summary_markdown_path": str(OUT_MD),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print("Saved:", OUT_MD)
    print("Saved:", OUT_JSON)


if __name__ == "__main__":
    main()
