#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/root/spx_algo")
OUT = ROOT / "output" / "monitoring" / "forecast_monitor_snapshot.json"
COMPARE_DIR = ROOT / "output" / "reports" / "daily_forecast_comparison"
SCORECARD_PATH = COMPARE_DIR / "daily_hybrid_vs_range_skew_scorecard.csv"
DRIFT_LOG_PATH = ROOT / "output" / "monitoring" / "drift_log.csv"
ERROR_HISTORY_PATH = ROOT / "output" / "monitoring" / "error_history.csv"
DRIFT_REPORT_DIR = ROOT / "output" / "reports"


def newest_matching(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[-1] if matches else None


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_scorecard(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "exists": path.exists(),
        "rows": 0,
        "latest_forecast_date": None,
        "rolling_5": {},
        "rolling_20": {},
    }
    if not path.exists():
        return out

    df = pd.read_csv(path)
    if df.empty:
        return out

    out["rows"] = int(len(df))
    if "forecast_for_date" in df.columns:
        out["latest_forecast_date"] = str(pd.to_datetime(df["forecast_for_date"]).max().date())

    def metric_block(sub: pd.DataFrame) -> dict[str, Any]:
        result: dict[str, Any] = {"rows": int(len(sub))}
        candidates = [
            "hybrid_ohlc_mae",
            "range_skew_ohlc_mae",
            "hybrid_range_mae",
            "range_skew_range_mae",
            "hybrid_inside_range_coverage",
            "range_skew_inside_range_coverage",
        ]
        for c in candidates:
            if c in sub.columns:
                result[c] = float(sub[c].mean())
        return result

    out["rolling_5"] = metric_block(df.tail(5))
    out["rolling_20"] = metric_block(df.tail(20))
    return out


def summarize_latest_comparison(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False}

    data = load_json(path) or {}
    result: dict[str, Any] = {"exists": True, "path": str(path), "keys": sorted(list(data.keys()))[:20]}

    for key in ["forecast_for_date", "generated_from_feature_date"]:
        if key in data:
            result[key] = data.get(key)

    if "mean_ohlc_mae" in data:
        result["mean_ohlc_mae"] = data.get("mean_ohlc_mae")
    if "range_mae" in data:
        result["range_mae"] = data.get("range_mae")
    if "inside_range_coverage" in data:
        result["inside_range_coverage"] = data.get("inside_range_coverage")

    return result


def summarize_drift_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}

    df = pd.read_csv(path)
    if df.empty:
        return {"exists": True, "rows": 0}

    result: dict[str, Any] = {
        "exists": True,
        "rows": int(len(df)),
        "latest_date": str(df.iloc[-1]["date"]) if "date" in df.columns else None,
    }
    if "drift_status" in df.columns:
        result["latest_drift_status"] = str(df.iloc[-1]["drift_status"])
        result["status_counts"] = {str(k): int(v) for k, v in df["drift_status"].value_counts().to_dict().items()}
    if "high_error_pct" in df.columns:
        result["rolling_21_high_error_pct"] = float(df["high_error_pct"].tail(21).mean())
    if "low_error_pct" in df.columns:
        result["rolling_21_low_error_pct"] = float(df["low_error_pct"].tail(21).mean())
    return result


def summarize_error_history(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    df = pd.read_csv(path)
    if df.empty:
        return {"exists": True, "rows": 0}
    out: dict[str, Any] = {"exists": True, "rows": int(len(df))}
    for c in ["error_high", "error_low", "high_error", "low_error"]:
        if c in df.columns:
            out[f"rolling_20_{c}_mean"] = float(df[c].tail(20).mean())
    return out


def main() -> None:
    latest_compare = newest_matching(COMPARE_DIR, "*_hybrid_vs_range_skew_actuals.json")
    latest_drift_report = newest_matching(DRIFT_REPORT_DIR, "drift_*.json")

    scorecard = summarize_scorecard(SCORECARD_PATH)
    latest_compare_summary = summarize_latest_comparison(latest_compare)
    latest_drift_summary = load_json(latest_drift_report) or {}
    drift_log_summary = summarize_drift_log(DRIFT_LOG_PATH)
    error_history_summary = summarize_error_history(ERROR_HISTORY_PATH)

    classification = "STABLE"
    evidence: list[str] = []

    if drift_log_summary.get("latest_drift_status") in {"WARNING", "DEGRADED"}:
        classification = "DRIFT_ALERT"
        evidence.append(f"legacy drift status = {drift_log_summary.get('latest_drift_status')}")

    r20 = scorecard.get("rolling_20", {})
    if r20.get("hybrid_range_mae") is not None and r20.get("range_skew_range_mae") is not None:
        if float(r20["hybrid_range_mae"]) > 25 or float(r20["range_skew_range_mae"]) > 20:
            classification = "RANGE_DRIFT"
            evidence.append("rolling 20-day range MAE elevated")

    payload = {
        "as_of": datetime.now().astimezone().isoformat(),
        "classification": classification,
        "evidence": evidence,
        "scorecard": scorecard,
        "latest_comparison": latest_compare_summary,
        "latest_drift_report_path": None if latest_drift_report is None else str(latest_drift_report),
        "latest_drift_report": latest_drift_summary,
        "drift_log": drift_log_summary,
        "error_history": error_history_summary,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved:", OUT)
    print("classification:", payload["classification"])


if __name__ == "__main__":
    main()
