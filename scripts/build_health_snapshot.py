#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/root/spx_algo")
OUT = ROOT / "output" / "monitoring" / "health_snapshot.json"

SIGNAL_PATH = ROOT / "output" / "signals" / "latest_signal.json"
HYBRID_FORECAST_PATH = ROOT / "output" / "forecasts" / "latest_gap_augmented_hybrid_ohlc_forecast.json"
RANGE_SKEW_FORECAST_PATH = ROOT / "output" / "forecasts" / "latest_gap_augmented_range_skew_forecast.json"
SCORECARD_PATH = ROOT / "output" / "reports" / "daily_forecast_comparison" / "daily_hybrid_vs_range_skew_scorecard.csv"

FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
DB_OVERNIGHT_PATH = ROOT / "data" / "processed" / "es_databento_overnight_features.parquet"
ES_DAILY_PATH = ROOT / "data" / "raw" / "es_daily.parquet"
ES_5M_PATH = ROOT / "data" / "raw" / "es_5m_recent.parquet"
DB_1M_PATH = ROOT / "data" / "raw" / "es_databento_1m.parquet"


def iso_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parquet_max_index(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        idx = df.index
        if "Date" in df.columns:
            idx = pd.to_datetime(df["Date"])
        else:
            idx = pd.to_datetime(idx)
        return str(idx.max())
    except Exception:
        return None


def latest_scorecard_date(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        if "forecast_for_date" in df.columns:
            return str(pd.to_datetime(df["forecast_for_date"]).max().date())
        return None
    except Exception:
        return None


def main() -> None:
    reasons: list[str] = []

    signal = load_json(SIGNAL_PATH)
    hybrid = load_json(HYBRID_FORECAST_PATH)
    range_skew = load_json(RANGE_SKEW_FORECAST_PATH)

    signal_status = "OK"
    forecast_status = "OK"
    comparison_status = "OK"

    if signal is None:
        signal_status = "BAD"
        reasons.append("latest_signal.json missing")

    if hybrid is None or range_skew is None:
        forecast_status = "BAD"
        reasons.append("one or more latest forecast files missing")
    else:
        h_ffd = hybrid.get("forecast_for_date")
        h_gfd = hybrid.get("generated_from_feature_date")
        r_ffd = range_skew.get("forecast_for_date")
        r_gfd = range_skew.get("generated_from_feature_date")

        if not h_ffd or not h_gfd or not r_ffd or not r_gfd:
            forecast_status = "BAD"
            reasons.append("forecast json missing key date fields")
        elif h_ffd != r_ffd or h_gfd != r_gfd:
            forecast_status = "WARN"
            reasons.append("hybrid/range-skew forecast dates do not match")

    if not SCORECARD_PATH.exists():
        comparison_status = "WARN"
        reasons.append("comparison scorecard csv missing")

    overall = "OK"
    if "BAD" in {signal_status, forecast_status, comparison_status}:
        overall = "BAD"
    elif "WARN" in {signal_status, forecast_status, comparison_status}:
        overall = "WARN"

    payload = {
        "as_of": datetime.now().astimezone().isoformat(),
        "overall_status": overall,
        "signal_status": signal_status,
        "forecast_status": forecast_status,
        "comparison_status": comparison_status,
        "signal": {
            "path": str(SIGNAL_PATH),
            "mtime": iso_mtime(SIGNAL_PATH),
            "signal_date": None if signal is None else signal.get("signal_date"),
            "regime": None if signal is None else signal.get("regime"),
            "direction": None if signal is None else signal.get("direction"),
            "tradeable": None if signal is None else signal.get("tradeable"),
        },
        "forecasts": {
            "hybrid_path": str(HYBRID_FORECAST_PATH),
            "range_skew_path": str(RANGE_SKEW_FORECAST_PATH),
            "hybrid_mtime": iso_mtime(HYBRID_FORECAST_PATH),
            "range_skew_mtime": iso_mtime(RANGE_SKEW_FORECAST_PATH),
            "forecast_for_date": None if hybrid is None else hybrid.get("forecast_for_date"),
            "generated_from_feature_date": None if hybrid is None else hybrid.get("generated_from_feature_date"),
        },
        "comparison": {
            "scorecard_path": str(SCORECARD_PATH),
            "scorecard_mtime": iso_mtime(SCORECARD_PATH),
            "latest_scorecard_forecast_date": latest_scorecard_date(SCORECARD_PATH),
        },
        "inputs": {
            "features_max_index": parquet_max_index(FEATURES_PATH),
            "es_databento_overnight_max_index": parquet_max_index(DB_OVERNIGHT_PATH),
            "es_daily_max_index": parquet_max_index(ES_DAILY_PATH),
            "es_5m_recent_max_index": parquet_max_index(ES_5M_PATH),
            "es_databento_1m_max_index": parquet_max_index(DB_1M_PATH),
        },
        "reasons": reasons,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved:", OUT)
    print("overall_status:", payload["overall_status"])
    print("signal_status:", payload["signal_status"])
    print("forecast_status:", payload["forecast_status"])
    print("comparison_status:", payload["comparison_status"])


if __name__ == "__main__":
    main()
