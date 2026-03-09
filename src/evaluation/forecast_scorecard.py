from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json

import pandas as pd


REQUIRED_HISTORY_COLUMNS = [
    "forecast_for_date",
    "open_abs_error",
    "high_abs_error",
    "low_abs_error",
    "close_abs_error",
    "range_abs_error",
    "direction_correct",
    "gap_source",
    "high_source",
    "low_source",
    "close_source",
]


def load_forecast_history(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Forecast history file not found: {p}")

    df = pd.read_csv(p)
    missing = [c for c in REQUIRED_HISTORY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Forecast history missing required columns: {missing}")

    df["forecast_for_date"] = pd.to_datetime(df["forecast_for_date"])
    df = df.sort_values("forecast_for_date").reset_index(drop=True)
    return df


def _summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "open_mae": None,
            "high_mae": None,
            "low_mae": None,
            "close_mae": None,
            "range_mae": None,
            "direction_accuracy": None,
        }

    return {
        "rows": int(len(df)),
        "open_mae": float(df["open_abs_error"].mean()),
        "high_mae": float(df["high_abs_error"].mean()),
        "low_mae": float(df["low_abs_error"].mean()),
        "close_mae": float(df["close_abs_error"].mean()),
        "range_mae": float(df["range_abs_error"].mean()),
        "direction_accuracy": float(df["direction_correct"].astype(float).mean()),
    }


def _source_counts(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    out = {}
    for col in ["gap_source", "high_source", "low_source", "close_source"]:
        counts = df[col].fillna("unknown").value_counts().to_dict()
        out[col] = {str(k): int(v) for k, v in counts.items()}
    return out


def build_scorecard(history: pd.DataFrame) -> Dict[str, Any]:
    return {
        "all_history": _summary(history),
        "rolling_5": _summary(history.tail(5)),
        "rolling_20": _summary(history.tail(20)),
        "source_counts": _source_counts(history),
        "latest_forecast_date": None if history.empty else str(history["forecast_for_date"].iloc[-1].date()),
    }


def save_scorecard(scorecard: dict, output_file: str | Path) -> None:
    p = Path(output_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
