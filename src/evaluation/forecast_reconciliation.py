from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json

import pandas as pd


def load_forecast(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Forecast file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_spx_actuals(path: str | Path = "data/raw/spx_daily.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def reconcile_forecast_to_actual(forecast: dict, actual_row: pd.Series) -> Dict[str, Any]:
    pred = forecast["predicted_ohlc"]
    prev_close = float(forecast["prev_close"])

    actual_open = float(actual_row["Open"])
    actual_high = float(actual_row["High"])
    actual_low = float(actual_row["Low"])
    actual_close = float(actual_row["Close"])

    pred_open = float(pred["open"])
    pred_high = float(pred["high"])
    pred_low = float(pred["low"])
    pred_close = float(pred["close"])

    actual_range = actual_high - actual_low
    pred_range = pred_high - pred_low

    actual_dir = 0
    pred_dir = 0
    if actual_close > prev_close:
        actual_dir = 1
    elif actual_close < prev_close:
        actual_dir = -1

    if pred_close > prev_close:
        pred_dir = 1
    elif pred_close < prev_close:
        pred_dir = -1

    return {
        "forecast_for_date": forecast["forecast_for_date"],
        "generated_from_feature_date": forecast["generated_from_feature_date"],
        "prev_close": prev_close,
        "component_source_selection": forecast.get("component_source_selection", {}),
        "predicted_ohlc": pred,
        "actual_ohlc": {
            "open": actual_open,
            "high": actual_high,
            "low": actual_low,
            "close": actual_close,
        },
        "errors": {
            "open_error": actual_open - pred_open,
            "high_error": actual_high - pred_high,
            "low_error": actual_low - pred_low,
            "close_error": actual_close - pred_close,
            "open_abs_error": abs(actual_open - pred_open),
            "high_abs_error": abs(actual_high - pred_high),
            "low_abs_error": abs(actual_low - pred_low),
            "close_abs_error": abs(actual_close - pred_close),
            "range_error": actual_range - pred_range,
            "range_abs_error": abs(actual_range - pred_range),
        },
        "direction": {
            "actual_close_vs_prev_close": actual_dir,
            "pred_close_vs_prev_close": pred_dir,
            "direction_correct": actual_dir == pred_dir,
        },
    }


def append_reconciliation_history(
    reconciliation: dict,
    history_csv: str | Path,
) -> pd.DataFrame:
    row = {
        "forecast_for_date": reconciliation["forecast_for_date"],
        "generated_from_feature_date": reconciliation["generated_from_feature_date"],
        "prev_close": reconciliation["prev_close"],
        "pred_open": reconciliation["predicted_ohlc"]["open"],
        "pred_high": reconciliation["predicted_ohlc"]["high"],
        "pred_low": reconciliation["predicted_ohlc"]["low"],
        "pred_close": reconciliation["predicted_ohlc"]["close"],
        "actual_open": reconciliation["actual_ohlc"]["open"],
        "actual_high": reconciliation["actual_ohlc"]["high"],
        "actual_low": reconciliation["actual_ohlc"]["low"],
        "actual_close": reconciliation["actual_ohlc"]["close"],
        "open_abs_error": reconciliation["errors"]["open_abs_error"],
        "high_abs_error": reconciliation["errors"]["high_abs_error"],
        "low_abs_error": reconciliation["errors"]["low_abs_error"],
        "close_abs_error": reconciliation["errors"]["close_abs_error"],
        "range_abs_error": reconciliation["errors"]["range_abs_error"],
        "direction_correct": reconciliation["direction"]["direction_correct"],
        "gap_source": reconciliation["component_source_selection"].get("target_gap_ret"),
        "high_source": reconciliation["component_source_selection"].get("target_high_from_open"),
        "low_source": reconciliation["component_source_selection"].get("target_low_from_open"),
        "close_source": reconciliation["component_source_selection"].get("target_close_from_open"),
    }

    path = Path(history_csv)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        hist = pd.read_csv(path)
        hist = hist[hist["forecast_for_date"] != row["forecast_for_date"]]
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])

    hist = hist.sort_values("forecast_for_date").reset_index(drop=True)
    hist.to_csv(path, index=False)
    return hist


def save_reconciliation_json(reconciliation: dict, output_file: str | Path) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reconciliation, f, indent=2)
