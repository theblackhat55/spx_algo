from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json

import pandas as pd

from src.features.builder import load_feature_matrix
from src.models.ohlc_forecaster import (
    load_models,
    predict_ohlc_components,
    reconstruct_ohlc,
)

MODEL_DIR = Path("output/models/ohlc")
FORECAST_FILE = Path("output/forecasts/latest_ohlc_forecast.json")


def _load_spx_raw() -> pd.DataFrame:
    spx = pd.read_parquet("data/raw/spx_daily.parquet")
    if "Date" in spx.columns:
        spx["Date"] = pd.to_datetime(spx["Date"])
        spx = spx.set_index("Date")
    spx.index = pd.to_datetime(spx.index)
    spx = spx.sort_index()
    return spx


def _infer_next_trading_day(index: pd.DatetimeIndex) -> str:
    if len(index) == 0:
        raise ValueError("Cannot infer next trading day from empty index")
    last_date = pd.Timestamp(index[-1])

    candidate = last_date + pd.Timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += pd.Timedelta(days=1)
    return candidate.strftime("%Y-%m-%d")


def generate_latest_ohlc_forecast() -> Dict[str, Any]:
    features = load_feature_matrix()
    if features is None or features.empty:
        raise RuntimeError("Feature matrix is empty or missing")

    spx = _load_spx_raw()
    models = load_models(MODEL_DIR)

    latest_idx = features.index[-1]
    latest_features = features.tail(1)

    if latest_idx not in spx.index:
        raise RuntimeError(f"Latest feature date {latest_idx} not found in SPX raw data")

    prev_close = spx.loc[[latest_idx], "Close"]
    component_preds = predict_ohlc_components(models, latest_features)
    pred_ohlc = reconstruct_ohlc(prev_close=prev_close, component_preds=component_preds)

    next_trading_day = _infer_next_trading_day(features.index)

    row = pred_ohlc.iloc[0]
    comp = component_preds.iloc[0]

    forecast = {
        "forecast_for_date": next_trading_day,
        "generated_from_feature_date": str(pd.Timestamp(latest_idx).date()),
        "prev_close": float(prev_close.iloc[0]),
        "predicted_components": {
            "target_gap_ret": float(comp["target_gap_ret"]),
            "target_high_from_open": float(comp["target_high_from_open"]),
            "target_low_from_open": float(comp["target_low_from_open"]),
            "target_close_from_open": float(comp["target_close_from_open"]),
        },
        "predicted_ohlc": {
            "open": float(row["pred_open"]),
            "high": float(row["pred_high"]),
            "low": float(row["pred_low"]),
            "close": float(row["pred_close"]),
        },
        "model_artifacts": {
            "model_dir": str(MODEL_DIR),
            "forecast_file": str(FORECAST_FILE),
        },
    }
    return forecast


def save_latest_ohlc_forecast(output_file: Path = FORECAST_FILE) -> Dict[str, Any]:
    forecast = generate_latest_ohlc_forecast()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(forecast, f, indent=2)
    return forecast


if __name__ == "__main__":
    forecast = save_latest_ohlc_forecast()
    print("Saved forecast to:", FORECAST_FILE)
    print(json.dumps(forecast, indent=2))
