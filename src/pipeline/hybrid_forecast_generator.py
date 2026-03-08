from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json

import pandas as pd

from src.features.builder import load_feature_matrix
from src.models.ohlc_forecaster import (
    OHLC_TARGET_COLUMNS,
    load_models,
    predict_ohlc_components,
    reconstruct_ohlc,
)
from src.targets.ohlc_targets import build_ohlc_component_targets


MODEL_DIR = Path("output/models/ohlc")
FORECAST_FILE = Path("output/forecasts/latest_hybrid_ohlc_forecast.json")

DEFAULT_COMPONENT_SELECTION = {
    "target_gap_ret": "rolling_baseline",
    "target_high_from_open": "model",
    "target_low_from_open": "model",
    "target_close_from_open": "model",
}

ROLLING_WINDOW = 20


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


def _compute_latest_rolling_baseline(targets: pd.DataFrame, latest_date: pd.Timestamp) -> pd.Series:
    hist = targets.loc[targets.index < latest_date, OHLC_TARGET_COLUMNS].tail(ROLLING_WINDOW)
    if hist.empty:
        raise RuntimeError("Not enough target history to compute rolling baseline")
    row = hist.mean()
    row.name = latest_date
    return row


def _assemble_hybrid_components(
    model_components: pd.Series,
    baseline_components: pd.Series,
    selection: Dict[str, str],
) -> pd.DataFrame:
    out = {}
    for col in OHLC_TARGET_COLUMNS:
        source = selection[col]
        if source == "model":
            out[col] = float(model_components[col])
        elif source == "rolling_baseline":
            out[col] = float(baseline_components[col])
        else:
            raise ValueError(f"Unknown component source for {col}: {source}")
    return pd.DataFrame([out], index=[model_components.name])


def generate_latest_hybrid_ohlc_forecast(
    selection: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    selection = selection or DEFAULT_COMPONENT_SELECTION

    features = load_feature_matrix()
    if features is None or features.empty:
        raise RuntimeError("Feature matrix is empty or missing")

    spx = _load_spx_raw()
    targets = build_ohlc_component_targets(spx, dropna=True)
    models = load_models(MODEL_DIR)

    latest_idx = pd.Timestamp(features.index[-1])
    latest_features = features.tail(1)

    if latest_idx not in spx.index:
        raise RuntimeError(f"Latest feature date {latest_idx} not found in SPX raw data")

    model_component_df = predict_ohlc_components(models, latest_features)
    model_component_row = model_component_df.iloc[0]
    baseline_component_row = _compute_latest_rolling_baseline(targets, latest_idx)

    hybrid_components = _assemble_hybrid_components(
        model_components=model_component_row,
        baseline_components=baseline_component_row,
        selection=selection,
    )

    prev_close = spx.loc[[latest_idx], "Close"]
    pred_ohlc = reconstruct_ohlc(prev_close=prev_close, component_preds=hybrid_components)

    next_trading_day = _infer_next_trading_day(features.index)
    row = pred_ohlc.iloc[0]
    comp = hybrid_components.iloc[0]

    forecast = {
        "forecast_for_date": next_trading_day,
        "generated_from_feature_date": str(latest_idx.date()),
        "prev_close": float(prev_close.iloc[0]),
        "component_source_selection": selection,
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
            "rolling_window": ROLLING_WINDOW,
        },
    }
    return forecast


def save_latest_hybrid_ohlc_forecast(output_file: Path = FORECAST_FILE) -> Dict[str, Any]:
    forecast = generate_latest_hybrid_ohlc_forecast()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(forecast, f, indent=2)
    return forecast


if __name__ == "__main__":
    forecast = save_latest_hybrid_ohlc_forecast()
    print("Saved hybrid forecast to:", FORECAST_FILE)
    print(json.dumps(forecast, indent=2))
