from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.builder import build_feature_matrix
from src.features.es_databento_overnight import load_es_databento_overnight_features
from src.models.gap_forecaster_databento import (
    build_gap_feature_matrix,
    load_gap_model,
    predict_gap,
)
from src.models.ohlc_forecaster import load_models, predict_ohlc_components, reconstruct_ohlc


DEFAULT_OHLC_MODEL_DIR = Path("output/models/ohlc")
DEFAULT_GAP_MODEL_PATH = Path("output/models/ohlc/gap_databento_model.joblib")
DEFAULT_FORECAST_PATH = Path("output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json")
RAW_SPX_PATH = Path("data/raw/spx_daily.parquet")


def _next_business_day(ts: pd.Timestamp) -> pd.Timestamp:
    nxt = ts + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt.normalize()


def generate_gap_augmented_hybrid_forecast(
    ohlc_model_dir: str | Path = DEFAULT_OHLC_MODEL_DIR,
    gap_model_path: str | Path = DEFAULT_GAP_MODEL_PATH,
) -> dict:
    base_features = build_feature_matrix()
    base_features.index = pd.to_datetime(base_features.index)

    db_feat = load_es_databento_overnight_features()
    db_feat.index = pd.to_datetime(db_feat.index)

    common_dates = base_features.index.intersection(db_feat.index)
    if common_dates.empty:
        raise RuntimeError("No overlapping dates between base features and Databento overnight features")

    latest_date = common_dates.max()
    latest_base_row = base_features.loc[[latest_date]]

    gap_X = build_gap_feature_matrix(latest_base_row, db_feat)
    if gap_X.empty:
        raise RuntimeError(
            f"No Databento gap features available for latest common date {latest_date.date()}"
        )

    component_models = load_models(ohlc_model_dir)
    component_preds = predict_ohlc_components(component_models, latest_base_row)

    gap_model = load_gap_model(gap_model_path)
    gap_pred = predict_gap(gap_model, gap_X)

    component_preds.loc[latest_date, "target_gap_ret"] = float(gap_pred.iloc[0])

    raw_spx = pd.read_parquet(RAW_SPX_PATH).copy()
    raw_spx.index = pd.to_datetime(raw_spx.index)
    if latest_date not in raw_spx.index:
        raise RuntimeError(f"Latest feature date {latest_date.date()} not found in raw SPX data")
    loc = raw_spx.index.get_loc(latest_date)
    if isinstance(loc, slice) or isinstance(loc, list):
        raise RuntimeError("Unexpected duplicate/latest-date lookup result in raw SPX data")
    if loc == 0:
        raise RuntimeError("Cannot compute prev_close for the first SPX row")
    prev_close_value = float(raw_spx["Close"].iloc[loc - 1])
    prev_close = pd.Series([prev_close_value], index=[latest_date], name="prev_close")

    pred_ohlc = reconstruct_ohlc(prev_close=prev_close, component_preds=component_preds)

    latest_spx_date = pd.to_datetime(raw_spx.index).max()
    forecast_for_date = _next_business_day(pd.Timestamp(latest_spx_date))

    return {
        "forecast_for_date": str(forecast_for_date.date()),
        "generated_from_feature_date": str(pd.Timestamp(latest_date).date()),
        "prev_close": float(prev_close.iloc[0]),
        "component_source_selection": {
            "target_gap_ret": "databento_gap_model",
            "target_high_from_open": "model",
            "target_low_from_open": "model",
            "target_close_from_open": "model",
        },
        "databento_gap_features_used": list(gap_X.columns[-4:]),
        "predicted_components": {
            k: float(v) for k, v in component_preds.iloc[0].to_dict().items()
        },
        "predicted_ohlc": {
            k: float(v) for k, v in pred_ohlc.iloc[0].to_dict().items()
        },
        "model_artifacts": {
            "ohlc_model_dir": str(Path(ohlc_model_dir)),
            "gap_model_path": str(Path(gap_model_path)),
        },
    }


def save_gap_augmented_hybrid_forecast(
    out_path: str | Path = DEFAULT_FORECAST_PATH,
    ohlc_model_dir: str | Path = DEFAULT_OHLC_MODEL_DIR,
    gap_model_path: str | Path = DEFAULT_GAP_MODEL_PATH,
) -> dict:
    forecast = generate_gap_augmented_hybrid_forecast(
        ohlc_model_dir=ohlc_model_dir,
        gap_model_path=gap_model_path,
    )
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(forecast, indent=2))
    return forecast
