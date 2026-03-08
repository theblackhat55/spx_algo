#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.features.builder import build_feature_matrix
from src.features.es_databento_overnight import load_es_databento_overnight_features
from src.models.gap_forecaster_databento import build_gap_feature_matrix, load_gap_model, predict_gap
from src.models.ohlc_forecaster import load_models, predict_ohlc_components, reconstruct_ohlc


RAW_SPX_PATH = Path("data/raw/spx_daily.parquet")
OHLC_MODEL_DIR = Path("output/models/ohlc")
GAP_MODEL_PATH = Path("output/models/ohlc/gap_databento_model.joblib")
DEFAULT_OUT_DIR = Path("output/forecasts/backtests")


def _next_business_day(ts: pd.Timestamp) -> pd.Timestamp:
    nxt = ts + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt.normalize()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate backdated gap-augmented hybrid OHLC forecast(s) from one feature date or a range.",
    )
    p.add_argument(
        "--feature-date",
        default=None,
        help="Single feature date (YYYY-MM-DD). Example: 2026-03-03 predicts the next trading day.",
    )
    p.add_argument(
        "--start-date",
        default=None,
        help="Start feature date for range mode (YYYY-MM-DD).",
    )
    p.add_argument(
        "--end-date",
        default=None,
        help="End feature date for range mode (YYYY-MM-DD).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional explicit output JSON path for single-date mode only.",
    )
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    single = args.feature_date is not None
    ranged = args.start_date is not None or args.end_date is not None

    if single and ranged:
        raise SystemExit("Use either --feature-date OR (--start-date and --end-date), not both.")

    if single:
        return

    if args.start_date and args.end_date:
        return

    raise SystemExit(
        "You must provide either --feature-date YYYY-MM-DD or both --start-date YYYY-MM-DD --end-date YYYY-MM-DD"
    )


def _load_inputs():
    raw_spx = pd.read_parquet(RAW_SPX_PATH).copy()
    raw_spx.index = pd.to_datetime(raw_spx.index)

    base_features = build_feature_matrix()
    base_features.index = pd.to_datetime(base_features.index)

    db_feat = load_es_databento_overnight_features()
    db_feat.index = pd.to_datetime(db_feat.index)

    component_models = load_models(OHLC_MODEL_DIR)
    gap_model = load_gap_model(GAP_MODEL_PATH)

    return raw_spx, base_features, db_feat, component_models, gap_model


def _generate_one(
    feature_date: pd.Timestamp,
    raw_spx: pd.DataFrame,
    base_features: pd.DataFrame,
    db_feat: pd.DataFrame,
    component_models,
    gap_model,
) -> dict:
    if feature_date not in base_features.index:
        raise RuntimeError(f"Feature date {feature_date.date()} not found in base feature matrix")
    if feature_date not in db_feat.index:
        raise RuntimeError(f"Feature date {feature_date.date()} not found in Databento overnight features")
    if feature_date not in raw_spx.index:
        raise RuntimeError(f"Feature date {feature_date.date()} not found in raw SPX data")

    loc = raw_spx.index.get_loc(feature_date)
    if isinstance(loc, slice) or isinstance(loc, list):
        raise RuntimeError("Unexpected duplicate index result for feature date in raw SPX data")
    if loc == 0:
        raise RuntimeError("Cannot compute prev_close for the first SPX row")

    prev_close_value = float(raw_spx["Close"].iloc[loc - 1])

    base_row = base_features.loc[[feature_date]]
    gap_X = build_gap_feature_matrix(base_features.loc[[feature_date]], db_feat)

    component_preds = predict_ohlc_components(component_models, base_row)

    gap_pred = predict_gap(gap_model, gap_X)
    component_preds.loc[feature_date, "target_gap_ret"] = float(gap_pred.iloc[0])

    prev_close = pd.Series([prev_close_value], index=[feature_date], name="prev_close")
    pred_ohlc = reconstruct_ohlc(prev_close=prev_close, component_preds=component_preds)

    forecast_for_date = _next_business_day(feature_date)

    payload = {
        "forecast_for_date": str(forecast_for_date.date()),
        "generated_from_feature_date": str(feature_date.date()),
        "prev_close": prev_close_value,
        "component_source_selection": {
            "target_gap_ret": "databento_gap_model",
            "target_high_from_open": "model",
            "target_low_from_open": "model",
            "target_close_from_open": "model",
        },
        "predicted_components": {
            k: float(v) for k, v in component_preds.iloc[0].to_dict().items()
        },
        "predicted_ohlc": {
            k: float(v) for k, v in pred_ohlc.iloc[0].to_dict().items()
        },
        "model_artifacts": {
            "ohlc_model_dir": str(OHLC_MODEL_DIR),
            "gap_model_path": str(GAP_MODEL_PATH),
        },
    }
    return payload


def _write_single(payload: dict, out: str | None) -> Path:
    if out:
        out_path = Path(out)
    else:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_OUT_DIR / (
            f"{payload['forecast_for_date']}_from_{payload['generated_from_feature_date']}_gap_augmented_hybrid_ohlc_forecast.json"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def _date_range(start_date: str, end_date: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        raise SystemExit("--end-date must be greater than or equal to --start-date")
    return list(pd.date_range(start, end, freq="D"))


def main() -> None:
    args = parse_args()
    _validate_args(args)

    raw_spx, base_features, db_feat, component_models, gap_model = _load_inputs()

    if args.feature_date:
        feature_date = pd.Timestamp(args.feature_date)
        payload = _generate_one(
            feature_date=feature_date,
            raw_spx=raw_spx,
            base_features=base_features,
            db_feat=db_feat,
            component_models=component_models,
            gap_model=gap_model,
        )
        out_path = _write_single(payload, args.out)

        print(f"Saved backdated forecast to: {out_path}")
        print("Forecast for date:", payload["forecast_for_date"])
        print("Generated from feature date:", payload["generated_from_feature_date"])
        print("Component source selection:", payload["component_source_selection"])
        print("Predicted OHLC:", payload["predicted_ohlc"])
        return

    dates = _date_range(args.start_date, args.end_date)
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    created = []
    skipped = []

    for d in dates:
        try:
            payload = _generate_one(
                feature_date=d,
                raw_spx=raw_spx,
                base_features=base_features,
                db_feat=db_feat,
                component_models=component_models,
                gap_model=gap_model,
            )
            out_path = _write_single(payload, None)
            created.append((str(d.date()), str(out_path)))
            print(f"[OK] {d.date()} -> {payload['forecast_for_date']}  saved: {out_path.name}")
        except Exception as exc:
            skipped.append((str(d.date()), str(exc)))
            print(f"[SKIP] {d.date()} -> {exc}")

    summary = {
        "requested_dates": [str(d.date()) for d in dates],
        "created_count": len(created),
        "skipped_count": len(skipped),
        "created": created,
        "skipped": skipped,
        "out_dir": str(DEFAULT_OUT_DIR),
    }

    summary_path = DEFAULT_OUT_DIR / (
        f"summary_{pd.Timestamp(args.start_date).date()}_to_{pd.Timestamp(args.end_date).date()}.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== RANGE SUMMARY ===")
    print("created_count:", len(created))
    print("skipped_count:", len(skipped))
    print("summary_path:", summary_path)


if __name__ == "__main__":
    main()
