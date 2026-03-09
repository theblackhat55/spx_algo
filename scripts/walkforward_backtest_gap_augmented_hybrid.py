#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAW_SPX_PATH = Path("data/raw/spx_daily.parquet")
BASE_FEATURES_PATH = Path("data/processed/features.parquet")
DB_FEATURES_PATH = Path("data/processed/es_databento_overnight_features.parquet")

DEFAULT_OUTDIR = Path("output/backtests/gap_augmented_hybrid")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward backtest for gap-augmented hybrid OHLC forecast."
    )
    p.add_argument("--start-date", required=True, help="First forecast date, YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="Last forecast date, YYYY-MM-DD")
    p.add_argument(
        "--mode",
        choices=["frozen", "rolling"],
        default="rolling",
        help="Frozen = fit once before test window. Rolling = periodic refits during test window.",
    )
    p.add_argument(
        "--train-window-years",
        type=float,
        default=5.0,
        help="Approximate rolling training window length in years.",
    )
    p.add_argument(
        "--retrain-gap-every",
        type=int,
        default=5,
        help="Retrain Databento gap model every N forecast dates in rolling mode.",
    )
    p.add_argument(
        "--retrain-ohlc-every",
        type=int,
        default=20,
        help="Retrain base OHLC models every N forecast dates in rolling mode.",
    )
    p.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help=f"Output directory (default: {DEFAULT_OUTDIR})",
    )
    return p.parse_args()


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not RAW_SPX_PATH.exists():
        raise FileNotFoundError(f"Missing raw SPX file: {RAW_SPX_PATH}")
    if not BASE_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing base features file: {BASE_FEATURES_PATH}")
    if not DB_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing Databento features file: {DB_FEATURES_PATH}")

    spx = pd.read_parquet(RAW_SPX_PATH).copy()
    base = pd.read_parquet(BASE_FEATURES_PATH).copy()
    db = pd.read_parquet(DB_FEATURES_PATH).copy()

    spx.index = pd.to_datetime(spx.index).tz_localize(None)
    base.index = pd.to_datetime(base.index).tz_localize(None)
    db.index = pd.to_datetime(db.index).tz_localize(None)

    spx = spx.sort_index()
    base = base.sort_index()
    db = db.sort_index()

    return spx, base, db


def _validate_spx_columns(spx: pd.DataFrame) -> None:
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(spx.columns)
    if missing:
        raise ValueError(f"SPX parquet missing columns: {sorted(missing)}")


def _next_index_date(index: pd.Index, d: pd.Timestamp) -> pd.Timestamp | None:
    later = index[index > d]
    if len(later) == 0:
        return None
    return pd.Timestamp(later[0])


def _prepare_targets(spx: pd.DataFrame) -> pd.DataFrame:
    df = spx[["Open", "High", "Low", "Close"]].copy()
    prev_close = df["Close"].shift(1)

    targets = pd.DataFrame(index=df.index)
    targets["prev_close"] = prev_close
    targets["target_gap_ret"] = df["Open"] / prev_close - 1.0
    targets["target_high_from_open"] = df["High"] / df["Open"] - 1.0
    targets["target_low_from_open"] = df["Low"] / df["Open"] - 1.0
    targets["target_close_from_open"] = df["Close"] / df["Open"] - 1.0

    targets["actual_open"] = df["Open"]
    targets["actual_high"] = df["High"]
    targets["actual_low"] = df["Low"]
    targets["actual_close"] = df["Close"]

    return targets


def _safe_import_models():
    from src.models.gap_forecaster_databento import train_gap_model, predict_gap
    from src.models.ohlc_forecaster import (
        train_ohlc_models,
        predict_ohlc_components,
        reconstruct_ohlc,
    )
    return train_gap_model, predict_gap, train_ohlc_models, predict_ohlc_components, reconstruct_ohlc


def _build_gap_feature_matrix(base_features: pd.DataFrame, db_features: pd.DataFrame) -> pd.DataFrame:
    return base_features.join(db_features, how="inner")


def _subset_train_window(df: pd.DataFrame, end_date: pd.Timestamp, years: float) -> pd.DataFrame:
    approx_days = max(int(round(years * 252)), 60)
    eligible = df.loc[df.index <= end_date]
    if len(eligible) <= approx_days:
        return eligible
    return eligible.iloc[-approx_days:]


def _fit_gap_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    train_gap_model, _, _, _, _ = _safe_import_models()
    return train_gap_model(X_train, y_train)


def _fit_ohlc_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
):
    _, _, train_ohlc_models, _, _ = _safe_import_models()
    return train_ohlc_models(X_train, y_train, model_params=None)


def _predict_one(
    feature_date: pd.Timestamp,
    forecast_date: pd.Timestamp,
    prev_close: float,
    X_gap_row: pd.DataFrame,
    X_base_row: pd.DataFrame,
    gap_model: Any,
    ohlc_models: Any,
) -> dict[str, float]:
    _, predict_gap, _, predict_ohlc_components, reconstruct_ohlc = _safe_import_models()

    gap_raw = predict_gap(gap_model, X_gap_row)
    gap_pred = float(gap_raw.iloc[0] if hasattr(gap_raw, "iloc") else gap_raw[0])

    comp_pred = predict_ohlc_components(ohlc_models, X_base_row)
    if isinstance(comp_pred, pd.DataFrame):
        comp_row = comp_pred.iloc[0].to_dict()
    elif isinstance(comp_pred, dict):
        comp_row = {k: float(v[0] if hasattr(v, "__len__") else v) for k, v in comp_pred.items()}
    else:
        raise TypeError(f"Unexpected predict_ohlc_components return type: {type(comp_pred)}")

    pred_components = {
        "target_gap_ret": gap_pred,
        "target_high_from_open": float(comp_row["target_high_from_open"]),
        "target_low_from_open": float(comp_row["target_low_from_open"]),
        "target_close_from_open": float(comp_row["target_close_from_open"]),
    }

    prev_close_series = pd.Series([prev_close], index=X_base_row.index, name="prev_close")
    component_preds_df = pd.DataFrame([pred_components], index=X_base_row.index)
    reconstructed = reconstruct_ohlc(prev_close_series, component_preds_df)
    if isinstance(reconstructed, pd.DataFrame):
        reconstructed = reconstructed.iloc[0].to_dict()

    return {
        "feature_date": feature_date.strftime("%Y-%m-%d"),
        "forecast_date": forecast_date.strftime("%Y-%m-%d"),
        "prev_close": float(prev_close),
        "pred_gap_ret": float(pred_components["target_gap_ret"]),
        "pred_high_from_open": float(pred_components["target_high_from_open"]),
        "pred_low_from_open": float(pred_components["target_low_from_open"]),
        "pred_close_from_open": float(pred_components["target_close_from_open"]),
        "pred_open": float(reconstructed["pred_open"]),
        "pred_high": float(reconstructed["pred_high"]),
        "pred_low": float(reconstructed["pred_low"]),
        "pred_close": float(reconstructed["pred_close"]),
    }


def _compute_summary(preds: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": int(len(preds)),
        "date_min": None if preds.empty else preds["forecast_date"].min(),
        "date_max": None if preds.empty else preds["forecast_date"].max(),
    }
    if preds.empty:
        return out

    for field in ["open", "high", "low", "close"]:
        err = preds[f"pred_{field}"] - preds[f"actual_{field}"]
        out[f"{field}_mae"] = float(np.abs(err).mean())
        out[f"{field}_rmse"] = float(np.sqrt(np.mean(np.square(err))))
        out[f"{field}_bias"] = float(err.mean())

    close_dir_actual = np.sign(preds["actual_close"] - preds["prev_close"])
    close_dir_pred = np.sign(preds["pred_close"] - preds["prev_close"])
    out["close_direction_accuracy"] = float((close_dir_actual == close_dir_pred).mean())

    actual_range = preds["actual_high"] - preds["actual_low"]
    pred_range = preds["pred_high"] - preds["pred_low"]
    out["range_mae"] = float(np.abs(pred_range - actual_range).mean())
    out["actual_range_mean"] = float(actual_range.mean())
    out["pred_range_mean"] = float(pred_range.mean())

    gap_actual = np.sign(preds["actual_open"] / preds["prev_close"] - 1.0)
    gap_pred = np.sign(preds["pred_open"] / preds["prev_close"] - 1.0)
    out["gap_sign_accuracy"] = float((gap_actual == gap_pred).mean())

    return out


def _compute_monthly(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()

    df = preds.copy()
    df["month"] = pd.to_datetime(df["forecast_date"]).dt.to_period("M").astype(str)

    rows = []
    for month, g in df.groupby("month"):
        row = {"month": month, "rows": int(len(g))}
        for field in ["open", "high", "low", "close"]:
            err = g[f"pred_{field}"] - g[f"actual_{field}"]
            row[f"{field}_mae"] = float(np.abs(err).mean())
            row[f"{field}_bias"] = float(err.mean())
        actual_range = g["actual_high"] - g["actual_low"]
        pred_range = g["pred_high"] - g["pred_low"]
        row["range_mae"] = float(np.abs(pred_range - actual_range).mean())
        row["close_direction_accuracy"] = float(
            (np.sign(g["actual_close"] - g["prev_close"]) == np.sign(g["pred_close"] - g["prev_close"])).mean()
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    spx, base_features, db_features = _load_inputs()
    _validate_spx_columns(spx)
    targets = _prepare_targets(spx)

    start_date = pd.Timestamp(args.start_date).normalize()
    end_date = pd.Timestamp(args.end_date).normalize()
    if end_date < start_date:
        raise SystemExit("--end-date must be >= --start-date")

    common_feature_dates = base_features.index.intersection(db_features.index).intersection(targets.index)
    common_feature_dates = common_feature_dates.sort_values()

    eligible_rows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for feature_date in common_feature_dates:
        forecast_date = _next_index_date(spx.index, feature_date)
        if forecast_date is None:
            continue
        if start_date <= forecast_date <= end_date:
            eligible_rows.append((feature_date, forecast_date))

    if not eligible_rows:
        raise SystemExit("No eligible feature/forecast date pairs found in requested range.")

    gap_features_all = _build_gap_feature_matrix(base_features, db_features)

    gap_model = None
    ohlc_models = None
    last_gap_fit_idx = None
    last_ohlc_fit_idx = None

    results: list[dict[str, Any]] = []

    for i, (feature_date, forecast_date) in enumerate(eligible_rows):
        train_cutoff = feature_date

        train_targets = targets.loc[targets.index <= train_cutoff].copy()
        train_targets = train_targets.dropna(subset=[
            "prev_close",
            "target_gap_ret",
            "target_high_from_open",
            "target_low_from_open",
            "target_close_from_open",
        ])

        gap_train_df = gap_features_all.join(
            train_targets[["target_gap_ret"]],
            how="inner"
        ).dropna()

        base_train_df = base_features.join(
            train_targets[[
                "target_gap_ret",
                "target_high_from_open",
                "target_low_from_open",
                "target_close_from_open",
            ]],
            how="inner"
        ).dropna()

        gap_train_df = _subset_train_window(gap_train_df, train_cutoff, min(args.train_window_years, 2.0))
        base_train_df = _subset_train_window(base_train_df, train_cutoff, args.train_window_years)

        need_gap_fit = False
        need_ohlc_fit = False

        if args.mode == "frozen":
            if gap_model is None:
                need_gap_fit = True
            if ohlc_models is None:
                need_ohlc_fit = True
        else:
            if gap_model is None or last_gap_fit_idx is None or (i - last_gap_fit_idx) >= args.retrain_gap_every:
                need_gap_fit = True
            if ohlc_models is None or last_ohlc_fit_idx is None or (i - last_ohlc_fit_idx) >= args.retrain_ohlc_every:
                need_ohlc_fit = True

        if need_gap_fit:
            X_gap_train = gap_train_df.drop(columns=["target_gap_ret"])
            y_gap_train = gap_train_df["target_gap_ret"]
            gap_model = _fit_gap_model(X_gap_train, y_gap_train)
            last_gap_fit_idx = i

        if need_ohlc_fit:
            X_base_train = base_train_df.drop(columns=[
                "target_gap_ret",
                "target_high_from_open",
                "target_low_from_open",
                "target_close_from_open",
            ])
            y_base_train = base_train_df[[
                "target_gap_ret",
                "target_high_from_open",
                "target_low_from_open",
                "target_close_from_open",
            ]]
            ohlc_models = _fit_ohlc_models(X_base_train, y_base_train)
            last_ohlc_fit_idx = i

        if feature_date not in gap_features_all.index or feature_date not in base_features.index:
            continue
        if forecast_date not in targets.index:
            continue

        X_gap_row = gap_features_all.loc[[feature_date]].copy()
        X_base_row = base_features.loc[[feature_date]].copy()

        prev_close = float(targets.loc[forecast_date, "prev_close"])
        pred = _predict_one(
            feature_date=feature_date,
            forecast_date=forecast_date,
            prev_close=prev_close,
            X_gap_row=X_gap_row,
            X_base_row=X_base_row,
            gap_model=gap_model,
            ohlc_models=ohlc_models,
        )

        actual = targets.loc[forecast_date]
        pred["actual_open"] = float(actual["actual_open"])
        pred["actual_high"] = float(actual["actual_high"])
        pred["actual_low"] = float(actual["actual_low"])
        pred["actual_close"] = float(actual["actual_close"])

        results.append(pred)
        print(
            f"[{i+1}/{len(eligible_rows)}] "
            f"feature={feature_date.date()} -> forecast={forecast_date.date()} | "
            f"pred_open={pred['pred_open']:.2f} actual_open={pred['actual_open']:.2f}"
        )

    preds = pd.DataFrame(results)
    if preds.empty:
        raise SystemExit("Backtest ran but produced no predictions.")

    preds = preds.sort_values("forecast_date").reset_index(drop=True)

    summary = _compute_summary(preds)
    summary["mode"] = args.mode
    summary["train_window_years"] = args.train_window_years
    summary["retrain_gap_every"] = args.retrain_gap_every
    summary["retrain_ohlc_every"] = args.retrain_ohlc_every

    monthly = _compute_monthly(preds)

    preds_path = outdir / "predictions.csv"
    summary_path = outdir / "summary.json"
    monthly_path = outdir / "monthly_metrics.csv"

    preds.to_csv(preds_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))
    monthly.to_csv(monthly_path, index=False)

    print()
    print("=== WALKFORWARD BACKTEST CHECK ===")
    print("predictions_csv:", preds_path)
    print("summary_json:", summary_path)
    print("monthly_csv:", monthly_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
