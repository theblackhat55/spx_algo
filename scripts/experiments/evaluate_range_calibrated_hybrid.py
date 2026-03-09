#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


DEFAULT_PREDS = Path("output/backtests/gap_augmented_hybrid/predictions.csv")
DEFAULT_OUTDIR = Path("output/reports")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate post-model range calibration for Databento gap-hybrid walk-forward predictions."
    )
    p.add_argument(
        "--predictions",
        default=str(DEFAULT_PREDS),
        help=f"Hybrid predictions CSV (default: {DEFAULT_PREDS})",
    )
    p.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help=f"Output directory (default: {DEFAULT_OUTDIR})",
    )
    p.add_argument(
        "--test-size",
        type=int,
        default=120,
        help="Holdout tail size for evaluation.",
    )
    p.add_argument(
        "--min-scale",
        type=float,
        default=0.5,
        help="Minimum calibrated range multiplier clip.",
    )
    p.add_argument(
        "--max-scale",
        type=float,
        default=2.0,
        help="Maximum calibrated range multiplier clip.",
    )
    return p.parse_args()


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {path}")

    df = pd.read_csv(path)
    required = {
        "forecast_date", "prev_close",
        "pred_open", "pred_high", "pred_low", "pred_close",
        "actual_open", "actual_high", "actual_low", "actual_close",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {sorted(missing)}")

    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    return df.sort_values("forecast_date").reset_index(drop=True)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["pred_range"] = out["pred_high"] - out["pred_low"]
    out["actual_range"] = out["actual_high"] - out["actual_low"]

    out["pred_up_ext"] = out["pred_high"] - out["pred_open"]
    out["pred_down_ext"] = out["pred_open"] - out["pred_low"]

    out["pred_up_share"] = np.where(
        out["pred_range"] > 0,
        out["pred_up_ext"] / out["pred_range"],
        0.5,
    )
    out["pred_down_share"] = np.where(
        out["pred_range"] > 0,
        out["pred_down_ext"] / out["pred_range"],
        0.5,
    )

    out["pred_gap_abs"] = np.abs(out["pred_open"] / out["prev_close"] - 1.0)
    out["pred_close_from_open_abs"] = np.abs(out["pred_close"] / out["pred_open"] - 1.0)
    out["pred_open_from_prev_close"] = out["pred_open"] / out["prev_close"] - 1.0
    out["pred_close_from_open"] = out["pred_close"] / out["pred_open"] - 1.0

    out["actual_gap_abs"] = np.abs(out["actual_open"] / out["prev_close"] - 1.0)
    out["actual_close_from_open_abs"] = np.abs(out["actual_close"] / out["actual_open"] - 1.0)

    out["range_scale_target"] = np.where(
        out["pred_range"] > 1e-9,
        out["actual_range"] / out["pred_range"],
        1.0,
    )

    out["weekday"] = out["forecast_date"].dt.weekday
    out["month"] = out["forecast_date"].dt.month

    for col in ["pred_range", "actual_range", "pred_gap_abs", "pred_close_from_open_abs"]:
        out[f"{col}_lag1"] = out[col].shift(1)
        out[f"{col}_lag2"] = out[col].shift(2)
        out[f"{col}_ma5"] = out[col].rolling(5).mean().shift(1)

    return out


def _train_test_split(df: pd.DataFrame, test_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.dropna().reset_index(drop=True)
    if len(df) <= test_size + 50:
        raise SystemExit(f"Not enough rows after feature prep for test_size={test_size}. Rows={len(df)}")
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    return train, test


def _fit_range_model(train: pd.DataFrame) -> tuple[LGBMRegressor, list[str]]:
    feature_cols = [
        "pred_range",
        "pred_up_share",
        "pred_down_share",
        "pred_gap_abs",
        "pred_close_from_open_abs",
        "pred_open_from_prev_close",
        "pred_close_from_open",
        "weekday",
        "month",
        "pred_range_lag1",
        "pred_range_lag2",
        "pred_range_ma5",
        "actual_range_lag1",
        "actual_range_lag2",
        "actual_range_ma5",
        "pred_gap_abs_lag1",
        "pred_gap_abs_lag2",
        "pred_gap_abs_ma5",
        "pred_close_from_open_abs_lag1",
        "pred_close_from_open_abs_lag2",
        "pred_close_from_open_abs_ma5",
    ]
    feature_cols = [c for c in feature_cols if c in train.columns]

    model = LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(train[feature_cols], train["range_scale_target"])
    return model, feature_cols


def _apply_calibration(
    df: pd.DataFrame,
    model: LGBMRegressor,
    feature_cols: list[str],
    min_scale: float,
    max_scale: float,
) -> pd.DataFrame:
    out = df.copy()

    raw_scale = model.predict(out[feature_cols])
    out["pred_range_scale_raw"] = raw_scale
    out["pred_range_scale"] = np.clip(raw_scale, min_scale, max_scale)

    out["cal_pred_range"] = out["pred_range"] * out["pred_range_scale"]

    # Preserve original asymmetry around predicted open
    out["cal_pred_high"] = out["pred_open"] + out["cal_pred_range"] * out["pred_up_share"]
    out["cal_pred_low"] = out["pred_open"] - out["cal_pred_range"] * out["pred_down_share"]

    # Close/open unchanged in this calibration pass
    out["cal_pred_open"] = out["pred_open"]
    out["cal_pred_close"] = out["pred_close"]

    return out


def _metric_bundle(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}

    field_map = {
        "open": f"{prefix}_open",
        "high": f"{prefix}_high",
        "low": f"{prefix}_low",
        "close": f"{prefix}_close",
    }

    for field, pred_col in field_map.items():
        err = df[pred_col] - df[f"actual_{field}"]
        out[f"{field}_mae"] = float(np.abs(err).mean())
        out[f"{field}_rmse"] = float(np.sqrt(np.mean(np.square(err))))
        out[f"{field}_bias"] = float(err.mean())

    close_dir_actual = np.sign(df["actual_close"] - df["prev_close"])
    close_dir_pred = np.sign(df[field_map["close"]] - df["prev_close"])
    out["close_direction_accuracy"] = float((close_dir_actual == close_dir_pred).mean())

    gap_actual = np.sign(df["actual_open"] / df["prev_close"] - 1.0)
    gap_pred = np.sign(df[field_map["open"]] / df["prev_close"] - 1.0)
    out["gap_sign_accuracy"] = float((gap_actual == gap_pred).mean())

    actual_range = df["actual_high"] - df["actual_low"]
    pred_range = df[field_map["high"]] - df[field_map["low"]]
    out["range_mae"] = float(np.abs(pred_range - actual_range).mean())
    out["actual_range_mean"] = float(actual_range.mean())
    out["pred_range_mean"] = float(pred_range.mean())

    # Simple coverage metrics
    out["high_coverage"] = float((df["actual_high"] <= df[field_map["high"]]).mean())
    out["low_coverage"] = float((df["actual_low"] >= df[field_map["low"]]).mean())
    out["inside_range_coverage"] = float(
        ((df["actual_high"] <= df[field_map["high"]]) & (df["actual_low"] >= df[field_map["low"]])).mean()
    )

    return out


def _monthly_compare(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["month"] = tmp["forecast_date"].dt.to_period("M").astype(str)

    rows = []
    for month, g in tmp.groupby("month"):
        row = {"month": month, "rows": int(len(g))}

        orig = _metric_bundle(g, "pred")
        cal = _metric_bundle(g, "cal_pred")

        for k, v in orig.items():
            row[f"orig_{k}"] = v
        for k, v in cal.items():
            row[f"cal_{k}"] = v
        for k in orig.keys():
            row[f"diff_{k}_cal_minus_orig"] = cal[k] - orig[k]

        rows.append(row)

    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _feature_importance(model: LGBMRegressor, feature_cols: list[str]) -> list[dict[str, Any]]:
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return imp.to_dict(orient="records")


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    preds = _load_predictions(Path(args.predictions))
    feat_df = _build_features(preds)

    train_df, test_df = _train_test_split(feat_df, args.test_size)
    model, feature_cols = _fit_range_model(train_df)
    calibrated_test = _apply_calibration(
        test_df,
        model=model,
        feature_cols=feature_cols,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
    )

    orig_metrics = _metric_bundle(calibrated_test, "pred")
    cal_metrics = _metric_bundle(calibrated_test, "cal_pred")
    diff = {k: cal_metrics[k] - orig_metrics[k] for k in orig_metrics.keys()}

    monthly = _monthly_compare(calibrated_test)

    pred_cols = [
        "forecast_date", "prev_close",
        "pred_open", "pred_high", "pred_low", "pred_close",
        "cal_pred_open", "cal_pred_high", "cal_pred_low", "cal_pred_close",
        "actual_open", "actual_high", "actual_low", "actual_close",
        "pred_range", "actual_range", "pred_range_scale_raw", "pred_range_scale", "cal_pred_range",
    ]
    pred_cols = [c for c in pred_cols if c in calibrated_test.columns]
    pred_out = calibrated_test[pred_cols].copy()

    report = {
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "test_date_min": calibrated_test["forecast_date"].min().strftime("%Y-%m-%d"),
        "test_date_max": calibrated_test["forecast_date"].max().strftime("%Y-%m-%d"),
        "min_scale": args.min_scale,
        "max_scale": args.max_scale,
        "feature_count": len(feature_cols),
        "feature_importance_top20": _feature_importance(model, feature_cols)[:20],
        "original_metrics": orig_metrics,
        "calibrated_metrics": cal_metrics,
        "metric_diff_calibrated_minus_original": diff,
    }

    summary_path = outdir / "range_calibrated_hybrid_summary.json"
    monthly_path = outdir / "range_calibrated_hybrid_monthly.csv"
    preds_path = outdir / "range_calibrated_hybrid_predictions.csv"

    summary_path.write_text(json.dumps(report, indent=2))
    monthly.to_csv(monthly_path, index=False)
    pred_out.to_csv(preds_path, index=False)

    print("Saved summary:", summary_path)
    print("Saved monthly:", monthly_path)
    print("Saved predictions:", preds_path)
    print()
    print("=== RANGE CALIBRATED HYBRID CHECK ===")
    print("rows_train:", report["rows_train"])
    print("rows_test:", report["rows_test"])
    print("test_range:", report["test_date_min"], "->", report["test_date_max"])
    print()
    print("Original metrics:")
    for k, v in report["original_metrics"].items():
        print(" ", k, v)
    print()
    print("Calibrated metrics:")
    for k, v in report["calibrated_metrics"].items():
        print(" ", k, v)
    print()
    print("Diff (calibrated - original):")
    for k, v in report["metric_diff_calibrated_minus_original"].items():
        print(" ", k, v)


if __name__ == "__main__":
    main()
