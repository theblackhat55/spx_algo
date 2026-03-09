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
        description="Evaluate two-sided post-model range calibration for Databento gap-hybrid predictions."
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
        "--min-up-scale",
        type=float,
        default=0.5,
        help="Minimum clip for upside extension multiplier.",
    )
    p.add_argument(
        "--max-up-scale",
        type=float,
        default=2.0,
        help="Maximum clip for upside extension multiplier.",
    )
    p.add_argument(
        "--min-down-scale",
        type=float,
        default=0.5,
        help="Minimum clip for downside extension multiplier.",
    )
    p.add_argument(
        "--max-down-scale",
        type=float,
        default=2.5,
        help="Maximum clip for downside extension multiplier.",
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

    out["pred_up_ext"] = np.maximum(out["pred_high"] - out["pred_open"], 0.0)
    out["pred_down_ext"] = np.maximum(out["pred_open"] - out["pred_low"], 0.0)

    out["actual_up_ext"] = np.maximum(out["actual_high"] - out["pred_open"], 0.0)
    out["actual_down_ext"] = np.maximum(out["pred_open"] - out["actual_low"], 0.0)

    out["up_scale_target"] = np.where(out["pred_up_ext"] > 1e-9, out["actual_up_ext"] / out["pred_up_ext"], 1.0)
    out["down_scale_target"] = np.where(out["pred_down_ext"] > 1e-9, out["actual_down_ext"] / out["pred_down_ext"], 1.0)

    out["pred_gap_abs"] = np.abs(out["pred_open"] / out["prev_close"] - 1.0)
    out["pred_close_from_open"] = out["pred_close"] / out["pred_open"] - 1.0
    out["pred_close_from_open_abs"] = np.abs(out["pred_close_from_open"])

    out["pred_up_share"] = np.where(
        out["pred_range"] > 1e-9,
        out["pred_up_ext"] / out["pred_range"],
        0.5,
    )
    out["pred_down_share"] = np.where(
        out["pred_range"] > 1e-9,
        out["pred_down_ext"] / out["pred_range"],
        0.5,
    )

    out["weekday"] = out["forecast_date"].dt.weekday
    out["month"] = out["forecast_date"].dt.month

    lag_cols = [
        "pred_range",
        "actual_range",
        "pred_gap_abs",
        "pred_close_from_open_abs",
        "pred_up_ext",
        "pred_down_ext",
        "actual_up_ext",
        "actual_down_ext",
    ]
    for col in lag_cols:
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


def _model_params() -> dict[str, Any]:
    return {
        "objective": "regression",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }


def _fit_model(train: pd.DataFrame, feature_cols: list[str], target_col: str) -> LGBMRegressor:
    model = LGBMRegressor(**_model_params())
    model.fit(train[feature_cols], train[target_col])
    return model


def _apply_two_sided_calibration(
    df: pd.DataFrame,
    up_model: LGBMRegressor,
    down_model: LGBMRegressor,
    feature_cols: list[str],
    min_up_scale: float,
    max_up_scale: float,
    min_down_scale: float,
    max_down_scale: float,
) -> pd.DataFrame:
    out = df.copy()

    up_raw = up_model.predict(out[feature_cols])
    down_raw = down_model.predict(out[feature_cols])

    out["up_scale_raw"] = up_raw
    out["down_scale_raw"] = down_raw

    out["up_scale"] = np.clip(out["up_scale_raw"], min_up_scale, max_up_scale)
    out["down_scale"] = np.clip(out["down_scale_raw"], min_down_scale, max_down_scale)

    out["cal_up_ext"] = out["pred_up_ext"] * out["up_scale"]
    out["cal_down_ext"] = out["pred_down_ext"] * out["down_scale"]

    out["cal_pred_open"] = out["pred_open"]
    out["cal_pred_close"] = out["pred_close"]
    out["cal_pred_high"] = out["pred_open"] + out["cal_up_ext"]
    out["cal_pred_low"] = out["pred_open"] - out["cal_down_ext"]
    out["cal_pred_range"] = out["cal_pred_high"] - out["cal_pred_low"]

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

    feature_cols = [
        "pred_range",
        "pred_up_ext",
        "pred_down_ext",
        "pred_up_share",
        "pred_down_share",
        "pred_gap_abs",
        "pred_close_from_open",
        "pred_close_from_open_abs",
        "weekday",
        "month",
        "pred_range_lag1",
        "pred_range_lag2",
        "pred_range_ma5",
        "actual_range_lag1",
        "actual_range_lag2",
        "actual_range_ma5",
        "pred_up_ext_lag1",
        "pred_up_ext_lag2",
        "pred_up_ext_ma5",
        "pred_down_ext_lag1",
        "pred_down_ext_lag2",
        "pred_down_ext_ma5",
        "actual_up_ext_lag1",
        "actual_up_ext_lag2",
        "actual_up_ext_ma5",
        "actual_down_ext_lag1",
        "actual_down_ext_lag2",
        "actual_down_ext_ma5",
        "pred_gap_abs_lag1",
        "pred_gap_abs_lag2",
        "pred_gap_abs_ma5",
        "pred_close_from_open_abs_lag1",
        "pred_close_from_open_abs_lag2",
        "pred_close_from_open_abs_ma5",
    ]
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    up_model = _fit_model(train_df, feature_cols, "up_scale_target")
    down_model = _fit_model(train_df, feature_cols, "down_scale_target")

    calibrated = _apply_two_sided_calibration(
        test_df,
        up_model=up_model,
        down_model=down_model,
        feature_cols=feature_cols,
        min_up_scale=args.min_up_scale,
        max_up_scale=args.max_up_scale,
        min_down_scale=args.min_down_scale,
        max_down_scale=args.max_down_scale,
    )

    orig_metrics = _metric_bundle(calibrated, "pred")
    cal_metrics = _metric_bundle(calibrated, "cal_pred")
    diff = {k: cal_metrics[k] - orig_metrics[k] for k in orig_metrics.keys()}

    monthly = _monthly_compare(calibrated)

    pred_cols = [
        "forecast_date", "prev_close",
        "pred_open", "pred_high", "pred_low", "pred_close",
        "cal_pred_open", "cal_pred_high", "cal_pred_low", "cal_pred_close",
        "actual_open", "actual_high", "actual_low", "actual_close",
        "pred_up_ext", "pred_down_ext",
        "actual_up_ext", "actual_down_ext",
        "up_scale_raw", "down_scale_raw", "up_scale", "down_scale",
        "cal_up_ext", "cal_down_ext", "pred_range", "cal_pred_range", "actual_range",
    ]
    pred_cols = [c for c in pred_cols if c in calibrated.columns]
    pred_out = calibrated[pred_cols].copy()

    report = {
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "test_date_min": calibrated["forecast_date"].min().strftime("%Y-%m-%d"),
        "test_date_max": calibrated["forecast_date"].max().strftime("%Y-%m-%d"),
        "min_up_scale": args.min_up_scale,
        "max_up_scale": args.max_up_scale,
        "min_down_scale": args.min_down_scale,
        "max_down_scale": args.max_down_scale,
        "feature_count": len(feature_cols),
        "up_feature_importance_top20": _feature_importance(up_model, feature_cols)[:20],
        "down_feature_importance_top20": _feature_importance(down_model, feature_cols)[:20],
        "original_metrics": orig_metrics,
        "calibrated_metrics": cal_metrics,
        "metric_diff_calibrated_minus_original": diff,
    }

    summary_path = outdir / "two_sided_range_calibrated_hybrid_summary.json"
    monthly_path = outdir / "two_sided_range_calibrated_hybrid_monthly.csv"
    preds_path = outdir / "two_sided_range_calibrated_hybrid_predictions.csv"

    summary_path.write_text(json.dumps(report, indent=2))
    monthly.to_csv(monthly_path, index=False)
    pred_out.to_csv(preds_path, index=False)

    print("Saved summary:", summary_path)
    print("Saved monthly:", monthly_path)
    print("Saved predictions:", preds_path)
    print()
    print("=== TWO-SIDED RANGE CALIBRATED HYBRID CHECK ===")
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
