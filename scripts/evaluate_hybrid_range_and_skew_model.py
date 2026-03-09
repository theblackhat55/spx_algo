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
        description="Evaluate hybrid forecast with separate range and skew models."
    )
    p.add_argument("--predictions", default=str(DEFAULT_PREDS))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--test-size", type=int, default=120)
    p.add_argument("--min-up-share", type=float, default=0.05)
    p.add_argument("--max-up-share", type=float, default=0.95)
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


def _safe_div(a: pd.Series, b: pd.Series, default: float = 0.5) -> pd.Series:
    out = pd.Series(default, index=a.index, dtype=float)
    mask = b.abs() > 1e-9
    out.loc[mask] = a.loc[mask] / b.loc[mask]
    return out


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["pred_range"] = out["pred_high"] - out["pred_low"]
    out["actual_range"] = out["actual_high"] - out["actual_low"]

    out["pred_up_ext"] = np.maximum(out["pred_high"] - out["pred_open"], 0.0)
    out["pred_down_ext"] = np.maximum(out["pred_open"] - out["pred_low"], 0.0)

    out["actual_up_ext_from_pred_open"] = np.maximum(out["actual_high"] - out["pred_open"], 0.0)
    out["actual_down_ext_from_pred_open"] = np.maximum(out["pred_open"] - out["actual_low"], 0.0)

    out["pred_up_share"] = _safe_div(out["pred_up_ext"], out["pred_range"], default=0.5)
    out["actual_up_share_from_pred_open"] = _safe_div(
        out["actual_up_ext_from_pred_open"], out["actual_range"], default=0.5
    ).clip(0.0, 1.0)

    out["pred_gap_abs"] = np.abs(out["pred_open"] / out["prev_close"] - 1.0)
    out["pred_close_from_open"] = out["pred_close"] / out["pred_open"] - 1.0
    out["pred_close_from_open_abs"] = np.abs(out["pred_close_from_open"])

    out["actual_gap_abs"] = np.abs(out["actual_open"] / out["prev_close"] - 1.0)
    out["actual_close_from_open"] = out["actual_close"] / out["actual_open"] - 1.0
    out["actual_close_from_open_abs"] = np.abs(out["actual_close_from_open"])

    out["weekday"] = out["forecast_date"].dt.weekday
    out["month"] = out["forecast_date"].dt.month

    lag_cols = [
        "pred_range", "actual_range",
        "pred_gap_abs", "actual_gap_abs",
        "pred_close_from_open_abs", "actual_close_from_open_abs",
        "pred_up_share", "actual_up_share_from_pred_open",
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


def _apply_models(
    df: pd.DataFrame,
    range_model: LGBMRegressor,
    skew_model: LGBMRegressor,
    range_features: list[str],
    skew_features: list[str],
    min_up_share: float,
    max_up_share: float,
) -> pd.DataFrame:
    out = df.copy()

    out["range_model_pred"] = range_model.predict(out[range_features])
    out["range_model_pred"] = np.maximum(out["range_model_pred"], 0.0)

    out["skew_model_pred_raw"] = skew_model.predict(out[skew_features])
    out["skew_model_pred"] = np.clip(out["skew_model_pred_raw"], min_up_share, max_up_share)

    out["rsk_pred_open"] = out["pred_open"]
    out["rsk_pred_close"] = out["pred_close"]

    out["rsk_pred_high"] = out["pred_open"] + out["range_model_pred"] * out["skew_model_pred"]
    out["rsk_pred_low"] = out["pred_open"] - out["range_model_pred"] * (1.0 - out["skew_model_pred"])
    out["rsk_pred_range"] = out["rsk_pred_high"] - out["rsk_pred_low"]

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
        new = _metric_bundle(g, "rsk_pred")

        for k, v in orig.items():
            row[f"orig_{k}"] = v
        for k, v in new.items():
            row[f"rsk_{k}"] = v
        for k in orig.keys():
            row[f"diff_{k}_rsk_minus_orig"] = new[k] - orig[k]
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

    range_features = [
        "pred_range",
        "pred_up_share",
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
        "pred_gap_abs_lag1",
        "pred_gap_abs_lag2",
        "pred_gap_abs_ma5",
        "pred_close_from_open_abs_lag1",
        "pred_close_from_open_abs_lag2",
        "pred_close_from_open_abs_ma5",
    ]
    range_features = [c for c in range_features if c in train_df.columns]

    skew_features = [
        "pred_up_share",
        "pred_gap_abs",
        "pred_close_from_open",
        "pred_close_from_open_abs",
        "pred_range",
        "weekday",
        "month",
        "pred_up_share_lag1",
        "pred_up_share_lag2",
        "pred_up_share_ma5",
        "actual_up_share_from_pred_open_lag1",
        "actual_up_share_from_pred_open_lag2",
        "actual_up_share_from_pred_open_ma5",
        "pred_gap_abs_lag1",
        "pred_gap_abs_lag2",
        "pred_gap_abs_ma5",
        "pred_close_from_open_abs_lag1",
        "pred_close_from_open_abs_lag2",
        "pred_close_from_open_abs_ma5",
        "actual_range_lag1",
        "actual_range_ma5",
    ]
    skew_features = [c for c in skew_features if c in train_df.columns]

    range_model = _fit_model(train_df, range_features, "actual_range")
    skew_model = _fit_model(train_df, skew_features, "actual_up_share_from_pred_open")

    evaluated = _apply_models(
        test_df,
        range_model=range_model,
        skew_model=skew_model,
        range_features=range_features,
        skew_features=skew_features,
        min_up_share=args.min_up_share,
        max_up_share=args.max_up_share,
    )

    orig_metrics = _metric_bundle(evaluated, "pred")
    new_metrics = _metric_bundle(evaluated, "rsk_pred")
    diff = {k: new_metrics[k] - orig_metrics[k] for k in orig_metrics.keys()}

    monthly = _monthly_compare(evaluated)

    pred_cols = [
        "forecast_date", "prev_close",
        "pred_open", "pred_high", "pred_low", "pred_close",
        "rsk_pred_open", "rsk_pred_high", "rsk_pred_low", "rsk_pred_close",
        "actual_open", "actual_high", "actual_low", "actual_close",
        "pred_range", "actual_range", "range_model_pred",
        "pred_up_share", "skew_model_pred_raw", "skew_model_pred", "rsk_pred_range",
    ]
    pred_cols = [c for c in pred_cols if c in evaluated.columns]
    pred_out = evaluated[pred_cols].copy()

    report = {
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "test_date_min": evaluated["forecast_date"].min().strftime("%Y-%m-%d"),
        "test_date_max": evaluated["forecast_date"].max().strftime("%Y-%m-%d"),
        "min_up_share": args.min_up_share,
        "max_up_share": args.max_up_share,
        "range_feature_count": len(range_features),
        "skew_feature_count": len(skew_features),
        "range_feature_importance_top20": _feature_importance(range_model, range_features)[:20],
        "skew_feature_importance_top20": _feature_importance(skew_model, skew_features)[:20],
        "original_metrics": orig_metrics,
        "range_skew_metrics": new_metrics,
        "metric_diff_range_skew_minus_original": diff,
    }

    summary_path = outdir / "hybrid_range_and_skew_model_summary.json"
    monthly_path = outdir / "hybrid_range_and_skew_model_monthly.csv"
    preds_path = outdir / "hybrid_range_and_skew_model_predictions.csv"

    summary_path.write_text(json.dumps(report, indent=2))
    monthly.to_csv(monthly_path, index=False)
    pred_out.to_csv(preds_path, index=False)

    print("Saved summary:", summary_path)
    print("Saved monthly:", monthly_path)
    print("Saved predictions:", preds_path)
    print()
    print("=== HYBRID RANGE+SKEW MODEL CHECK ===")
    print("rows_train:", report["rows_train"])
    print("rows_test:", report["rows_test"])
    print("test_range:", report["test_date_min"], "->", report["test_date_max"])
    print()
    print("Original metrics:")
    for k, v in report["original_metrics"].items():
        print(" ", k, v)
    print()
    print("Range+skew metrics:")
    for k, v in report["range_skew_metrics"].items():
        print(" ", k, v)
    print()
    print("Diff (range+skew - original):")
    for k, v in report["metric_diff_range_skew_minus_original"].items():
        print(" ", k, v)


if __name__ == "__main__":
    main()
