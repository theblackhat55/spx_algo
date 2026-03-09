#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BASE_PREDS = Path("output/backtests/base_ohlc/predictions.csv")
DEFAULT_BASE_SUMMARY = Path("output/backtests/base_ohlc/summary.json")
DEFAULT_BASE_MONTHLY = Path("output/backtests/base_ohlc/monthly_metrics.csv")

DEFAULT_HYBRID_PREDS = Path("output/backtests/gap_augmented_hybrid/predictions.csv")
DEFAULT_HYBRID_SUMMARY = Path("output/backtests/gap_augmented_hybrid/summary.json")
DEFAULT_HYBRID_MONTHLY = Path("output/backtests/gap_augmented_hybrid/monthly_metrics.csv")

DEFAULT_OUTDIR = Path("output/reports")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare walk-forward base OHLC vs Databento gap-hybrid backtests."
    )
    p.add_argument("--base-preds", default=str(DEFAULT_BASE_PREDS))
    p.add_argument("--base-summary", default=str(DEFAULT_BASE_SUMMARY))
    p.add_argument("--base-monthly", default=str(DEFAULT_BASE_MONTHLY))

    p.add_argument("--hybrid-preds", default=str(DEFAULT_HYBRID_PREDS))
    p.add_argument("--hybrid-summary", default=str(DEFAULT_HYBRID_SUMMARY))
    p.add_argument("--hybrid-monthly", default=str(DEFAULT_HYBRID_MONTHLY))

    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text())


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    return pd.read_csv(path)


def _metric_bundle(df: pd.DataFrame, prefix: str = "pred") -> dict[str, float]:
    out: dict[str, float] = {}

    for field in ["open", "high", "low", "close"]:
        err = df[f"{prefix}_{field}"] - df[f"actual_{field}"]
        out[f"{field}_mae"] = float(np.abs(err).mean())
        out[f"{field}_rmse"] = float(np.sqrt(np.mean(np.square(err))))
        out[f"{field}_bias"] = float(err.mean())

    close_dir_actual = np.sign(df["actual_close"] - df["prev_close"])
    close_dir_pred = np.sign(df[f"{prefix}_close"] - df["prev_close"])
    out["close_direction_accuracy"] = float((close_dir_actual == close_dir_pred).mean())

    gap_actual = np.sign(df["actual_open"] / df["prev_close"] - 1.0)
    gap_pred = np.sign(df[f"{prefix}_open"] / df["prev_close"] - 1.0)
    out["gap_sign_accuracy"] = float((gap_actual == gap_pred).mean())

    actual_range = df["actual_high"] - df["actual_low"]
    pred_range = df[f"{prefix}_high"] - df[f"{prefix}_low"]
    out["range_mae"] = float(np.abs(pred_range - actual_range).mean())
    out["actual_range_mean"] = float(actual_range.mean())
    out["pred_range_mean"] = float(pred_range.mean())

    return out


def _diff_metrics(hybrid: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    diff = {}
    keys = sorted(set(base.keys()) & set(hybrid.keys()))
    for k in keys:
        if isinstance(base[k], (int, float)) and isinstance(hybrid[k], (int, float)):
            diff[k] = hybrid[k] - base[k]
    return diff


def _better_direction(metric_name: str) -> str:
    higher_better = {
        "close_direction_accuracy",
        "gap_sign_accuracy",
    }
    lower_better = {
        "open_mae", "open_rmse", "open_bias",
        "high_mae", "high_rmse", "high_bias",
        "low_mae", "low_rmse", "low_bias",
        "close_mae", "close_rmse", "close_bias",
        "range_mae",
    }
    if metric_name in higher_better:
        return "higher"
    if metric_name in lower_better:
        return "lower"
    return "context"


def _winner(metric_name: str, base_val: float, hybrid_val: float) -> str:
    direction = _better_direction(metric_name)
    if direction == "higher":
        if hybrid_val > base_val:
            return "hybrid"
        if hybrid_val < base_val:
            return "base"
        return "tie"
    if direction == "lower":
        if hybrid_val < base_val:
            return "hybrid"
        if hybrid_val > base_val:
            return "base"
        return "tie"
    return "n/a"


def _build_monthly_compare(base_monthly: pd.DataFrame, hybrid_monthly: pd.DataFrame) -> pd.DataFrame:
    base = base_monthly.copy()
    hybrid = hybrid_monthly.copy()

    base = base.add_prefix("base_")
    hybrid = hybrid.add_prefix("hybrid_")

    if "base_month" not in base.columns or "hybrid_month" not in hybrid.columns:
        raise ValueError("Monthly CSVs must contain a 'month' column.")

    merged = base.merge(
        hybrid,
        left_on="base_month",
        right_on="hybrid_month",
        how="inner",
    ).rename(columns={"base_month": "month"})

    drop_cols = [c for c in ["hybrid_month"] if c in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    compare_metrics = [
        "open_mae", "high_mae", "low_mae", "close_mae",
        "open_bias", "high_bias", "low_bias", "close_bias",
        "range_mae", "close_direction_accuracy"
    ]
    for m in compare_metrics:
        b = f"base_{m}"
        h = f"hybrid_{m}"
        if b in merged.columns and h in merged.columns:
            merged[f"diff_{m}_hybrid_minus_base"] = merged[h] - merged[b]

    return merged.sort_values("month").reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_preds = _read_csv(Path(args.base_preds))
    base_summary = _read_json(Path(args.base_summary))
    base_monthly = _read_csv(Path(args.base_monthly))

    hybrid_preds = _read_csv(Path(args.hybrid_preds))
    hybrid_summary = _read_json(Path(args.hybrid_summary))
    hybrid_monthly = _read_csv(Path(args.hybrid_monthly))

    required_cols = {
        "forecast_date", "prev_close",
        "pred_open", "pred_high", "pred_low", "pred_close",
        "actual_open", "actual_high", "actual_low", "actual_close",
    }
    for name, df in [("base_preds", base_preds), ("hybrid_preds", hybrid_preds)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")

    base_preds["forecast_date"] = pd.to_datetime(base_preds["forecast_date"])
    hybrid_preds["forecast_date"] = pd.to_datetime(hybrid_preds["forecast_date"])

    overlap = base_preds.merge(
        hybrid_preds,
        on="forecast_date",
        suffixes=("_base", "_hybrid"),
        how="inner",
    )

    if overlap.empty:
        raise SystemExit("No overlapping forecast dates between base and hybrid predictions.")

    compare_df = pd.DataFrame({
        "forecast_date": overlap["forecast_date"],
        "prev_close": overlap["prev_close_base"],
        "actual_open": overlap["actual_open_base"],
        "actual_high": overlap["actual_high_base"],
        "actual_low": overlap["actual_low_base"],
        "actual_close": overlap["actual_close_base"],
        "base_open": overlap["pred_open_base"],
        "base_high": overlap["pred_high_base"],
        "base_low": overlap["pred_low_base"],
        "base_close": overlap["pred_close_base"],
        "hybrid_open": overlap["pred_open_hybrid"],
        "hybrid_high": overlap["pred_high_hybrid"],
        "hybrid_low": overlap["pred_low_hybrid"],
        "hybrid_close": overlap["pred_close_hybrid"],
    })

    base_eval = compare_df.rename(columns={
        "base_open": "pred_open",
        "base_high": "pred_high",
        "base_low": "pred_low",
        "base_close": "pred_close",
    })[[
        "forecast_date", "prev_close",
        "pred_open", "pred_high", "pred_low", "pred_close",
        "actual_open", "actual_high", "actual_low", "actual_close"
    ]]

    hybrid_eval = compare_df.rename(columns={
        "hybrid_open": "pred_open",
        "hybrid_high": "pred_high",
        "hybrid_low": "pred_low",
        "hybrid_close": "pred_close",
    })[[
        "forecast_date", "prev_close",
        "pred_open", "pred_high", "pred_low", "pred_close",
        "actual_open", "actual_high", "actual_low", "actual_close"
    ]]

    overlap_base_metrics = _metric_bundle(base_eval)
    overlap_hybrid_metrics = _metric_bundle(hybrid_eval)
    overlap_diff = _diff_metrics(overlap_hybrid_metrics, overlap_base_metrics)

    winners = {}
    for k in sorted(overlap_base_metrics.keys()):
        bv = overlap_base_metrics[k]
        hv = overlap_hybrid_metrics[k]
        if isinstance(bv, (int, float)) and isinstance(hv, (int, float)):
            winners[k] = _winner(k, bv, hv)

    monthly_compare = _build_monthly_compare(base_monthly, hybrid_monthly)

    report = {
        "overlap_rows": int(len(compare_df)),
        "base_rows_full": base_summary.get("rows"),
        "hybrid_rows_full": hybrid_summary.get("rows"),
        "overlap_date_min": compare_df["forecast_date"].min().strftime("%Y-%m-%d"),
        "overlap_date_max": compare_df["forecast_date"].max().strftime("%Y-%m-%d"),
        "base_metrics_on_overlap": overlap_base_metrics,
        "hybrid_metrics_on_overlap": overlap_hybrid_metrics,
        "metric_diff_hybrid_minus_base": overlap_diff,
        "metric_winner": winners,
    }

    summary_path = outdir / "walkforward_base_vs_gap_hybrid_summary.json"
    monthly_path = outdir / "walkforward_base_vs_gap_hybrid_monthly.csv"

    summary_path.write_text(json.dumps(report, indent=2))
    monthly_compare.to_csv(monthly_path, index=False)

    print("Saved summary:", summary_path)
    print("Saved monthly:", monthly_path)
    print()
    print("=== WALKFORWARD BASE VS GAP HYBRID CHECK ===")
    print("overlap_rows:", report["overlap_rows"])
    print("overlap_range:", report["overlap_date_min"], "->", report["overlap_date_max"])
    print()

    print("Base metrics on overlap:")
    for k, v in report["base_metrics_on_overlap"].items():
        print(" ", k, v)

    print()
    print("Hybrid metrics on overlap:")
    for k, v in report["hybrid_metrics_on_overlap"].items():
        print(" ", k, v)

    print()
    print("Diff (hybrid - base):")
    for k, v in report["metric_diff_hybrid_minus_base"].items():
        print(" ", k, v)

    print()
    print("Winners:")
    for k, v in report["metric_winner"].items():
        print(" ", k, v)


if __name__ == "__main__":
    main()
