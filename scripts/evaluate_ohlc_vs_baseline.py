#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.ohlc_benchmark import (
    build_naive_component_baseline,
    build_rolling_component_baseline,
    compare_metric_dicts,
    evaluate_ohlc_frame,
    export_detailed_test_predictions,
    export_feature_importance_csvs,
    export_feature_importance_summary,
    reconstruct_baseline_ohlc,
    save_json,
)
from src.features.builder import load_feature_matrix
from src.models.ohlc_forecaster import (
    align_features_and_ohlc_targets,
    evaluate_component_predictions,
    load_models,
    predict_ohlc_components,
    reconstruct_ohlc,
)
from src.targets.ohlc_targets import build_ohlc_component_targets


MODEL_DIR = Path("output/models/ohlc")
REPORT_FILE = Path("output/reports/ohlc_metrics_vs_baseline.json")
DETAIL_CSV = Path("output/analysis/ohlc/ohlc_test_window_predictions.csv")
FEATURE_IMP_DIR = Path("output/analysis/ohlc/feature_importance")
FEATURE_IMP_SUMMARY = Path("output/analysis/ohlc/feature_importance_summary.txt")
TEST_SIZE = 252
ROLLING_WINDOW = 20


def main() -> None:
    features = load_feature_matrix()
    if features is None or features.empty:
        raise RuntimeError("Feature matrix is empty or missing. Build features first.")

    spx = pd.read_parquet("data/raw/spx_daily.parquet")
    if "Date" in spx.columns:
        spx["Date"] = pd.to_datetime(spx["Date"])
        spx = spx.set_index("Date")
    spx.index = pd.to_datetime(spx.index)
    spx = spx.sort_index()

    targets = build_ohlc_component_targets(spx, dropna=True)
    X, y = align_features_and_ohlc_targets(features, targets)

    common_index = X.index.intersection(spx.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    spx = spx.loc[common_index]

    if len(X) <= TEST_SIZE + 252:
        raise RuntimeError(f"Not enough rows for evaluation. Got {len(X)} rows.")

    split_idx = len(X) - TEST_SIZE
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    spx_test = spx.iloc[split_idx:].copy()

    models = load_models(MODEL_DIR)
    model_components = predict_ohlc_components(models, X_test)
    mean_baseline_components = build_naive_component_baseline(y_train, X_test.index)
    rolling_baseline_components = build_rolling_component_baseline(y, X_test.index, window=ROLLING_WINDOW)

    prev_close = spx["Close"].shift(1).loc[X_test.index]

    model_ohlc = reconstruct_ohlc(prev_close=prev_close, component_preds=model_components)
    mean_baseline_ohlc = reconstruct_baseline_ohlc(prev_close=prev_close, component_preds=mean_baseline_components)
    rolling_baseline_ohlc = reconstruct_baseline_ohlc(prev_close=prev_close, component_preds=rolling_baseline_components)

    model_component_metrics = evaluate_component_predictions(y_test, model_components)
    mean_baseline_component_metrics = evaluate_component_predictions(y_test, mean_baseline_components)
    rolling_baseline_component_metrics = evaluate_component_predictions(y_test, rolling_baseline_components)

    model_ohlc_metrics = evaluate_ohlc_frame(
        actual_ohlc=spx_test[["Open", "High", "Low", "Close"]],
        pred_ohlc=model_ohlc,
        prev_close=prev_close,
    )
    mean_baseline_ohlc_metrics = evaluate_ohlc_frame(
        actual_ohlc=spx_test[["Open", "High", "Low", "Close"]],
        pred_ohlc=mean_baseline_ohlc,
        prev_close=prev_close,
    )
    rolling_baseline_ohlc_metrics = evaluate_ohlc_frame(
        actual_ohlc=spx_test[["Open", "High", "Low", "Close"]],
        pred_ohlc=rolling_baseline_ohlc,
        prev_close=prev_close,
    )

    report = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "rolling_window": ROLLING_WINDOW,
        "model_component_metrics": model_component_metrics,
        "mean_baseline_component_metrics": mean_baseline_component_metrics,
        "rolling_baseline_component_metrics": rolling_baseline_component_metrics,
        "component_improvement_vs_mean_baseline": compare_metric_dicts(
            model_component_metrics, mean_baseline_component_metrics
        ),
        "component_improvement_vs_rolling_baseline": compare_metric_dicts(
            model_component_metrics, rolling_baseline_component_metrics
        ),
        "model_ohlc_metrics": model_ohlc_metrics,
        "mean_baseline_ohlc_metrics": mean_baseline_ohlc_metrics,
        "rolling_baseline_ohlc_metrics": rolling_baseline_ohlc_metrics,
        "ohlc_improvement_vs_mean_baseline": compare_metric_dicts(
            model_ohlc_metrics, mean_baseline_ohlc_metrics
        ),
        "ohlc_improvement_vs_rolling_baseline": compare_metric_dicts(
            model_ohlc_metrics, rolling_baseline_ohlc_metrics
        ),
    }

    save_json(report, REPORT_FILE)

    export_detailed_test_predictions(
        output_file=DETAIL_CSV,
        actual_ohlc=spx_test[["Open", "High", "Low", "Close"]],
        prev_close=prev_close,
        y_test=y_test,
        model_components=model_components,
        mean_baseline_components=mean_baseline_components,
        rolling_baseline_components=rolling_baseline_components,
        model_ohlc=model_ohlc,
        mean_baseline_ohlc=mean_baseline_ohlc,
        rolling_baseline_ohlc=rolling_baseline_ohlc,
    )

    export_feature_importance_csvs(
        models=models,
        feature_names=X.columns,
        output_dir=FEATURE_IMP_DIR,
        top_n=20,
    )
    export_feature_importance_summary(
        models=models,
        feature_names=X.columns,
        output_file=FEATURE_IMP_SUMMARY,
        top_n=10,
    )

    print("\nSaved benchmark report to:", REPORT_FILE)
    print("Saved detailed test predictions to:", DETAIL_CSV)
    print("Saved feature importance CSVs to:", FEATURE_IMP_DIR)
    print("Saved feature importance summary to:", FEATURE_IMP_SUMMARY)

    print("\nOHLC improvement vs mean baseline:")
    for key, vals in report["ohlc_improvement_vs_mean_baseline"].items():
        if isinstance(vals, dict):
            joined = " ".join(f"{k}={v:.6f}" for k, v in vals.items())
            print(f"  {key}: {joined}")
        else:
            print(f"  {key}: {vals:.6f}")

    print("\nOHLC improvement vs rolling baseline:")
    for key, vals in report["ohlc_improvement_vs_rolling_baseline"].items():
        if isinstance(vals, dict):
            joined = " ".join(f"{k}={v:.6f}" for k, v in vals.items())
            print(f"  {key}: {joined}")
        else:
            print(f"  {key}: {vals:.6f}")


if __name__ == "__main__":
    main()
