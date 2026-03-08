#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.ohlc_benchmark import (
    build_rolling_component_baseline,
    evaluate_ohlc_frame,
    save_json,
)
from src.evaluation.ohlc_hybrid import (
    build_hybrid_components,
    choose_best_component_source,
    export_hybrid_predictions,
    save_text,
    summarize_selection,
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
REPORT_FILE = Path("output/reports/ohlc_hybrid_report.json")
DETAIL_CSV = Path("output/analysis/ohlc/ohlc_hybrid_test_window_predictions.csv")
SUMMARY_FILE = Path("output/analysis/ohlc/hybrid_selection_summary.txt")
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
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    spx_test = spx.iloc[split_idx:].copy()

    models = load_models(MODEL_DIR)
    model_components = predict_ohlc_components(models, X_test)
    rolling_components = build_rolling_component_baseline(y, X_test.index, window=ROLLING_WINDOW)

    model_component_metrics = evaluate_component_predictions(y_test, model_components)
    rolling_component_metrics = evaluate_component_predictions(y_test, rolling_components)

    selection = choose_best_component_source(
        model_component_metrics=model_component_metrics,
        rolling_component_metrics=rolling_component_metrics,
    )

    hybrid_components = build_hybrid_components(
        model_components=model_components,
        rolling_components=rolling_components,
        selection=selection,
    )

    prev_close = spx["Close"].shift(1).loc[X_test.index]

    model_ohlc = reconstruct_ohlc(prev_close=prev_close, component_preds=model_components)
    rolling_ohlc = reconstruct_ohlc(prev_close=prev_close, component_preds=rolling_components)
    hybrid_ohlc = reconstruct_ohlc(prev_close=prev_close, component_preds=hybrid_components)

    model_ohlc_metrics = evaluate_ohlc_frame(
        actual_ohlc=spx_test[["Open", "High", "Low", "Close"]],
        pred_ohlc=model_ohlc,
        prev_close=prev_close,
    )
    rolling_ohlc_metrics = evaluate_ohlc_frame(
        actual_ohlc=spx_test[["Open", "High", "Low", "Close"]],
        pred_ohlc=rolling_ohlc,
        prev_close=prev_close,
    )
    hybrid_ohlc_metrics = evaluate_ohlc_frame(
        actual_ohlc=spx_test[["Open", "High", "Low", "Close"]],
        pred_ohlc=hybrid_ohlc,
        prev_close=prev_close,
    )

    report = {
        "test_rows": int(len(X_test)),
        "rolling_window": ROLLING_WINDOW,
        "component_selection": selection,
        "model_component_metrics": model_component_metrics,
        "rolling_component_metrics": rolling_component_metrics,
        "model_ohlc_metrics": model_ohlc_metrics,
        "rolling_ohlc_metrics": rolling_ohlc_metrics,
        "hybrid_ohlc_metrics": hybrid_ohlc_metrics,
    }

    save_json(report, REPORT_FILE)
    save_text(
        summarize_selection(selection, model_component_metrics, rolling_component_metrics),
        SUMMARY_FILE,
    )
    export_hybrid_predictions(
        output_file=DETAIL_CSV,
        actual_ohlc=spx_test[["Open", "High", "Low", "Close"]],
        prev_close=prev_close,
        y_test=y_test,
        model_components=model_components,
        rolling_components=rolling_components,
        hybrid_components=hybrid_components,
        model_ohlc=model_ohlc,
        rolling_ohlc=rolling_ohlc,
        hybrid_ohlc=hybrid_ohlc,
    )

    print("\nSaved hybrid report to:", REPORT_FILE)
    print("Saved hybrid prediction details to:", DETAIL_CSV)
    print("Saved hybrid selection summary to:", SUMMARY_FILE)

    print("\nSelected component source per target:")
    for target, source in selection.items():
        print(f"  {target}: {source}")

    print("\nHybrid OHLC metrics:")
    for key, vals in hybrid_ohlc_metrics.items():
        if isinstance(vals, dict):
            joined = " ".join(f"{k}={v:.6f}" for k, v in vals.items())
            print(f"  {key}: {joined}")
        else:
            print(f"  {key}: {vals:.6f}")


if __name__ == "__main__":
    main()
