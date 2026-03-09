#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.builder import load_feature_matrix
from src.models.ohlc_forecaster import (
    align_features_and_ohlc_targets,
    evaluate_component_predictions,
    evaluate_reconstructed_ohlc,
    predict_ohlc_components,
    reconstruct_ohlc,
    save_metrics,
    save_models,
    train_ohlc_models,
)
from src.targets.ohlc_targets import build_ohlc_component_targets


MODEL_DIR = Path("output/models/ohlc")
METRICS_FILE = Path("output/reports/ohlc_metrics.json")
TEST_SIZE = 252


def main() -> None:
    features = load_feature_matrix()
    if features is None or features.empty:
        raise RuntimeError(
            "Feature matrix is empty or missing. Run feature builder first."
        )

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
        raise RuntimeError(
            f"Not enough rows for train/test split. Need > {TEST_SIZE + 252}, got {len(X)}"
        )

    split_idx = len(X) - TEST_SIZE
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    spx_test = spx.iloc[split_idx:].copy()

    models = train_ohlc_models(X_train, y_train)
    component_preds = predict_ohlc_components(models, X_test)

    prev_close = spx["Close"].shift(1).loc[X_test.index]
    pred_ohlc = reconstruct_ohlc(prev_close=prev_close, component_preds=component_preds)

    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "component_metrics": evaluate_component_predictions(y_test, component_preds),
        "ohlc_metrics": evaluate_reconstructed_ohlc(
            actual_ohlc=spx_test[["Open", "High", "Low", "Close"]],
            pred_ohlc=pred_ohlc,
            prev_close=prev_close,
        ),
    }

    save_models(models, MODEL_DIR)
    save_metrics(metrics, METRICS_FILE)

    print("\nSaved OHLC models to:", MODEL_DIR)
    print("Saved metrics to:", METRICS_FILE)
    print("\nComponent metrics:")
    for target, vals in metrics["component_metrics"].items():
        print(f"  {target}: MAE={vals['mae']:.6f} RMSE={vals['rmse']:.6f}")

    print("\nReconstructed OHLC metrics:")
    for key, vals in metrics["ohlc_metrics"].items():
        if isinstance(vals, dict):
            joined = " ".join(f"{k}={v:.6f}" for k, v in vals.items())
            print(f"  {key}: {joined}")
        else:
            print(f"  {key}: {vals:.6f}")


if __name__ == "__main__":
    main()
