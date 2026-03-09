#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.builder import build_feature_matrix
from src.features.es_databento_overnight import load_es_databento_overnight_features
from src.models.gap_forecaster_databento import (
    GAP_TARGET,
    build_gap_feature_matrix,
    align_gap_data,
    train_gap_model,
    predict_gap,
    save_gap_model,
    feature_importance_frame,
)
from src.targets.ohlc_targets import build_ohlc_component_targets


RAW_SPX_PATH = Path("data/raw/spx_daily.parquet")
MODEL_PATH = Path("output/models/ohlc/gap_databento_model.joblib")
REPORT_PATH = Path("output/reports/gap_databento_model_report.json")
FI_PATH = Path("output/reports/gap_databento_feature_importance.csv")
TEST_SIZE = 60


def main() -> None:
    raw_spx = pd.read_parquet(RAW_SPX_PATH).copy()
    raw_spx.index = pd.to_datetime(raw_spx.index)

    base_features = build_feature_matrix()
    base_features.index = pd.to_datetime(base_features.index)

    db_feat = load_es_databento_overnight_features()
    db_feat.index = pd.to_datetime(db_feat.index)

    targets = build_ohlc_component_targets(raw_spx)
    targets.index = pd.to_datetime(targets.index)

    X = build_gap_feature_matrix(base_features, db_feat)
    y = targets[GAP_TARGET].copy()

    X_aligned, y_aligned = align_gap_data(X, y)

    if len(X_aligned) <= TEST_SIZE:
        raise RuntimeError(f"Not enough aligned rows ({len(X_aligned)}) for TEST_SIZE={TEST_SIZE}")

    X_train, X_test = X_aligned.iloc[:-TEST_SIZE], X_aligned.iloc[-TEST_SIZE:]
    y_train, y_test = y_aligned.iloc[:-TEST_SIZE], y_aligned.iloc[-TEST_SIZE:]

    model = train_gap_model(X_train, y_train)
    pred = predict_gap(model, X_test)

    mae = float((pred - y_test).abs().mean())
    rmse = float((((pred - y_test) ** 2).mean()) ** 0.5)

    save_gap_model(model, MODEL_PATH)

    fi = feature_importance_frame(model, list(X_train.columns))
    FI_PATH.parent.mkdir(parents=True, exist_ok=True)
    fi.to_csv(FI_PATH, index=False)

    report = {
        "rows_total": int(len(X_aligned)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X_aligned.shape[1]),
        "target": GAP_TARGET,
        "mae": mae,
        "rmse": rmse,
        "model_path": str(MODEL_PATH),
        "feature_importance_path": str(FI_PATH),
        "databento_gap_features": [c for c in X_train.columns if c.startswith("es_")],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"Saved Databento gap model to: {MODEL_PATH}")
    print(f"Saved report to: {REPORT_PATH}")
    print(f"Saved feature importance to: {FI_PATH}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
