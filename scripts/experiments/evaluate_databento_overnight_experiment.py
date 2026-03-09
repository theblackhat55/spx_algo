#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.builder import build_feature_matrix
from src.features.es_databento_overnight import load_es_databento_overnight_features
from src.models.ohlc_forecaster import (
    OHLC_TARGET_COLUMNS,
    _default_model_params,
    train_ohlc_models,
    predict_ohlc_components,
    reconstruct_ohlc,
    evaluate_component_predictions,
    evaluate_reconstructed_ohlc,
)
from src.targets.ohlc_targets import build_ohlc_component_targets


TEST_SIZE = 60
RAW_SPX_PATH = Path("data/raw/spx_daily.parquet")
REPORT_PATH = Path("output/reports/databento_overnight_experiment.json")
CSV_PATH = Path("output/analysis/ohlc/databento_overnight_predictions.csv")


def _align(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = X.index.intersection(y.index)
    X2 = X.loc[idx].sort_index()
    y2 = y.loc[idx].sort_index()
    joined = X2.join(y2, how="inner").dropna()
    return joined[X2.columns], joined[y2.columns]


def _split_recent(
    X: pd.DataFrame, y: pd.DataFrame, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(X) <= test_size:
        raise ValueError(f"Not enough rows ({len(X)}) for test_size={test_size}")
    return X.iloc[:-test_size], X.iloc[-test_size:], y.iloc[:-test_size], y.iloc[-test_size:]


def _metric_diff(base_metrics: dict, db_metrics: dict) -> dict:
    out: dict = {}
    for key in db_metrics:
        if isinstance(db_metrics[key], dict) and key in base_metrics:
            out[key] = _metric_diff(base_metrics[key], db_metrics[key])
        elif (
            isinstance(db_metrics[key], (int, float))
            and key in base_metrics
            and isinstance(base_metrics[key], (int, float))
        ):
            out[key] = db_metrics[key] - base_metrics[key]
    return out


def _run_case(X: pd.DataFrame, y: pd.DataFrame, raw_spx: pd.DataFrame) -> dict:
    X_aligned, y_aligned = _align(X, y)
    X_train, X_test, y_train, y_test = _split_recent(X_aligned, y_aligned, TEST_SIZE)

    models = train_ohlc_models(
        X_train=X_train,
        y_train=y_train[OHLC_TARGET_COLUMNS],
        model_params=_default_model_params(),
    )

    pred_components = predict_ohlc_components(models, X_test)
    component_metrics = evaluate_component_predictions(
        y_true=y_test[OHLC_TARGET_COLUMNS],
        y_pred=pred_components,
    )

    prev_close = raw_spx["Close"].shift(1).reindex(X_test.index)
    actual_ohlc = raw_spx.loc[X_test.index, ["Open", "High", "Low", "Close"]].copy()

    pred_ohlc = reconstruct_ohlc(
        prev_close=prev_close,
        component_preds=pred_components,
    )

    ohlc_metrics = evaluate_reconstructed_ohlc(
        actual_ohlc=actual_ohlc,
        pred_ohlc=pred_ohlc,
        prev_close=prev_close,
    )

    pred_export = actual_ohlc.copy()
    pred_export.columns = [f"actual_{c.lower()}" for c in pred_export.columns]
    pred_export = pred_export.join(pred_ohlc.add_prefix("pred_"))
    pred_export = pred_export.join(prev_close.rename("prev_close"))
    pred_export = pred_export.join(pred_components.add_prefix("pred_component_"))

    return {
        "rows_total": len(X_aligned),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": X_aligned.shape[1],
        "component_metrics": component_metrics,
        "ohlc_metrics": ohlc_metrics,
        "predictions": pred_export,
    }


def main() -> None:
    if not RAW_SPX_PATH.exists():
        raise FileNotFoundError(f"Missing SPX data: {RAW_SPX_PATH}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_spx = pd.read_parquet(RAW_SPX_PATH).copy()
    raw_spx.index = pd.to_datetime(raw_spx.index)

    features_base = build_feature_matrix()
    features_base.index = pd.to_datetime(features_base.index)

    db_feat = load_es_databento_overnight_features()
    db_feat.index = pd.to_datetime(db_feat.index)

    targets = build_ohlc_component_targets(raw_spx)
    targets.index = pd.to_datetime(targets.index)

    overlap_idx = (
        features_base.index
        .intersection(db_feat.index)
        .intersection(targets.index)
        .intersection(raw_spx.index)
    )

    features_base_recent = features_base.loc[overlap_idx].copy()
    features_plus_db = features_base_recent.join(db_feat, how="inner")
    targets_recent = targets.loc[overlap_idx].copy()
    raw_spx_recent = raw_spx.loc[overlap_idx].copy()

    base_result = _run_case(features_base_recent, targets_recent, raw_spx_recent)
    db_result = _run_case(features_plus_db, targets_recent, raw_spx_recent)

    diff_ohlc = _metric_diff(base_result["ohlc_metrics"], db_result["ohlc_metrics"])
    diff_components = _metric_diff(base_result["component_metrics"], db_result["component_metrics"])

    export = base_result["predictions"].add_prefix("base_").join(
        db_result["predictions"].add_prefix("db_"),
        how="outer",
    )
    export.to_csv(CSV_PATH)

    report = {
        "overlap_rows": len(overlap_idx),
        "test_size": TEST_SIZE,
        "base_feature_count": base_result["feature_count"],
        "databento_feature_count": db_result["feature_count"],
        "databento_added_columns": list(db_feat.columns),
        "base_rows_total": base_result["rows_total"],
        "db_rows_total": db_result["rows_total"],
        "train_rows": db_result["train_rows"],
        "test_rows": db_result["test_rows"],
        "base_component_metrics": base_result["component_metrics"],
        "databento_component_metrics": db_result["component_metrics"],
        "component_metric_diff_databento_minus_base": diff_components,
        "base_ohlc_metrics": base_result["ohlc_metrics"],
        "databento_ohlc_metrics": db_result["ohlc_metrics"],
        "ohlc_metric_diff_databento_minus_base": diff_ohlc,
        "csv_path": str(CSV_PATH),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"Saved Databento overnight experiment report to: {REPORT_PATH}")
    print(f"Saved prediction comparison CSV to: {CSV_PATH}")
    print("\n=== BASE OHLC METRICS ===")
    for k, v in base_result["ohlc_metrics"].items():
        print(k, v)
    print("\n=== DATABENTO OHLC METRICS ===")
    for k, v in db_result["ohlc_metrics"].items():
        print(k, v)
    print("\n=== DIFF (DATABENTO - BASE) ===")
    for k, v in diff_ohlc.items():
        print(k, v)


if __name__ == "__main__":
    main()
