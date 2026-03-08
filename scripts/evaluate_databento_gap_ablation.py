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
REPORT_PATH = Path("output/reports/databento_gap_ablation.json")
CSV_PATH = Path("output/analysis/ohlc/databento_gap_ablation_predictions.csv")

DB_OPEN_FEATURES = [
    "es_overnight_ret",
    "es_preopen_ret_last_60m",
    "es_preopen_ret_last_30m",
    "es_overnight_range_pct",
]


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


def _metric_diff(base_metrics: dict, new_metrics: dict) -> dict:
    out: dict = {}
    for key in new_metrics:
        if isinstance(new_metrics[key], dict) and key in base_metrics:
            out[key] = _metric_diff(base_metrics[key], new_metrics[key])
        elif (
            isinstance(new_metrics[key], (int, float))
            and key in base_metrics
            and isinstance(base_metrics[key], (int, float))
        ):
            out[key] = new_metrics[key] - base_metrics[key]
    return out


def _train_predict_all(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    models = train_ohlc_models(
        X_train=X_train,
        y_train=y_train[OHLC_TARGET_COLUMNS],
        model_params=_default_model_params(),
    )
    return predict_ohlc_components(models, X_test)


def _train_single_target(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
):
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(**_default_model_params())
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return pd.Series(pred, index=X_test.index, name=y_train.name)


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

    missing = [c for c in DB_OPEN_FEATURES if c not in db_feat.columns]
    if missing:
        raise ValueError(f"Missing required Databento features: {missing}")

    db_small = db_feat[DB_OPEN_FEATURES].copy()
    targets = build_ohlc_component_targets(raw_spx)
    targets.index = pd.to_datetime(targets.index)

    overlap_idx = (
        features_base.index
        .intersection(db_small.index)
        .intersection(targets.index)
        .intersection(raw_spx.index)
    )

    X_base = features_base.loc[overlap_idx].copy()
    X_gap = X_base.join(db_small, how="inner")
    y = targets.loc[overlap_idx].copy()
    raw_spx_recent = raw_spx.loc[overlap_idx].copy()

    X_base_aligned, y_base_aligned = _align(X_base, y)
    X_gap_aligned, y_gap_aligned = _align(X_gap, y)

    Xb_train, Xb_test, yb_train, yb_test = _split_recent(X_base_aligned, y_base_aligned, TEST_SIZE)
    Xg_train, Xg_test, yg_train, yg_test = _split_recent(X_gap_aligned, y_gap_aligned, TEST_SIZE)

    # Baseline: current model on all targets using base features only
    base_pred_components = _train_predict_all(Xb_train, yb_train, Xb_test)

    # Ablation: only target_gap_ret gets the extra Databento features
    # other targets remain on base model
    gap_series = _train_single_target(
        X_train=Xg_train,
        y_train=yg_train["target_gap_ret"],
        X_test=Xg_test,
    )

    ablation_pred_components = base_pred_components.copy()
    ablation_pred_components["target_gap_ret"] = gap_series.reindex(ablation_pred_components.index)

    base_component_metrics = evaluate_component_predictions(
        y_true=yb_test[OHLC_TARGET_COLUMNS],
        y_pred=base_pred_components,
    )
    ablation_component_metrics = evaluate_component_predictions(
        y_true=yg_test[OHLC_TARGET_COLUMNS],
        y_pred=ablation_pred_components,
    )

    prev_close = raw_spx_recent["Close"].shift(1).reindex(Xb_test.index)
    actual_ohlc = raw_spx_recent.loc[Xb_test.index, ["Open", "High", "Low", "Close"]].copy()

    base_pred_ohlc = reconstruct_ohlc(
        prev_close=prev_close,
        component_preds=base_pred_components,
    )
    ablation_pred_ohlc = reconstruct_ohlc(
        prev_close=prev_close,
        component_preds=ablation_pred_components,
    )

    base_ohlc_metrics = evaluate_reconstructed_ohlc(
        actual_ohlc=actual_ohlc,
        pred_ohlc=base_pred_ohlc,
        prev_close=prev_close,
    )
    ablation_ohlc_metrics = evaluate_reconstructed_ohlc(
        actual_ohlc=actual_ohlc,
        pred_ohlc=ablation_pred_ohlc,
        prev_close=prev_close,
    )

    export = actual_ohlc.copy()
    export.columns = [f"actual_{c.lower()}" for c in export.columns]
    export = export.join(prev_close.rename("prev_close"))
    export = export.join(base_pred_components.add_prefix("base_component_"))
    export = export.join(ablation_pred_components.add_prefix("ablation_component_"))
    export = export.join(base_pred_ohlc.add_prefix("base_pred_"))
    export = export.join(ablation_pred_ohlc.add_prefix("ablation_pred_"))
    export.to_csv(CSV_PATH)

    report = {
        "overlap_rows": len(overlap_idx),
        "test_size": TEST_SIZE,
        "base_feature_count": X_base_aligned.shape[1],
        "gap_ablation_feature_count": X_gap_aligned.shape[1],
        "databento_gap_features": DB_OPEN_FEATURES,
        "base_component_metrics": base_component_metrics,
        "gap_ablation_component_metrics": ablation_component_metrics,
        "component_metric_diff_ablation_minus_base": _metric_diff(base_component_metrics, ablation_component_metrics),
        "base_ohlc_metrics": base_ohlc_metrics,
        "gap_ablation_ohlc_metrics": ablation_ohlc_metrics,
        "ohlc_metric_diff_ablation_minus_base": _metric_diff(base_ohlc_metrics, ablation_ohlc_metrics),
        "csv_path": str(CSV_PATH),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"Saved Databento gap ablation report to: {REPORT_PATH}")
    print(f"Saved prediction CSV to: {CSV_PATH}")

    print("\n=== BASE OHLC METRICS ===")
    for k, v in base_ohlc_metrics.items():
        print(k, v)

    print("\n=== GAP ABLATION OHLC METRICS ===")
    for k, v in ablation_ohlc_metrics.items():
        print(k, v)

    print("\n=== DIFF (ABLATION - BASE) ===")
    for k, v in report["ohlc_metric_diff_ablation_minus_base"].items():
        print(k, v)


if __name__ == "__main__":
    main()
