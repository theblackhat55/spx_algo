#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from src.features.builder import load_feature_matrix
from src.features.es_overnight import load_es_overnight_features
from src.models.ohlc_forecaster import (
    align_features_and_ohlc_targets,
    train_ohlc_models,
    predict_ohlc_components,
    reconstruct_ohlc,
    evaluate_component_predictions,
    evaluate_reconstructed_ohlc,
)
from src.targets.ohlc_targets import build_ohlc_component_targets


REPORT_FILE = Path("output/reports/recent_overnight_experiment.json")
DETAIL_FILE = Path("output/analysis/ohlc/recent_overnight_predictions.csv")
OVERNIGHT_FILE = Path("data/processed/es_overnight_features.parquet")

TEST_SIZE = 10
MIN_TRAIN_ROWS = 20


def _load_spx() -> pd.DataFrame:
    spx = pd.read_parquet("data/raw/spx_daily.parquet")
    if "Date" in spx.columns:
        spx["Date"] = pd.to_datetime(spx["Date"])
        spx = spx.set_index("Date")
    spx.index = pd.to_datetime(spx.index)
    spx = spx.sort_index()
    return spx


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    if not OVERNIGHT_FILE.exists():
        raise FileNotFoundError(f"Missing overnight feature file: {OVERNIGHT_FILE}")

    features = load_feature_matrix()
    overnight = load_es_overnight_features(str(OVERNIGHT_FILE))
    spx = _load_spx()
    targets = build_ohlc_component_targets(spx, dropna=True)

    features.index = pd.to_datetime(features.index)
    overnight.index = pd.to_datetime(overnight.index)
    spx.index = pd.to_datetime(spx.index)
    targets.index = pd.to_datetime(targets.index)

    overlap = features.index.intersection(overnight.index).intersection(targets.index).intersection(spx.index)
    overlap = overlap.sort_values()

    if len(overlap) < (MIN_TRAIN_ROWS + TEST_SIZE):
        raise RuntimeError(
            f"Not enough overlap rows for recent overnight experiment. "
            f"Need at least {MIN_TRAIN_ROWS + TEST_SIZE}, got {len(overlap)}"
        )

    base_recent = features.loc[overlap].copy()
    overnight_recent = features.loc[overlap].join(overnight.loc[overlap], how="left")
    targets_recent = targets.loc[overlap].copy()
    spx_recent = spx.loc[overlap].copy()

    X_base, y = align_features_and_ohlc_targets(base_recent, targets_recent)
    X_overnight, y2 = align_features_and_ohlc_targets(overnight_recent, targets_recent)

    common = X_base.index.intersection(X_overnight.index).intersection(y.index).intersection(y2.index)
    X_base = X_base.loc[common]
    X_overnight = X_overnight.loc[common]
    y = y.loc[common]
    spx_recent = spx_recent.loc[common]

    split_idx = len(common) - TEST_SIZE
    Xb_train, Xb_test = X_base.iloc[:split_idx], X_base.iloc[split_idx:]
    Xo_train, Xo_test = X_overnight.iloc[:split_idx], X_overnight.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    spx_test = spx_recent.iloc[split_idx:].copy()

    base_models = train_ohlc_models(Xb_train, y_train)
    overnight_models = train_ohlc_models(Xo_train, y_train)

    base_comp = predict_ohlc_components(base_models, Xb_test)
    overnight_comp = predict_ohlc_components(overnight_models, Xo_test)

    prev_close = spx_recent["Close"].shift(1).loc[y_test.index]

    base_ohlc = reconstruct_ohlc(prev_close=prev_close, component_preds=base_comp)
    overnight_ohlc = reconstruct_ohlc(prev_close=prev_close, component_preds=overnight_comp)

    base_component_metrics = evaluate_component_predictions(y_test, base_comp)
    overnight_component_metrics = evaluate_component_predictions(y_test, overnight_comp)

    base_ohlc_metrics = evaluate_reconstructed_ohlc(
        actual_ohlc=spx_test[["Open", "High", "Low", "Close"]],
        pred_ohlc=base_ohlc,
        prev_close=prev_close,
    )
    overnight_ohlc_metrics = evaluate_reconstructed_ohlc(
        actual_ohlc=spx_test[["Open", "High", "Low", "Close"]],
        pred_ohlc=overnight_ohlc,
        prev_close=prev_close,
    )

    report = {
        "overlap_rows": int(len(common)),
        "train_rows": int(len(Xb_train)),
        "test_rows": int(len(Xb_test)),
        "base_feature_count": int(X_base.shape[1]),
        "overnight_feature_count": int(X_overnight.shape[1]),
        "overnight_added_columns": [c for c in X_overnight.columns if c not in X_base.columns],
        "base_component_metrics": base_component_metrics,
        "overnight_component_metrics": overnight_component_metrics,
        "base_ohlc_metrics": base_ohlc_metrics,
        "overnight_ohlc_metrics": overnight_ohlc_metrics,
    }

    detail = pd.DataFrame(index=y_test.index)
    detail["prev_close"] = prev_close
    detail["actual_open"] = spx_test["Open"]
    detail["actual_high"] = spx_test["High"]
    detail["actual_low"] = spx_test["Low"]
    detail["actual_close"] = spx_test["Close"]

    for col in y.columns:
        detail[f"actual_{col}"] = y_test[col]
        detail[f"base_{col}"] = base_comp[col]
        detail[f"overnight_{col}"] = overnight_comp[col]

    for col in ["pred_open", "pred_high", "pred_low", "pred_close"]:
        detail[f"base_{col}"] = base_ohlc[col]
        detail[f"overnight_{col}"] = overnight_ohlc[col]

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)

    _save_json(report, REPORT_FILE)
    detail.to_csv(DETAIL_FILE, index=True)

    print("\nSaved recent overnight report to:", REPORT_FILE)
    print("Saved recent overnight detail CSV to:", DETAIL_FILE)

    print("\nBase OHLC metrics:")
    for k, v in base_ohlc_metrics.items():
        print(" ", k, v)

    print("\nOvernight OHLC metrics:")
    for k, v in overnight_ohlc_metrics.items():
        print(" ", k, v)


if __name__ == "__main__":
    main()
