from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import pandas as pd

from src.models.ohlc_forecaster import OHLC_TARGET_COLUMNS


def build_naive_component_baseline(
    y_train: pd.DataFrame,
    y_test_index: pd.Index,
) -> pd.DataFrame:
    missing = [c for c in OHLC_TARGET_COLUMNS if c not in y_train.columns]
    if missing:
        raise ValueError(f"Missing OHLC target columns in y_train: {missing}")

    means = y_train[OHLC_TARGET_COLUMNS].mean()
    baseline = pd.DataFrame(
        {col: np.full(len(y_test_index), float(means[col])) for col in OHLC_TARGET_COLUMNS},
        index=y_test_index,
    )
    return baseline


def reconstruct_baseline_ohlc(
    prev_close: pd.Series,
    component_preds: pd.DataFrame,
) -> pd.DataFrame:
    required = set(OHLC_TARGET_COLUMNS)
    missing = required - set(component_preds.columns)
    if missing:
        raise ValueError(f"Missing component prediction columns: {sorted(missing)}")

    prev_close = prev_close.reindex(component_preds.index)

    pred_open = prev_close * (1.0 + component_preds["target_gap_ret"])
    pred_high = pred_open * (1.0 + component_preds["target_high_from_open"])
    pred_low = pred_open * (1.0 - component_preds["target_low_from_open"])
    pred_close = pred_open * (1.0 + component_preds["target_close_from_open"])

    out = pd.DataFrame(
        {
            "pred_open": pred_open,
            "pred_high": pred_high,
            "pred_low": pred_low,
            "pred_close": pred_close,
        },
        index=component_preds.index,
    )

    out["pred_high"] = pd.concat(
        [out["pred_high"], out["pred_open"], out["pred_close"]], axis=1
    ).max(axis=1)

    out["pred_low"] = pd.concat(
        [out["pred_low"], out["pred_open"], out["pred_close"]], axis=1
    ).min(axis=1)

    return out


def evaluate_ohlc_frame(
    actual_ohlc: pd.DataFrame,
    pred_ohlc: pd.DataFrame,
    prev_close: pd.Series,
) -> dict:
    actual = actual_ohlc.loc[pred_ohlc.index].copy()
    prev_close = prev_close.reindex(pred_ohlc.index)

    metrics = {}
    for actual_col, pred_col, name in [
        ("Open", "pred_open", "open"),
        ("High", "pred_high", "high"),
        ("Low", "pred_low", "low"),
        ("Close", "pred_close", "close"),
    ]:
        err = actual[actual_col] - pred_ohlc[pred_col]
        metrics[name] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
        }

    actual_dir = np.sign(actual["Close"] - prev_close)
    pred_dir = np.sign(pred_ohlc["pred_close"] - prev_close)
    metrics["close_direction_accuracy"] = float((actual_dir == pred_dir).mean())

    actual_range = actual["High"] - actual["Low"]
    pred_range = pred_ohlc["pred_high"] - pred_ohlc["pred_low"]
    metrics["range"] = {
        "actual_mean": float(actual_range.mean()),
        "pred_mean": float(pred_range.mean()),
        "mae": float(np.mean(np.abs(actual_range - pred_range))),
    }
    return metrics


def compare_metric_dicts(model_metrics: dict, baseline_metrics: dict) -> dict:
    out = {}
    shared = set(model_metrics).intersection(baseline_metrics)
    for key in sorted(shared):
        mv = model_metrics[key]
        bv = baseline_metrics[key]

        if isinstance(mv, dict) and isinstance(bv, dict):
            out[key] = {}
            for subk in sorted(set(mv).intersection(bv)):
                if isinstance(mv[subk], (int, float)) and isinstance(bv[subk], (int, float)):
                    out[key][subk] = float(bv[subk] - mv[subk])
        elif isinstance(mv, (int, float)) and isinstance(bv, (int, float)):
            out[key] = float(mv - bv)
    return out


def export_feature_importance_csvs(
    models: Dict[str, object],
    feature_names: pd.Index,
    output_dir: str | Path,
    top_n: int = 20,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for target, model in models.items():
        if not hasattr(model, "feature_importances_"):
            continue

        imp = pd.DataFrame(
            {
                "feature": list(feature_names),
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        imp.head(top_n).to_csv(output_path / f"{target}_feature_importance.csv", index=False)


def export_detailed_test_predictions(
    output_file: str | Path,
    actual_ohlc: pd.DataFrame,
    prev_close: pd.Series,
    y_test: pd.DataFrame,
    model_components: pd.DataFrame,
    baseline_components: pd.DataFrame,
    model_ohlc: pd.DataFrame,
    baseline_ohlc: pd.DataFrame,
) -> None:
    idx = y_test.index
    df = pd.DataFrame(index=idx)

    df["prev_close"] = prev_close.reindex(idx)
    df["actual_open"] = actual_ohlc.loc[idx, "Open"]
    df["actual_high"] = actual_ohlc.loc[idx, "High"]
    df["actual_low"] = actual_ohlc.loc[idx, "Low"]
    df["actual_close"] = actual_ohlc.loc[idx, "Close"]

    for col in OHLC_TARGET_COLUMNS:
        df[f"actual_{col}"] = y_test[col]
        df[f"model_{col}"] = model_components[col]
        df[f"baseline_{col}"] = baseline_components[col]

    for col in ["pred_open", "pred_high", "pred_low", "pred_close"]:
        df[f"model_{col}"] = model_ohlc[col]
        df[f"baseline_{col}"] = baseline_ohlc[col]

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=True)


def save_json(data: dict, output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
