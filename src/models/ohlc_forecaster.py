from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


OHLC_TARGET_COLUMNS = [
    "target_gap_ret",
    "target_high_from_open",
    "target_low_from_open",
    "target_close_from_open",
]


def _default_model_params() -> dict:
    return {
        "objective": "regression",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "n_jobs": -1,
    }


def _validate_targets(targets: pd.DataFrame) -> None:
    missing = [c for c in OHLC_TARGET_COLUMNS if c not in targets.columns]
    if missing:
        raise ValueError(f"Missing OHLC target columns: {missing}")


def align_features_and_ohlc_targets(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _validate_targets(targets)
    joined = features.join(targets[OHLC_TARGET_COLUMNS], how="inner")
    joined = joined.dropna(subset=OHLC_TARGET_COLUMNS)
    X = joined[features.columns].copy()
    y = joined[OHLC_TARGET_COLUMNS].copy()
    return X, y


def train_ohlc_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_params: dict | None = None,
) -> Dict[str, LGBMRegressor]:
    _validate_targets(y_train)
    params = _default_model_params()
    if model_params:
        params.update(model_params)

    models: Dict[str, LGBMRegressor] = {}
    for target in OHLC_TARGET_COLUMNS:
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train[target])
        models[target] = model
    return models


def predict_ohlc_components(
    models: Dict[str, LGBMRegressor],
    X: pd.DataFrame,
) -> pd.DataFrame:
    preds = {}
    for target in OHLC_TARGET_COLUMNS:
        if target not in models:
            raise ValueError(f"Missing trained model for {target}")
        preds[target] = models[target].predict(X)
    return pd.DataFrame(preds, index=X.index)


def reconstruct_ohlc(
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


def evaluate_component_predictions(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> dict:
    metrics = {}
    for col in OHLC_TARGET_COLUMNS:
        err = y_true[col] - y_pred[col]
        metrics[col] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
        }
    return metrics


def evaluate_reconstructed_ohlc(
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


def save_models(models: Dict[str, LGBMRegressor], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for target, model in models.items():
        joblib.dump(model, output_path / f"{target}.joblib")


def load_models(output_dir: str | Path) -> Dict[str, LGBMRegressor]:
    output_path = Path(output_dir)
    models = {}
    for target in OHLC_TARGET_COLUMNS:
        model_path = output_path / f"{target}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        models[target] = joblib.load(model_path)
    return models


def save_metrics(metrics: dict, output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
