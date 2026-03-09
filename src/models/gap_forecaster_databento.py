from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from lightgbm import LGBMRegressor


GAP_TARGET = "target_gap_ret"
DB_GAP_FEATURES = [
    "es_overnight_ret",
    "es_preopen_ret_last_60m",
    "es_preopen_ret_last_30m",
    "es_overnight_range_pct",
]


def default_gap_model_params() -> dict:
    return {
        "objective": "regression",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }


def build_gap_feature_matrix(
    base_features: pd.DataFrame,
    databento_features: pd.DataFrame,
) -> pd.DataFrame:
    missing = [c for c in DB_GAP_FEATURES if c not in databento_features.columns]
    if missing:
        raise ValueError(f"Missing Databento gap features: {missing}")
    db_small = databento_features[DB_GAP_FEATURES].copy()
    X = base_features.join(db_small, how="inner")
    X.index = pd.to_datetime(X.index)
    X = X.sort_index()
    return X


def align_gap_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    idx = X.index.intersection(y.index)
    X2 = X.loc[idx].sort_index()
    y2 = y.loc[idx].sort_index()
    joined = X2.join(y2.rename(GAP_TARGET), how="inner").dropna()
    return joined[X2.columns], joined[GAP_TARGET]


def train_gap_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict | None = None,
) -> LGBMRegressor:
    params = model_params or default_gap_model_params()
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model


def predict_gap(model: LGBMRegressor, X: pd.DataFrame) -> pd.Series:
    pred = model.predict(X)
    return pd.Series(pred, index=X.index, name=GAP_TARGET)


def save_gap_model(model: LGBMRegressor, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)


def load_gap_model(path: str | Path) -> LGBMRegressor:
    return joblib.load(Path(path))


def feature_importance_frame(model: LGBMRegressor, feature_names: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
