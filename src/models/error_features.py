"""
src/models/error_features.py
============================
Build features for the error correction model from recent prediction errors.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ERROR_LOG_PATH = Path("output/monitoring/error_history.csv")


def load_error_history(path: Path = ERROR_LOG_PATH) -> pd.DataFrame:
    """Load the error history CSV."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df.sort_index()


def save_error_history(df: pd.DataFrame, path: Path = ERROR_LOG_PATH):
    """Save the error history CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.info("Error history saved: %d rows → %s", len(df), path)


def record_error(
    date: str,
    pred_high_pct: float,
    pred_low_pct: float,
    actual_high_pct: float,
    actual_low_pct: float,
    regime: str,
    vix: float,
    day_of_week: int,
    path: Path = ERROR_LOG_PATH,
):
    """Append one day's prediction error to the history."""
    df = load_error_history(path)

    new_row = pd.DataFrame({
        "date": [pd.Timestamp(date)],
        "pred_high_pct": [pred_high_pct],
        "pred_low_pct": [pred_low_pct],
        "actual_high_pct": [actual_high_pct],
        "actual_low_pct": [actual_low_pct],
        "error_high": [actual_high_pct - pred_high_pct],
        "error_low": [actual_low_pct - pred_low_pct],
        "abs_error_high": [abs(actual_high_pct - pred_high_pct)],
        "abs_error_low": [abs(actual_low_pct - pred_low_pct)],
        "regime": [regime],
        "vix": [vix],
        "day_of_week": [day_of_week],
    }).set_index("date")

    df = pd.concat([df, new_row])
    df = df[~df.index.duplicated(keep="last")]
    save_error_history(df, path)
    return df


def build_correction_features(error_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for the correction model from error history.

    Features per target (high/low):
    - mean_error_5d: rolling 5-day signed error mean (bias direction)
    - mean_error_20d: rolling 20-day signed error mean (longer-term bias)
    - std_error_5d: recent error volatility
    - error_trend: slope of last 5 errors (is bias growing?)
    - regime_GREEN/YELLOW/RED: one-hot regime
    - vix_bucket: low (<15), mid (15-25), high (>25)
    - day_of_week_0..4: one-hot day of week
    - direction_accuracy_10d: % of last 10 days where error sign was correct
    """
    if len(error_df) < 5:
        return pd.DataFrame()

    features = pd.DataFrame(index=error_df.index)

    for target in ["high", "low"]:
        err_col = f"error_{target}"
        abs_col = f"abs_error_{target}"

        # Rolling stats
        features[f"mean_error_5d_{target}"] = error_df[err_col].rolling(5, min_periods=3).mean()
        features[f"mean_error_20d_{target}"] = error_df[err_col].rolling(20, min_periods=5).mean()
        features[f"std_error_5d_{target}"] = error_df[err_col].rolling(5, min_periods=3).std()

        # Error trend (linear slope over last 5)
        def _slope(s):
            if len(s) < 3:
                return 0.0
            x = np.arange(len(s))
            return np.polyfit(x, s.values, 1)[0]

        features[f"error_trend_{target}"] = (
            error_df[err_col].rolling(5, min_periods=3).apply(_slope, raw=False)
        )

        # Direction accuracy (did model predict correct side?)
        features[f"overpredict_rate_10d_{target}"] = (
            (error_df[err_col] > 0).astype(float).rolling(10, min_periods=5).mean()
        )

    # Regime one-hot
    for r in ["GREEN", "YELLOW", "RED"]:
        features[f"regime_{r}"] = (error_df["regime"] == r).astype(float)

    # VIX bucket
    features["vix_low"] = (error_df["vix"] < 15).astype(float)
    features["vix_mid"] = ((error_df["vix"] >= 15) & (error_df["vix"] <= 25)).astype(float)
    features["vix_high"] = (error_df["vix"] > 25).astype(float)

    # Day of week one-hot
    for d in range(5):
        features[f"dow_{d}"] = (error_df["day_of_week"] == d).astype(float)

    return features.dropna()
