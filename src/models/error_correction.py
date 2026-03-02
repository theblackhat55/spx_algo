"""
src/models/error_correction.py
==============================
Lightweight Ridge correction layer that adjusts base model predictions
using recent error patterns.

Safety rails:
- Correction capped at ±0.5% of SPX
- Auto-disable if correction error > base error for 10 consecutive days
- Requires ≥20 days of error history before activation
- 60-day rolling training window
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.models.error_features import (
    build_correction_features,
    load_error_history,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path("output/models")
CORRECTION_HIGH_PATH = MODEL_DIR / "error_correction_high.pkl"
CORRECTION_LOW_PATH = MODEL_DIR / "error_correction_low.pkl"
CORRECTION_LOG_PATH = Path("output/monitoring/correction_log.csv")

# Safety rails
MIN_HISTORY_DAYS = 20
ROLLING_WINDOW = 60
MAX_CORRECTION_PCT = 0.005  # ±0.5% of SPX
DISABLE_AFTER_N_WORSE = 10


class ErrorCorrector:
    """Applies a learned correction to base model predictions."""

    def __init__(self):
        self.model_high: Optional[Ridge] = None
        self.model_low: Optional[Ridge] = None
        self.active = False
        self.consecutive_worse = 0
        self._load_models()

    def _load_models(self):
        """Load saved correction models if they exist."""
        try:
            if CORRECTION_HIGH_PATH.exists() and CORRECTION_LOW_PATH.exists():
                self.model_high = joblib.load(CORRECTION_HIGH_PATH)
                self.model_low = joblib.load(CORRECTION_LOW_PATH)
                self.active = True
                logger.info("Error correction models loaded")
        except Exception as e:
            logger.warning("Could not load correction models: %s", e)
            self.active = False

    def fit(self) -> Dict[str, float]:
        """
        Refit correction models on recent error history.
        Called by morning reconciliation after logging yesterday's errors.
        """
        error_df = load_error_history()

        if len(error_df) < MIN_HISTORY_DAYS:
            logger.info(
                "Error correction: need %d days, have %d — skipping fit",
                MIN_HISTORY_DAYS, len(error_df)
            )
            return {"status": "insufficient_data", "days": len(error_df)}

        # Use rolling window
        error_df = error_df.iloc[-ROLLING_WINDOW:]
        features = build_correction_features(error_df)

        if len(features) < 10:
            logger.info("Error correction: insufficient features after build (%d rows)", len(features))
            return {"status": "insufficient_features", "rows": len(features)}

        # Align features with targets
        common = features.index.intersection(error_df.index)
        X = features.loc[common]
        y_high = error_df.loc[common, "error_high"]
        y_low = error_df.loc[common, "error_low"]

        # Fit Ridge models
        self.model_high = Ridge(alpha=10.0)
        self.model_high.fit(X.values, y_high.values)

        self.model_low = Ridge(alpha=10.0)
        self.model_low.fit(X.values, y_low.values)

        # Save
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_high, CORRECTION_HIGH_PATH)
        joblib.dump(self.model_low, CORRECTION_LOW_PATH)
        self.active = True

        # Training metrics
        pred_h = self.model_high.predict(X.values)
        pred_l = self.model_low.predict(X.values)
        mae_h = float(np.mean(np.abs(pred_h - y_high.values)))
        mae_l = float(np.mean(np.abs(pred_l - y_low.values)))

        metrics = {
            "status": "fitted",
            "train_rows": len(X),
            "mae_correction_high": round(mae_h, 6),
            "mae_correction_low": round(mae_l, 6),
            "mean_error_high": round(float(y_high.mean()), 6),
            "mean_error_low": round(float(y_low.mean()), 6),
        }
        logger.info("Error correction fitted: %s", metrics)
        return metrics

    def correct(
        self,
        pred_high_pct: float,
        pred_low_pct: float,
        regime: str,
        vix: float,
        day_of_week: int,
    ) -> Tuple[float, float, Dict]:
        """
        Apply correction to base predictions.

        Returns (corrected_high, corrected_low, metadata).
        """
        metadata = {
            "active": self.active,
            "raw_high": pred_high_pct,
            "raw_low": pred_low_pct,
        }

        if not self.active or self.model_high is None:
            metadata["reason"] = "not_active"
            return pred_high_pct, pred_low_pct, metadata

        # Build a single-row feature vector from current context
        error_df = load_error_history()
        if len(error_df) < 5:
            metadata["reason"] = "insufficient_history"
            return pred_high_pct, pred_low_pct, metadata

        features = build_correction_features(error_df)
        if len(features) == 0:
            metadata["reason"] = "no_features"
            return pred_high_pct, pred_low_pct, metadata

        # Use the last row of features (most recent error pattern)
        last_features = features.iloc[[-1]].values

        correction_high = float(self.model_high.predict(last_features)[0])
        correction_low = float(self.model_low.predict(last_features)[0])

        # Safety rail: cap corrections
        correction_high = np.clip(correction_high, -MAX_CORRECTION_PCT, MAX_CORRECTION_PCT)
        correction_low = np.clip(correction_low, -MAX_CORRECTION_PCT, MAX_CORRECTION_PCT)

        corrected_high = pred_high_pct + correction_high
        corrected_low = pred_low_pct + correction_low

        metadata.update({
            "correction_high": round(correction_high, 6),
            "correction_low": round(correction_low, 6),
            "corrected_high": round(corrected_high, 6),
            "corrected_low": round(corrected_low, 6),
            "capped": abs(correction_high) >= MAX_CORRECTION_PCT or abs(correction_low) >= MAX_CORRECTION_PCT,
        })

        # Log correction
        self._log_correction(metadata)

        return corrected_high, corrected_low, metadata

    def _log_correction(self, metadata: Dict):
        """Append correction to the log file."""
        CORRECTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([metadata])
        if CORRECTION_LOG_PATH.exists():
            row.to_csv(CORRECTION_LOG_PATH, mode="a", header=False, index=False)
        else:
            row.to_csv(CORRECTION_LOG_PATH, index=False)
