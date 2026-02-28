"""
src/calibration/conformal.py
=============================
Instruction 3 — Conformal Prediction Intervals for regression models.

Wraps a fitted regression BaseModel to produce distribution-free
prediction intervals via the MAPIE library (EnbPI method for time series).

Output
------
For each prediction date, returns:
    predicted       : point prediction (float)
    lower_68        : lower bound of 68% prediction interval
    upper_68        : upper bound of 68% prediction interval
    lower_90        : lower bound of 90% prediction interval
    upper_90        : upper bound of 90% prediction interval

Iron condor application
-----------------------
short call strike ≈ upper_90 (99th pct on busy days)
short put  strike ≈ lower_90

These intervals are the primary inputs to the Phase 4 backtest engine's
strike placement algorithm.

Dependencies
------------
mapie >= 0.8  : pip install mapie

Falls back to symmetric residual-based intervals if MAPIE not available.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MAPIE conformal wrapper
# ---------------------------------------------------------------------------

class ConformalPredictor:
    """
    Wraps a regression ``BaseModel`` to produce prediction intervals.

    Two backends:
    1. MAPIE EnbPI   — time-series-aware, uses online residuals
    2. Residual-ICP  — split-conformal fallback (no MAPIE required)

    Parameters
    ----------
    model       : A fitted regression BaseModel.
    alpha_list  : Coverage levels to compute (default [0.68, 0.90]).
    use_mapie   : Try MAPIE first; fall back to residual-ICP if unavailable.
    """

    def __init__(
        self,
        model: BaseModel,
        alpha_list: Optional[list] = None,
        use_mapie: bool = True,
    ):
        if model.task != "regression":
            raise ValueError(
                f"ConformalPredictor requires a regression model, "
                f"got task='{model.task}'."
            )
        self.model      = model
        self.alpha_list = alpha_list or [0.68, 0.90]
        self.use_mapie  = use_mapie

        self._residuals: Optional[np.ndarray] = None
        self._mapie_model = None
        self._calibrated  = False

    # ------------------------------------------------------------------
    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> "ConformalPredictor":
        """
        Fit the conformal calibrator on a held-out calibration set.

        For MAPIE: wraps the model's sklearn-compatible interface.
        For residual-ICP: simply stores the absolute residuals.
        """
        if self.use_mapie:
            try:
                from mapie.regression import MapieRegressor
                from mapie.conformity_scores import AbsoluteConformityScore

                # Wrap BaseModel in a minimal sklearn adapter
                adapter = _SklearnAdapter(self.model)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mapie = MapieRegressor(
                        estimator=adapter,
                        method="base",
                        cv="prefit",   # model already fitted
                        conformity_score=AbsoluteConformityScore(),
                    )
                    mapie.fit(X_cal.values, y_cal.values)

                self._mapie_model = mapie
                self._calibrated  = True
                logger.info("ConformalPredictor calibrated via MAPIE on %d samples.", len(y_cal))
                return self

            except ImportError:
                logger.warning("MAPIE not installed; falling back to residual-ICP.")
            except Exception as e:
                logger.warning("MAPIE failed (%s); falling back to residual-ICP.", e)

        # Residual ICP fallback
        preds = self.model.predict(X_cal).astype(float)
        self._residuals = np.abs(preds - y_cal.values.astype(float))
        self._calibrated = True
        logger.info("ConformalPredictor calibrated via residual-ICP on %d samples.", len(y_cal))
        return self

    # ------------------------------------------------------------------
    def predict_interval(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with columns:
            predicted, lower_68, upper_68, lower_90, upper_90

        Index matches X.index.
        """
        if not self._calibrated:
            raise RuntimeError(
                "ConformalPredictor is not calibrated. Call calibrate() first."
            )

        point = self.model.predict(X).astype(float)
        result = pd.DataFrame({"predicted": point}, index=X.index)

        for alpha in self.alpha_list:
            label = int(round(alpha * 100))   # 68, 90
            lo, hi = self._interval(X, point, alpha)
            result[f"lower_{label}"] = lo
            result[f"upper_{label}"] = hi

        return result

    # ------------------------------------------------------------------
    def _interval(
        self,
        X: pd.DataFrame,
        point: np.ndarray,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) arrays for a given coverage level."""

        if self._mapie_model is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, intervals = self._mapie_model.predict(
                    X.values,
                    alpha=1 - alpha,
                    ensemble=False,
                )
            lo = intervals[:, 0, 0]
            hi = intervals[:, 1, 0]
            return lo, hi

        # Residual-ICP fallback: use empirical quantile of calibration residuals
        q = np.quantile(self._residuals, alpha)
        return point - q, point + q

    # ------------------------------------------------------------------
    def interval_width(self, alpha: float = 0.90) -> float:
        """Return mean interval width (requires prior predict_interval call)."""
        if self._residuals is not None:
            return float(2 * np.quantile(self._residuals, alpha))
        raise RuntimeError("No residual-ICP data; use predict_interval results.")

    def __repr__(self) -> str:
        backend = "MAPIE" if self._mapie_model else "residual-ICP"
        return (
            f"ConformalPredictor(model={self.model.name!r}, "
            f"backend={backend!r}, calibrated={self._calibrated})"
        )


# ---------------------------------------------------------------------------
# Sklearn adapter so MAPIE can wrap BaseModel
# ---------------------------------------------------------------------------

class _SklearnAdapter:
    """
    Minimal sklearn-compatible wrapper so that MAPIE can call
    fit/predict on our BaseModel without issues.

    Note: MAPIE with cv='prefit' only calls predict(), never fit().
    """

    def __init__(self, model: BaseModel):
        self.model = model

    def fit(self, X, y=None):
        # cv='prefit' → MAPIE should not call fit, but implement for safety
        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            import pandas as pd
            # MAPIE strips column names; reconstruct minimal DataFrame
            X = pd.DataFrame(X)
        return self.model.predict(X).astype(float)

    def get_params(self, deep: bool = True) -> dict:
        return {"model": self.model}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ---------------------------------------------------------------------------
# Convenience: build conformal ensemble output
# ---------------------------------------------------------------------------

def build_conformal_levels(
    oos_df: pd.DataFrame,
    spx_close: pd.Series,
    high_col: str = "predicted_high_pct",
    low_col:  str = "predicted_low_pct",
    high_upper_col: str = "upper_90_high",
    low_lower_col:  str = "lower_90_low",
) -> pd.DataFrame:
    """
    Convert percentage-deviation predictions + intervals into absolute
    SPX price levels for direct use in the backtest engine.

    Parameters
    ----------
    oos_df      : Output of ConformalPredictor.predict_interval().
    spx_close   : Prior-day close prices (shifted +1 to align with OOS dates).

    Returns
    -------
    DataFrame with columns:
        pred_high, pred_low,
        call_strike (= upper_90 of predicted high),
        put_strike  (= lower_90 of predicted low)
    """
    close = spx_close.reindex(oos_df.index)

    out = pd.DataFrame(index=oos_df.index)
    if high_col in oos_df.columns:
        out["pred_high"] = close * (1 + oos_df[high_col])
    if low_col in oos_df.columns:
        out["pred_low"]  = close * (1 + oos_df[low_col])
    if high_upper_col in oos_df.columns:
        out["call_strike"] = close * (1 + oos_df[high_upper_col])
    if low_lower_col in oos_df.columns:
        out["put_strike"]  = close * (1 + oos_df[low_lower_col])

    return out
