"""
src/calibration/isotonic.py
============================
Task 18 — Probability calibration.

Provides
--------
IsotonicCalibrator    — Wraps sklearn IsotonicRegression for post-hoc calibration.
PlattCalibrator       — Wraps sklearn LogisticRegression (Platt scaling).
CalibratedPredictor   — Composes a BaseModel + any calibrator.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibrator interface
# ---------------------------------------------------------------------------

class _BaseCalibrator:
    """Minimal interface for all calibrators."""

    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> "_BaseCalibrator":
        raise NotImplementedError

    def transform(self, y_proba: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(
        self, y_proba: np.ndarray, y_true: np.ndarray
    ) -> np.ndarray:
        return self.fit(y_proba, y_true).transform(y_proba)


# ---------------------------------------------------------------------------
# Isotonic calibration
# ---------------------------------------------------------------------------

class IsotonicCalibrator(_BaseCalibrator):
    """Post-hoc probability calibration via isotonic regression."""

    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(y_proba.ravel(), y_true.ravel())
        self._fitted = True
        logger.debug("IsotonicCalibrator fitted on %d samples.", len(y_true))
        return self

    def transform(self, y_proba: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator is not fitted.")
        return self._iso.predict(y_proba.ravel())


# ---------------------------------------------------------------------------
# Platt (logistic) calibration
# ---------------------------------------------------------------------------

class PlattCalibrator(_BaseCalibrator):
    """Post-hoc probability calibration via Platt scaling (logistic regression)."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self._lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self._fitted = False

    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        X = y_proba.ravel().reshape(-1, 1)
        self._lr.fit(X, y_true.ravel())
        self._fitted = True
        logger.debug("PlattCalibrator fitted on %d samples.", len(y_true))
        return self

    def transform(self, y_proba: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PlattCalibrator is not fitted.")
        X = y_proba.ravel().reshape(-1, 1)
        return self._lr.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Composed predictor
# ---------------------------------------------------------------------------

class CalibratedPredictor:
    """
    Wraps a fitted ``BaseModel`` + a calibrator.

    Calibrator must be fit on a *held-out* calibration set (not the
    training data) to avoid over-fitting the calibration.

    Parameters
    ----------
    model       : A fitted BaseModel instance.
    calibrator  : An isotonic or Platt calibrator.
    threshold   : Decision threshold applied to calibrated probabilities.
    """

    def __init__(
        self,
        model: BaseModel,
        calibrator: Optional[_BaseCalibrator] = None,
        threshold: float = 0.5,
    ):
        self.model      = model
        self.calibrator = calibrator
        self.threshold  = threshold
        self._cal_fitted = calibrator is None   # no cal → pass-through

    # ------------------------------------------------------------------
    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: np.ndarray,
        calibrator_type: str = "isotonic",
    ) -> "CalibratedPredictor":
        """
        Fit the calibrator on a held-out calibration set.

        Parameters
        ----------
        calibrator_type : 'isotonic' | 'platt'
        """
        raw_proba = self.model.predict_proba(X_cal)[:, 1]

        if self.calibrator is None:
            if calibrator_type == "platt":
                self.calibrator = PlattCalibrator()
            else:
                self.calibrator = IsotonicCalibrator()

        self.calibrator.fit(raw_proba, y_cal)
        self._cal_fitted = True
        logger.info("Calibration fitted (%s) on %d samples.", calibrator_type, len(y_cal))
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated positive-class probabilities."""
        raw = self.model.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            if not self._cal_fitted:
                raise RuntimeError(
                    "Calibrator is set but not fitted. "
                    "Call calibrate(X_cal, y_cal) before predict_proba."
                )
            return self.calibrator.transform(raw)
        return raw

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions using self.threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        cal_name = type(self.calibrator).__name__ if self.calibrator else "None"
        return (
            f"CalibratedPredictor(model={self.model.name!r}, "
            f"calibrator={cal_name}, threshold={self.threshold})"
        )
