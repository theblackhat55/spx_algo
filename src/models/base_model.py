"""
src/models/base_model.py
========================
Task 14 — Abstract base class for all SPX predictors.

Every concrete model (XGBoost, LightGBM, Random Forest, Logistic Regression,
Ridge, etc.) inherits from ``BaseModel`` and must implement:

    fit(X_train, y_train)   → self
    predict(X)              → np.ndarray        (class labels or regression values)
    predict_proba(X)        → np.ndarray        (N×C probability matrix; classification only)
    feature_importances_    → pd.Series | None  (name → importance)

The base class provides:
    * Serialisation  (save / load via joblib)
    * Threshold calibration helpers
    * A unified score() that calls sklearn metrics
    * Logging helpers
"""
from __future__ import annotations

import abc
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseModel(abc.ABC):
    """Abstract base for all SPX prediction models."""

    # Subclasses set this to 'classification' or 'regression'
    task: str = "classification"

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name   = name
        self.params = params or {}
        self._fitted = False
        self._threshold: float = 0.5   # default decision threshold

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Train the model.  Must set self._fitted = True."""

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted labels (classification) or values (regression)."""

    # ------------------------------------------------------------------
    # Optional interface (provide defaults)
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability matrix.  Default: NotImplemented for regression."""
        if self.task == "regression":
            raise NotImplementedError("predict_proba not defined for regression models.")
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement predict_proba."
        )

    @property
    def feature_importances_(self) -> Optional[pd.Series]:
        """Return feature importances (name → float).  None if unavailable."""
        return None

    # ------------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------------

    def set_threshold(self, threshold: float) -> "BaseModel":
        """Override the default 0.5 decision threshold."""
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self._threshold = threshold
        return self

    def predict_with_threshold(self, X: pd.DataFrame) -> np.ndarray:
        """Binary predict using self._threshold on positive-class probability."""
        proba = self.predict_proba(X)
        pos_col = proba[:, 1] if proba.ndim == 2 else proba
        return (pos_col >= self._threshold).astype(int)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path) -> Path:
        """Serialise model to ``path`` (joblib preferred, pickle fallback)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import joblib
            joblib.dump(self, path)
        except ImportError:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        logger.info("Saved model '%s' → %s", self.name, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load a previously serialised model."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        try:
            import joblib
            model = joblib.load(path)
        except ImportError:
            with open(path, "rb") as f:
                model = pickle.load(f)
        logger.info("Loaded model '%s' from %s", model.name, path)
        return model

    # ------------------------------------------------------------------
    # Score helpers
    # ------------------------------------------------------------------

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = "auto",
    ) -> float:
        """
        Compute a scalar score.

        metric='auto' → accuracy for classification, R² for regression.
        Other valid values: 'accuracy', 'roc_auc', 'f1', 'r2', 'mae', 'rmse'.
        """
        from sklearn import metrics as skm

        if metric == "auto":
            metric = "accuracy" if self.task == "classification" else "r2"

        if self.task == "classification":
            if metric == "roc_auc":
                proba = self.predict_proba(X)[:, 1]
                return skm.roc_auc_score(y, proba)
            preds = self.predict(X)
            if metric == "accuracy":
                return skm.accuracy_score(y, preds)
            if metric == "f1":
                return skm.f1_score(y, preds, zero_division=0)
        else:
            preds = self.predict(X)
            if metric == "r2":
                return skm.r2_score(y, preds)
            if metric in ("mae", "mean_absolute_error"):
                return skm.mean_absolute_error(y, preds)
            if metric in ("rmse", "root_mean_squared_error"):
                return float(np.sqrt(skm.mean_squared_error(y, preds)))

        raise ValueError(f"Unknown metric '{metric}' for task '{self.task}'.")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"{self.__class__.__name__}(name={self.name!r}, task={self.task!r}, {status})"
