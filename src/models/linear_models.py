"""
src/models/linear_models.py
============================
Task 15 (cont.) — Linear and regularised model wrappers.

Concrete implementations
------------------------
LogisticRegressionModel   — sklearn LogisticRegression (classification)
RidgeRegressionModel      — sklearn Ridge (regression)
LassoRegressionModel      — sklearn Lasso (regression)
ElasticNetModel           — sklearn ElasticNet (regression)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


def _assert_fitted(model: BaseModel) -> None:
    if not model._fitted:
        raise RuntimeError(f"Model '{model.name}' is not fitted yet.")


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

class LogisticRegressionModel(BaseModel):
    """Wrapper around sklearn LogisticRegression."""

    task = "classification"

    def __init__(
        self,
        name: str = "logistic_regression",
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, params)
        self._model = None
        default_params = dict(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1,
            random_state=42,
        )
        self.params = {**default_params, **(params or {})}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel":
        from sklearn.linear_model import LogisticRegression
        self._model = LogisticRegression(**self.params)
        self._model.fit(X, y)
        self._fitted = True
        logger.info("LogisticRegression '%s' fitted.", self.name)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        _assert_fitted(self)
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        _assert_fitted(self)
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> Optional[pd.Series]:
        if not self._fitted:
            return None
        coef = self._model.coef_
        if coef.shape[0] == 1:
            coef = coef[0]
        else:
            coef = np.abs(coef).mean(axis=0)
        names = getattr(self._model, "feature_names_in_", None)
        return pd.Series(np.abs(coef), index=names).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Ridge Regression
# ---------------------------------------------------------------------------

class RidgeRegressionModel(BaseModel):
    """Wrapper around sklearn Ridge."""

    task = "regression"

    def __init__(
        self,
        name: str = "ridge",
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(name, params)
        self._model = None
        default_params = dict(alpha=1.0)
        # Accept hyperparams as direct kwargs (e.g. alpha=1.0) or via params dict
        self.params = {**default_params, **(params or {}), **kwargs}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeRegressionModel":
        from sklearn.linear_model import Ridge
        self._model = Ridge(**self.params)
        self._model.fit(X, y)
        self._fitted = True
        logger.info("Ridge '%s' fitted.", self.name)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        _assert_fitted(self)
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> Optional[pd.Series]:
        if not self._fitted:
            return None
        names = getattr(self._model, "feature_names_in_", None)
        return pd.Series(
            np.abs(self._model.coef_), index=names
        ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Lasso Regression
# ---------------------------------------------------------------------------

class LassoRegressionModel(BaseModel):
    """Wrapper around sklearn Lasso."""

    task = "regression"

    def __init__(
        self,
        name: str = "lasso",
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(name, params)
        self._model = None
        default_params = dict(alpha=0.01, max_iter=5000)
        self.params = {**default_params, **(params or {}), **kwargs}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LassoRegressionModel":
        from sklearn.linear_model import Lasso
        self._model = Lasso(**self.params)
        self._model.fit(X, y)
        self._fitted = True
        logger.info("Lasso '%s' fitted.", self.name)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        _assert_fitted(self)
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> Optional[pd.Series]:
        if not self._fitted:
            return None
        names = getattr(self._model, "feature_names_in_", None)
        return pd.Series(
            np.abs(self._model.coef_), index=names
        ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# ElasticNet
# ---------------------------------------------------------------------------

class ElasticNetModel(BaseModel):
    """Wrapper around sklearn ElasticNet."""

    task = "regression"

    def __init__(
        self,
        name: str = "elastic_net",
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(name, params)
        self._model = None
        default_params = dict(alpha=0.01, l1_ratio=0.5, max_iter=5000)
        self.params = {**default_params, **(params or {}), **kwargs}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ElasticNetModel":
        from sklearn.linear_model import ElasticNet
        self._model = ElasticNet(**self.params)
        self._model.fit(X, y)
        self._fitted = True
        logger.info("ElasticNet '%s' fitted.", self.name)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        _assert_fitted(self)
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> Optional[pd.Series]:
        if not self._fitted:
            return None
        names = getattr(self._model, "feature_names_in_", None)
        return pd.Series(
            np.abs(self._model.coef_), index=names
        ).sort_values(ascending=False)
