"""
src/models/tree_models.py
==========================
Task 15 — Gradient-boosted tree and Random Forest wrappers.

Concrete implementations
------------------------
XGBoostModel        — XGBClassifier / XGBRegressor
LightGBMModel       — LGBMClassifier / LGBMRegressor
RandomForestModel   — RandomForestClassifier / RandomForestRegressor
ExtraTreesModel     — ExtraTreesClassifier / ExtraTreesRegressor

All classes gracefully raise ``ImportError`` if the underlying library is
not installed, so tests can skip them with pytest.importorskip.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_fitted(model: BaseModel) -> None:
    if not model._fitted:
        raise RuntimeError(f"Model '{model.name}' is not fitted yet.")


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostModel(BaseModel):
    """Thin wrapper around xgboost.XGBClassifier / XGBRegressor."""

    def __init__(
        self,
        name: str = "xgboost",
        task: str = "classification",
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, params)
        self.task = task
        self._model = None

        try:
            import xgboost as xgb  # noqa: F401
        except ImportError as e:
            raise ImportError("xgboost not installed.  Run: pip install xgboost") from e

        default_params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            n_jobs=-1,
            random_state=42,
            eval_metric="logloss" if task == "classification" else "rmse",
        )
        self.params = {**default_params, **(params or {})}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[list] = None,
        early_stopping_rounds: int = 50,
    ) -> "XGBoostModel":
        import xgboost as xgb

        if self.task == "classification":
            self._model = xgb.XGBClassifier(**self.params)
        else:
            self._model = xgb.XGBRegressor(**self.params)

        fit_kwargs: Dict[str, Any] = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            fit_kwargs["verbose"] = False

        self._model.fit(X, y, **fit_kwargs)
        self._fitted = True
        logger.info("XGBoost '%s' fitted on %d rows, %d features.", self.name, len(X), X.shape[1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        _assert_fitted(self)
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError
        _assert_fitted(self)
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> Optional[pd.Series]:
        if not self._fitted:
            return None
        return pd.Series(
            self._model.feature_importances_,
            index=self._model.feature_names_in_
            if hasattr(self._model, "feature_names_in_") else None,
        ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LightGBMModel(BaseModel):
    """Thin wrapper around lightgbm.LGBMClassifier / LGBMRegressor."""

    def __init__(
        self,
        name: str = "lightgbm",
        task: str = "classification",
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, params)
        self.task = task
        self._model = None

        try:
            import lightgbm as lgb  # noqa: F401
        except ImportError as e:
            raise ImportError("lightgbm not installed.  Run: pip install lightgbm") from e

        default_params = dict(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        self.params = {**default_params, **(params or {})}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[list] = None,
        early_stopping_rounds: int = 50,
    ) -> "LightGBMModel":
        import lightgbm as lgb

        if self.task == "classification":
            self._model = lgb.LGBMClassifier(**self.params)
        else:
            self._model = lgb.LGBMRegressor(**self.params)

        fit_kwargs: Dict[str, Any] = {}
        if eval_set:
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False),
                         lgb.log_evaluation(-1)]
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["callbacks"] = callbacks

        self._model.fit(X, y, **fit_kwargs)
        self._fitted = True
        logger.info("LightGBM '%s' fitted on %d rows.", self.name, len(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        _assert_fitted(self)
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError
        _assert_fitted(self)
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> Optional[pd.Series]:
        if not self._fitted:
            return None
        return pd.Series(
            self._model.feature_importances_,
            index=self._model.feature_names_in_
            if hasattr(self._model, "feature_names_in_") else None,
        ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestModel(BaseModel):
    """Wrapper around sklearn RandomForestClassifier / RandomForestRegressor."""

    def __init__(
        self,
        name: str = "random_forest",
        task: str = "classification",
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, params)
        self.task = task
        self._model = None

        default_params = dict(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        self.params = {**default_params, **(params or {})}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        from sklearn.ensemble import (
            RandomForestClassifier, RandomForestRegressor
        )
        if self.task == "classification":
            self._model = RandomForestClassifier(**self.params)
        else:
            self._model = RandomForestRegressor(**self.params)

        self._model.fit(X, y)
        self._fitted = True
        logger.info("RandomForest '%s' fitted on %d rows.", self.name, len(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        _assert_fitted(self)
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError
        _assert_fitted(self)
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> Optional[pd.Series]:
        if not self._fitted:
            return None
        fi = self._model.feature_importances_
        names = getattr(self._model, "feature_names_in_", None)
        return pd.Series(fi, index=names).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Extra Trees
# ---------------------------------------------------------------------------

class ExtraTreesModel(BaseModel):
    """Wrapper around sklearn ExtraTreesClassifier / ExtraTreesRegressor."""

    def __init__(
        self,
        name: str = "extra_trees",
        task: str = "classification",
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, params)
        self.task = task
        self._model = None

        default_params = dict(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        self.params = {**default_params, **(params or {})}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ExtraTreesModel":
        from sklearn.ensemble import (
            ExtraTreesClassifier, ExtraTreesRegressor
        )
        if self.task == "classification":
            self._model = ExtraTreesClassifier(**self.params)
        else:
            self._model = ExtraTreesRegressor(**self.params)

        self._model.fit(X, y)
        self._fitted = True
        logger.info("ExtraTrees '%s' fitted on %d rows.", self.name, len(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        _assert_fitted(self)
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError
        _assert_fitted(self)
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> Optional[pd.Series]:
        if not self._fitted:
            return None
        fi = self._model.feature_importances_
        names = getattr(self._model, "feature_names_in_", None)
        return pd.Series(fi, index=names).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

class CatBoostModel(BaseModel):
    """Wrapper around catboost.CatBoostClassifier / CatBoostRegressor."""

    def __init__(
        self,
        name: str = "catboost",
        task: str = "classification",
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, params)
        self.task = task
        self._model = None

        try:
            import catboost  # noqa: F401
        except ImportError as e:
            raise ImportError("catboost not installed.  Run: pip install catboost") from e

        default_params = dict(
            iterations=500,
            learning_rate=0.05,
            depth=4,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
        )
        if task == "regression":
            default_params["loss_function"] = "Huber:delta=1.35"
        self.params = {**default_params, **(params or {})}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[list] = None,
        early_stopping_rounds: int = 50,
    ) -> "CatBoostModel":
        from catboost import CatBoostClassifier, CatBoostRegressor

        if self.task == "classification":
            self._model = CatBoostClassifier(**self.params)
        else:
            self._model = CatBoostRegressor(**self.params)

        fit_kwargs: Dict[str, Any] = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        self._model.fit(X, y, **fit_kwargs)
        self._fitted = True
        logger.info("CatBoost '%s' fitted on %d rows.", self.name, len(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        _assert_fitted(self)
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError
        _assert_fitted(self)
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> Optional[pd.Series]:
        if not self._fitted:
            return None
        fi     = self._model.get_feature_importance()
        names  = self._model.feature_names_
        return pd.Series(fi, index=names).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# HuberXGBoost  (XGBoost regression with Huber loss)
# ---------------------------------------------------------------------------

class HuberXGBoostModel(XGBoostModel):
    """XGBRegressor with pseudo-Huber loss (robust to outlier days)."""

    def __init__(
        self,
        name: str = "huber_xgboost",
        params: Optional[Dict[str, Any]] = None,
    ):
        # Override: always regression, Huber eval metric
        base_params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            n_jobs=-1,
            random_state=42,
            eval_metric="mae",
            objective="reg:pseudohubererror",
        )
        base_params.update(params or {})
        super().__init__(name=name, task="regression", params=base_params)


# ---------------------------------------------------------------------------
# HuberLightGBM  (LightGBM regression with Huber loss)
# ---------------------------------------------------------------------------

class HuberLightGBMModel(LightGBMModel):
    """LGBMRegressor with Huber loss (alpha=1.35)."""

    def __init__(
        self,
        name: str = "huber_lightgbm",
        params: Optional[Dict[str, Any]] = None,
    ):
        base_params = dict(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
            objective="huber",
            alpha=1.35,     # Huber delta
        )
        base_params.update(params or {})
        super().__init__(name=name, task="regression", params=base_params)
