"""
src/models/trainer.py
======================
Task 16 — Walk-forward model trainer.

``Trainer`` orchestrates:
1.  Walk-forward splits via ``WalkForwardSplitter``.
2.  Fitting any ``BaseModel``-compatible estimator on each train fold.
3.  Generating out-of-sample predictions on each test fold.
4.  Collecting per-fold metrics and returning a unified OOS prediction Series.

Usage
-----
    trainer = Trainer(model=XGBoostModel(), splitter=WalkForwardSplitter())
    oos = trainer.run(X, y, target_col="next_high_bin_050")
    print(oos.metrics_df)
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import metrics as skm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.base_model import BaseModel
from src.targets.splitter import WalkForwardSplitter, SplitConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Holds all outputs from a Trainer.run() call."""

    target_col:   str
    oos_proba:    pd.Series          # positive-class probability, OOS
    oos_pred:     pd.Series          # binary predictions (threshold applied)
    oos_actual:   pd.Series          # true labels, OOS

    metrics_df:   pd.DataFrame       # per-fold metrics
    overall:      Dict[str, float]   # aggregate metrics across all folds

    fold_models:  List[BaseModel] = field(default_factory=list)
    feature_importance: Optional[pd.DataFrame] = None

    @property
    def n_folds(self) -> int:
        return len(self.metrics_df)

    def summary(self) -> str:
        ov = self.overall
        return (
            f"Target: {self.target_col} | Folds: {self.n_folds} | "
            f"AUC={ov.get('roc_auc', float('nan')):.4f} | "
            f"Acc={ov.get('accuracy', float('nan')):.4f} | "
            f"F1={ov.get('f1', float('nan')):.4f}"
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Walk-forward model trainer."""

    def __init__(
        self,
        model: BaseModel,
        splitter: Optional[WalkForwardSplitter] = None,
        threshold: float = 0.5,
        verbose: bool = True,
    ):
        self.model_proto = model
        self.splitter    = splitter or WalkForwardSplitter()
        self.threshold   = threshold
        self.verbose     = verbose

    # ------------------------------------------------------------------
    def run(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        target_col: str,
        fit_kwargs: Optional[Dict[str, Any]] = None,
    ) -> TrainResult:
        """
        Execute walk-forward training and return a TrainResult.

        Parameters
        ----------
        X          : Feature DataFrame (date-indexed).
        y          : Target DataFrame or Series.
        target_col : Column in y to predict.
        fit_kwargs : Extra kwargs forwarded to model.fit().
        """
        fit_kwargs = fit_kwargs or {}

        # Extract single target column
        if isinstance(y, pd.DataFrame):
            y_raw = y[target_col].dropna()
        else:
            y_raw = y.dropna()
            target_col = target_col or y.name

        # Cast: classification targets must be int; regression stays float
        if self.model_proto.task == "classification":
            y_series = y_raw.astype(int)
        else:
            y_series = y_raw.astype(float)

        # Align X and y
        common = X.index.intersection(y_series.index)
        X_al   = X.loc[common]
        y_al   = y_series.loc[common]

        splits   = self.splitter.split(X_al)
        n_folds  = len(splits)

        logger.info("Trainer: %d folds, model=%s, target=%s",
                    n_folds, self.model_proto.name, target_col)

        oos_proba_list:  List[pd.Series] = []
        oos_pred_list:   List[pd.Series] = []
        oos_actual_list: List[pd.Series] = []
        fold_rows:       List[Dict] = []
        fold_models:     List[BaseModel] = []
        fi_frames:       List[pd.Series] = []

        for fold_i, (tr_idx, te_idx) in enumerate(splits):
            t0 = time.perf_counter()

            # Deep-copy the prototype so each fold gets a fresh model
            model = copy.deepcopy(self.model_proto)

            X_train, y_train = X_al.iloc[tr_idx], y_al.iloc[tr_idx]
            X_test,  y_test  = X_al.iloc[te_idx], y_al.iloc[te_idx]

            # Fit
            model.fit(X_train, y_train, **fit_kwargs)

            # Predict
            if model.task == "classification":
                proba  = model.predict_proba(X_test)[:, 1]
                pred   = (proba >= self.threshold).astype(int)
            else:
                proba  = model.predict(X_test).astype(float)
                pred   = proba   # no threshold for regression

            proba_s  = pd.Series(proba, index=X_test.index, name=target_col)
            pred_s   = pd.Series(pred,  index=X_test.index, name=target_col)

            oos_proba_list.append(proba_s)
            oos_pred_list.append(pred_s)
            oos_actual_list.append(y_test)

            # Fold metrics
            row = self._fold_metrics(y_test.values, pred, proba, fold_i,
                                     X_train.index[0], X_train.index[-1],
                                     X_test.index[0],  X_test.index[-1],
                                     time.perf_counter() - t0)
            fold_rows.append(row)
            fold_models.append(model)

            if model.feature_importances_ is not None:
                fi_frames.append(model.feature_importances_)

            if self.verbose:
                logger.info(
                    "  Fold %02d/%02d  AUC=%.4f  Acc=%.4f  F1=%.4f  (%.1fs)",
                    fold_i + 1, n_folds,
                    row.get("roc_auc", float("nan")),
                    row.get("accuracy", float("nan")),
                    row.get("f1", float("nan")),
                    row["elapsed_s"],
                )

        # ── aggregate OOS ────────────────────────────────────────────────────
        oos_proba  = pd.concat(oos_proba_list)
        oos_pred   = pd.concat(oos_pred_list)
        oos_actual = pd.concat(oos_actual_list)

        metrics_df = pd.DataFrame(fold_rows)
        overall    = self._aggregate_metrics(oos_actual.values,
                                             oos_pred.values,
                                             oos_proba.values)

        # ── feature importance (mean across folds) ────────────────────────────
        fi_df: Optional[pd.DataFrame] = None
        if fi_frames:
            fi_df = pd.concat(fi_frames, axis=1).mean(axis=1).sort_values(
                ascending=False
            ).rename("mean_importance").to_frame()

        result = TrainResult(
            target_col=target_col,
            oos_proba=oos_proba,
            oos_pred=oos_pred,
            oos_actual=oos_actual,
            metrics_df=metrics_df,
            overall=overall,
            fold_models=fold_models,
            feature_importance=fi_df,
        )

        logger.info("Training complete: %s", result.summary())
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _fold_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        fold_i: int,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
        elapsed: float,
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "fold": fold_i,
            "train_start": train_start,
            "train_end":   train_end,
            "test_start":  test_start,
            "test_end":    test_end,
            "elapsed_s":   round(elapsed, 2),
            "n_test":      len(y_true),
        }
        try:
            row["accuracy"] = skm.accuracy_score(y_true, y_pred)
            row["f1"]       = skm.f1_score(y_true, y_pred, zero_division=0)
            row["precision"] = skm.precision_score(y_true, y_pred, zero_division=0)
            row["recall"]    = skm.recall_score(y_true, y_pred, zero_division=0)
        except Exception:
            pass
        try:
            row["roc_auc"] = skm.roc_auc_score(y_true, y_proba)
        except Exception:
            row["roc_auc"] = float("nan")
        return row

    @staticmethod
    def _aggregate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            out["accuracy"]  = skm.accuracy_score(y_true, y_pred)
            out["f1"]        = skm.f1_score(y_true, y_pred, zero_division=0)
            out["precision"] = skm.precision_score(y_true, y_pred, zero_division=0)
            out["recall"]    = skm.recall_score(y_true, y_pred, zero_division=0)
        except Exception:
            pass
        try:
            out["roc_auc"] = skm.roc_auc_score(y_true, y_proba)
        except Exception:
            out["roc_auc"] = float("nan")
        return out
