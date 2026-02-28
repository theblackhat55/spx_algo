"""
src/validation/hyperparam_sweep.py
=====================================
Task 37 — Walk-Forward Hyperparameter Sweep with Optuna.

Uses Bayesian optimisation (Optuna) to tune ML hyperparameters within the
walk-forward framework, measuring both aggregate performance AND fold-to-fold
stability.

Classes
-------
HyperparamSweep     Core sweep runner (model-agnostic).

Functions
---------
stability_analysis  Compute per-combo fold variance.
recommend           Pick the best stable combo.
sweep_xgboost       Pre-configured XGBoost sweep.
sweep_lightgbm      Pre-configured LightGBM sweep.
sweep_catboost      Pre-configured CatBoost sweep.
"""
from __future__ import annotations

import logging
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

REPORT_DIR = Path("output/reports")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


METRIC_FNS: Dict[str, Callable] = {"mae": _mae, "rmse": _rmse}


# ---------------------------------------------------------------------------
# HyperparamSweep
# ---------------------------------------------------------------------------

class HyperparamSweep:
    """
    Walk-forward Bayesian hyperparameter sweep.

    Parameters
    ----------
    model_class     : A class with fit(X, y) / predict(X) interface.
    splitter        : Object with split(X) → list of (train_idx, test_idx).
    metric          : "mae" or "rmse".
    n_trials        : Max Optuna trials.
    n_jobs          : Parallel jobs (-1 = all cores).
    seed            : Random seed for reproducibility.
    """

    def __init__(
        self,
        model_class,
        splitter,
        metric:   str = "mae",
        n_trials: int = 100,
        n_jobs:   int = 1,
        seed:     int = 42,
    ):
        self.model_class = model_class
        self.splitter    = splitter
        self.metric      = metric
        self.n_trials    = n_trials
        self.n_jobs      = n_jobs
        self.seed        = seed

    # ------------------------------------------------------------------

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_fn: Callable,            # maps optuna.Trial → dict of params
    ) -> pd.DataFrame:
        """
        Run the walk-forward sweep.

        Parameters
        ----------
        param_fn : A function (trial) → dict mapping hyperparameter names
                   to values, using trial.suggest_* calls.

        Returns a DataFrame with columns:
          trial_id, params, fold_scores (list), mean_score, std_score.
        Sorted by mean_score ascending (lower is better).
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("optuna required: pip install optuna>=3.5.0")

        metric_fn = METRIC_FNS.get(self.metric, _mae)
        folds     = self.splitter.split(X)

        if not folds:
            raise ValueError("Splitter returned no folds.")

        def _objective(trial):
            params = param_fn(trial)
            fold_scores = []
            for tr_idx, te_idx in folds:
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
                try:
                    mdl = self.model_class(**params)
                    mdl.fit(X_tr, y_tr)
                    preds      = mdl.predict(X_te)
                    fold_scores.append(metric_fn(y_te.values, preds))
                except Exception as exc:
                    logger.debug("Trial failed on fold: %s", exc)
                    fold_scores.append(np.inf)
            return float(np.mean(fold_scores))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(
            _objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=False,
        )

        # Re-evaluate all trials with per-fold breakdown
        rows = []
        for t in study.trials:
            if t.state.name != "COMPLETE":
                continue
            params = t.params
            fold_scores = []
            for tr_idx, te_idx in folds:
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
                try:
                    mdl = self.model_class(**params)
                    mdl.fit(X_tr, y_tr)
                    preds = mdl.predict(X_te)
                    fold_scores.append(metric_fn(y_te.values, preds))
                except Exception:
                    fold_scores.append(np.nan)
            rows.append({
                "trial_id":    t.number,
                "params":      params,
                "fold_scores": fold_scores,
                "mean_score":  float(np.nanmean(fold_scores)),
                "std_score":   float(np.nanstd(fold_scores)),
            })

        results = pd.DataFrame(rows).sort_values("mean_score")
        return results

    # ------------------------------------------------------------------

    def run_grid(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List[Any]],
    ) -> pd.DataFrame:
        """
        Exhaustive grid search (small grids only).
        Falls back to grid_sampler via Optuna.
        """
        try:
            import optuna
            from optuna.samplers import GridSampler
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("optuna required: pip install optuna>=3.5.0")

        search_space = {k: v for k, v in param_grid.items()}
        import itertools
        combos = list(itertools.product(*param_grid.values()))
        keys   = list(param_grid.keys())
        metric_fn = METRIC_FNS.get(self.metric, _mae)
        folds     = self.splitter.split(X)

        # Empty grid guard
        if not combos:
            return pd.DataFrame(columns=["trial_id", "params", "fold_scores",
                                         "mean_score", "std_score"])

        rows = []
        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            fold_scores = []
            for tr_idx, te_idx in folds:
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
                try:
                    mdl = self.model_class(**params)
                    mdl.fit(X_tr, y_tr)
                    preds = mdl.predict(X_te)
                    fold_scores.append(metric_fn(y_te.values, preds))
                except Exception:
                    fold_scores.append(np.nan)
            rows.append({
                "trial_id":    i,
                "params":      params,
                "fold_scores": fold_scores,
                "mean_score":  float(np.nanmean(fold_scores)),
                "std_score":   float(np.nanstd(fold_scores)),
            })

        results = pd.DataFrame(rows).sort_values("mean_score")
        return results


# ---------------------------------------------------------------------------
# Stability analysis & recommendation
# ---------------------------------------------------------------------------

def stability_analysis(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a stability_flag column.
    High variance = std_score > 0.5 × mean_score → True (unstable).
    """
    df = results_df.copy()
    df["stability_flag"] = (
        df["std_score"] > 0.5 * df["mean_score"].abs().replace(0, 1e-9)
    )
    return df


def recommend(results_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Return the params with best (lowest) mean_score among stable combos.
    If no stable combo exists, return the globally best.
    """
    if results_df.empty:
        return None
    df = stability_analysis(results_df)
    stable = df[~df["stability_flag"]]
    best_df = stable if not stable.empty else df
    best_row = best_df.sort_values("mean_score").iloc[0]
    return {
        "params":     best_row["params"],
        "mean_score": best_row["mean_score"],
        "std_score":  best_row["std_score"],
        "stable":     bool(not df.loc[best_row.name, "stability_flag"]),
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results_df: pd.DataFrame, label: str = "sweep") -> Path:
    """Save results to output/reports/hyperparam_sweep_LABEL_YYYYMMDD.csv."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    path  = REPORT_DIR / f"hyperparam_sweep_{label}_{today}.csv"
    # Serialize params dict to string for CSV
    out = results_df.copy()
    out["params"] = out["params"].apply(str)
    out.to_csv(path, index=False)
    logger.info("Sweep results saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Pre-configured sweeps (convenience wrappers)
# ---------------------------------------------------------------------------

def sweep_xgboost(X: pd.DataFrame, y: pd.Series, splitter,
                  n_trials: int = 50) -> pd.DataFrame:
    """XGBoost sweep with Optuna TPE over the Phase 6 search space."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("xgboost required")

    import optuna

    def _param_fn(trial):
        return {
            "max_depth":         trial.suggest_int("max_depth", 3, 6),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000, step=100),
            "min_child_weight":  trial.suggest_int("min_child_weight", 3, 10),
            "subsample":         trial.suggest_float("subsample", 0.7, 0.9),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 0.8),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1.0, 10.0),
            "huber_slope":       trial.suggest_categorical("huber_slope",
                                                           [0.5, 1.0, 1.35, 2.0]),
            "random_state":      42,
            "verbosity":         0,
        }

    class _XGBWrapper:
        def __init__(self, huber_slope=1.35, **kwargs):
            self._m = XGBRegressor(
                objective="reg:pseudohubererror",
                huber_slope=huber_slope,
                **kwargs,
            )
        def fit(self, X, y):      self._m.fit(X, y); return self
        def predict(self, X):     return self._m.predict(X)

    sw = HyperparamSweep(_XGBWrapper, splitter, n_trials=n_trials)
    return sw.run(X, y, _param_fn)


def sweep_lightgbm(X: pd.DataFrame, y: pd.Series, splitter,
                   n_trials: int = 50) -> pd.DataFrame:
    """LightGBM sweep with Optuna TPE."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        raise ImportError("lightgbm required")

    def _param_fn(trial):
        return {
            "num_leaves":         trial.suggest_int("num_leaves", 15, 63),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "n_estimators":       trial.suggest_int("n_estimators", 200, 1000, step=100),
            "min_child_samples":  trial.suggest_int("min_child_samples", 10, 50),
            "subsample":          trial.suggest_float("subsample", 0.7, 0.8),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 0.8),
            "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 0.1),
            "reg_lambda":         trial.suggest_float("reg_lambda", 0.0, 1.0),
            "random_state":       42,
            "verbosity":          -1,
        }

    class _LGBMWrapper:
        def __init__(self, **kwargs): self._m = LGBMRegressor(**kwargs)
        def fit(self, X, y):         self._m.fit(X, y); return self
        def predict(self, X):        return self._m.predict(X)

    sw = HyperparamSweep(_LGBMWrapper, splitter, n_trials=n_trials)
    return sw.run(X, y, _param_fn)
