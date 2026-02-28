"""
src/features/selector.py
=========================
Task 11 — Feature selection pipeline.

Reduces the full feature set (~70 columns) to 15–25 high-quality,
non-redundant predictors using a three-stage approach:
  1. Correlation pruning     — remove near-duplicate features
  2. Importance ranking      — tree model gain importance
  3. Permutation importance  — out-of-sample shuffle test

For RFE the function accepts an optional walk_forward_func to compute
OOB MAE at each feature count; if not provided, a simple train/test
split is used.

LOOKAHEAD-BIAS CONTRACT
-----------------------
Feature selection is performed only on the training portion of the data.
It must never look at test-set targets.  The caller is responsible for
passing only training data to select_features().

Functions
---------
remove_correlated_features    — Pairwise correlation pruning
compute_feature_importance    — Gain-based tree importance
run_rfe                       — Recursive Feature Elimination
compute_permutation_importance— Shuffle-based importance with p-values
select_features               — Master orchestrator
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SelectionReport:
    """Summary of the feature selection process."""
    selected_features: list[str]
    removed_by_correlation: list[str]
    importance_scores: pd.DataFrame            # all features × importance
    rfe_curve: pd.DataFrame                    # n_features × MAE
    recommended_n: int
    permutation_importance: pd.DataFrame       # selected features × perm importance
    final_feature_count: int

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "FEATURE SELECTION REPORT",
            "=" * 60,
            f"  Input features       : {len(self.importance_scores)}",
            f"  Removed (correlation): {len(self.removed_by_correlation)}",
            f"  Final features       : {self.final_feature_count}",
            "",
            "  Selected features:",
        ]
        for f in self.selected_features:
            lines.append(f"    • {f}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Correlation pruning
# ─────────────────────────────────────────────────────────────────────────────

def remove_correlated_features(
    df: pd.DataFrame,
    target: pd.Series,
    threshold: float = 0.95,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove features with pairwise absolute correlation above *threshold*.

    When two features are highly correlated, keep the one with the higher
    univariate Spearman correlation with the target variable.

    Parameters
    ----------
    df:
        Feature matrix (rows = dates, columns = features).
    target:
        Target variable aligned to *df* (used for tie-breaking).
    threshold:
        Maximum allowed absolute correlation between retained features.

    Returns
    -------
    (reduced_df, removed_features_list)
    """
    corr_matrix = df.corr(method="spearman").abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Univariate correlations with target (for tie-breaking)
    target_aligned = target.reindex(df.index)
    uni_corr = df.apply(
        lambda col: col.corr(target_aligned, method="spearman")
    ).abs()

    to_drop: list[str] = []
    dropped_set: set[str] = set()

    for col in upper.columns:
        if col in dropped_set:
            continue
        highly_corr = upper[col][upper[col] > threshold].index.tolist()
        for other in highly_corr:
            if other in dropped_set:
                continue
            # Drop whichever has the lower target correlation
            if uni_corr.get(col, 0) >= uni_corr.get(other, 0):
                to_drop.append(other)
                dropped_set.add(other)
            else:
                to_drop.append(col)
                dropped_set.add(col)
                break   # col is now dropped; skip remaining pairs

    to_drop = list(set(to_drop))
    reduced = df.drop(columns=to_drop)
    logger.info(
        "Correlation pruning: %d/%d features removed (threshold=%.2f)",
        len(to_drop), len(df.columns), threshold,
    )
    return reduced, to_drop


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model=None,
) -> pd.DataFrame:
    """Fit a model and extract gain-based feature importance.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    y_train:
        Training target.
    model:
        A fitted or unfitted model with a ``feature_importances_`` attribute
        after fitting.  If None, attempts to use XGBoost.

    Returns
    -------
    DataFrame with columns ``feature`` and ``importance``, sorted descending.
    """
    if model is None:
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        except ImportError:
            try:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
                )
            except ImportError:
                logger.error("No tree model available for importance computation.")
                return pd.DataFrame(
                    {"feature": X_train.columns, "importance": 0.0}
                ).sort_values("importance", ascending=False)

    model.fit(X_train, y_train)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        logger.warning("Model has no feature_importances_ attribute.")
        return pd.DataFrame({"feature": X_train.columns, "importance": 0.0})

    result = pd.DataFrame({
        "feature": X_train.columns.tolist(),
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3a: RFE
# ─────────────────────────────────────────────────────────────────────────────

def run_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    model=None,
    min_features: int = 10,
    max_features: int = 30,
    walk_forward_func: Optional[Callable] = None,
) -> tuple[pd.DataFrame, int]:
    """Recursive Feature Elimination using MAE as the criterion.

    Starts with all features and iteratively removes the least important
    feature, recording the OOB MAE at each step.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target series.
    model:
        Model to use for importance ranking.  Defaults to XGBoost.
    min_features:
        Stop removing features when this count is reached.
    max_features:
        Start RFE from this many features (prune if more are provided).
    walk_forward_func:
        Optional callable(X, y) → MAE for more rigorous OOB evaluation.
        If None, a 70/30 train-test split is used.

    Returns
    -------
    (rfe_curve_df, recommended_n_features)
      rfe_curve_df has columns: n_features, mae, features_removed
    """
    current_features = X.columns.tolist()

    # Limit to max_features first
    if len(current_features) > max_features:
        imp = compute_feature_importance(
            X[current_features], y, model
        )
        current_features = imp["feature"].tolist()[:max_features]

    records = []

    while len(current_features) > min_features:
        X_sub = X[current_features]

        if walk_forward_func is not None:
            mae = walk_forward_func(X_sub, y)
        else:
            # Simple 70/30 split
            split_idx = int(len(X_sub) * 0.7)
            X_tr, X_te = X_sub.iloc[:split_idx], X_sub.iloc[split_idx:]
            y_tr, y_te = y.iloc[:split_idx],     y.iloc[split_idx:]

            if model is None:
                try:
                    from xgboost import XGBRegressor
                    _m = XGBRegressor(
                        n_estimators=100, max_depth=4,
                        tree_method="hist", random_state=42,
                        verbosity=0, n_jobs=-1,
                    )
                except ImportError:
                    from sklearn.ensemble import RandomForestRegressor
                    _m = RandomForestRegressor(
                        n_estimators=50, max_depth=6,
                        random_state=42, n_jobs=-1,
                    )
            else:
                import copy
                _m = copy.deepcopy(model)

            _m.fit(X_tr, y_tr)
            preds = _m.predict(X_te)
            mae = float(np.mean(np.abs(y_te.values - preds)))

        records.append({
            "n_features": len(current_features),
            "mae": mae,
            "features_kept": current_features.copy(),
        })

        # Remove the least important feature
        imp = compute_feature_importance(X[current_features], y, model)
        worst = imp["feature"].iloc[-1]
        current_features.remove(worst)

    rfe_df = pd.DataFrame(records).sort_values("n_features")

    # Recommend: minimum MAE feature count (allow 1% tolerance for parsimony)
    min_mae = rfe_df["mae"].min()
    tolerance = min_mae * 1.01
    parsimonious = rfe_df[rfe_df["mae"] <= tolerance]
    recommended_n = int(parsimonious["n_features"].min())

    logger.info(
        "RFE complete: recommended %d features (min MAE = %.4f at %d features)",
        recommended_n, min_mae,
        int(rfe_df.loc[rfe_df["mae"].idxmin(), "n_features"]),
    )
    return rfe_df, recommended_n


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3b: Permutation importance
# ─────────────────────────────────────────────────────────────────────────────

def compute_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Permutation importance on the test fold.

    For each feature, randomly shuffle its values *n_repeats* times and
    record the increase in MAE.  Features that don't matter will show
    near-zero change.

    Parameters
    ----------
    model:
        Fitted model with a ``predict`` method.
    X_test:
        Hold-out feature matrix.
    y_test:
        Hold-out target.
    n_repeats:
        Number of shuffle repetitions.

    Returns
    -------
    DataFrame with columns: feature, importance_mean, importance_std, p_value.
    """
    rng = np.random.default_rng(42)
    baseline_preds = model.predict(X_test)
    baseline_mae   = float(np.mean(np.abs(y_test.values - baseline_preds)))

    results = []
    for col in X_test.columns:
        maes = []
        for _ in range(n_repeats):
            X_perm         = X_test.copy()
            X_perm[col]    = rng.permutation(X_perm[col].values)
            perm_preds     = model.predict(X_perm)
            perm_mae       = float(np.mean(np.abs(y_test.values - perm_preds)))
            maes.append(perm_mae - baseline_mae)   # positive = useful feature

        arr = np.array(maes)
        # p-value: proportion of shuffles where importance <= 0
        # (one-sided test: H0 = feature is useless)
        p_value = float((arr <= 0).mean())

        results.append({
            "feature":          col,
            "importance_mean":  float(arr.mean()),
            "importance_std":   float(arr.std()),
            "p_value":          p_value,
        })

    return (
        pd.DataFrame(results)
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Master orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def select_features(
    feature_matrix: pd.DataFrame,
    target_series: pd.Series,
    n_features: int = 20,
    correlation_threshold: float = 0.95,
    model=None,
) -> tuple[list[str], SelectionReport]:
    """Run the full three-stage feature selection pipeline.

    Parameters
    ----------
    feature_matrix:
        Full feature matrix — TRAINING DATA ONLY.
    target_series:
        Target variable aligned to feature_matrix.
    n_features:
        Target number of features.
    correlation_threshold:
        Pairwise correlation threshold for pruning.
    model:
        Model for importance computation.  Defaults to XGBoost.

    Returns
    -------
    (selected_feature_names, SelectionReport)
    """
    logger.info("Starting feature selection pipeline …")
    logger.info("  Input:  %d features", len(feature_matrix.columns))

    # ── Stage 1: Correlation pruning ─────────────────────────────────────────
    X_pruned, removed_by_corr = remove_correlated_features(
        feature_matrix, target_series, threshold=correlation_threshold
    )

    # ── Stage 2: Full importance on pruned set ────────────────────────────────
    imp_df = compute_feature_importance(X_pruned, target_series, model)

    # ── Stage 3a: RFE ─────────────────────────────────────────────────────────
    rfe_df, recommended_n = run_rfe(
        X_pruned, target_series, model,
        min_features=max(10, n_features - 5),
        max_features=min(len(X_pruned.columns), n_features + 10),
    )

    # Retrieve the feature set at the recommended count
    rfe_row = rfe_df[rfe_df["n_features"] == recommended_n]
    if rfe_row.empty:
        # Fall back to top-N by importance
        selected = imp_df["feature"].tolist()[:n_features]
    else:
        selected = rfe_row.iloc[0]["features_kept"]

    # ── Stage 3b: Permutation importance on hold-out ──────────────────────────
    split_idx = int(len(X_pruned) * 0.7)
    X_tr = X_pruned[selected].iloc[:split_idx]
    y_tr = target_series.iloc[:split_idx]
    X_te = X_pruned[selected].iloc[split_idx:]
    y_te = target_series.iloc[split_idx:]

    if model is None:
        try:
            from xgboost import XGBRegressor
            _final_model = XGBRegressor(
                n_estimators=200, max_depth=4, tree_method="hist",
                random_state=42, n_jobs=-1, verbosity=0,
            )
        except ImportError:
            from sklearn.ensemble import RandomForestRegressor
            _final_model = RandomForestRegressor(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=-1,
            )
    else:
        import copy
        _final_model = copy.deepcopy(model)

    _final_model.fit(X_tr, y_tr)
    perm_imp = compute_permutation_importance(_final_model, X_te, y_te)

    logger.info("  Output: %d features selected", len(selected))

    report = SelectionReport(
        selected_features=list(selected),
        removed_by_correlation=removed_by_corr,
        importance_scores=imp_df,
        rfe_curve=rfe_df[["n_features", "mae"]],
        recommended_n=recommended_n,
        permutation_importance=perm_imp,
        final_feature_count=len(selected),
    )
    return list(selected), report
