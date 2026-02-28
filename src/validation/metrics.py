"""
src/validation/metrics.py
==========================
Task 17 — Validation & reporting metrics.

Functions
---------
classification_report_df    → Full sklearn report as DataFrame.
calibration_metrics         → Brier score, ECE, MCE, reliability curve.
trading_metrics             → Hit-rate, directional accuracy, P&L proxy.
generate_full_report        → Combined dict of all metric groups.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics as skm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------

def classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    target_name: str = "target",
) -> pd.DataFrame:
    """
    Return a DataFrame version of sklearn's classification_report plus
    ROC-AUC if y_proba is supplied.
    """
    report = skm.classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report).T
    df.index.name = "class"

    if y_proba is not None:
        try:
            auc = skm.roc_auc_score(y_true, y_proba)
            df.loc["roc_auc", "precision"] = auc
        except Exception:
            pass

    df.attrs["target"] = target_name
    return df


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute calibration diagnostics.

    Returns
    -------
    dict with keys: brier_score, ece, mce, reliability_df
    """
    y_true  = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)

    brier = float(skm.brier_score_loss(y_true, y_proba))

    # ECE / MCE via equal-width bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    rows: List[Dict] = []
    n   = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        conf     = y_proba[mask].mean()
        acc      = y_true[mask].mean()
        frac     = mask.sum() / n
        cal_err  = abs(conf - acc)
        ece     += frac * cal_err
        mce      = max(mce, cal_err)
        rows.append({"bin_lo": lo, "bin_hi": hi,
                     "confidence": conf, "accuracy": acc,
                     "count": mask.sum(), "fraction": frac})

    reliability_df = pd.DataFrame(rows)
    return {
        "brier_score": brier,
        "ece":         ece,
        "mce":         mce,
        "reliability_df": reliability_df,
    }


# ---------------------------------------------------------------------------
# Confusion matrix helper
# ---------------------------------------------------------------------------

def confusion_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, int]:
    """Return TP, FP, TN, FN counts."""
    cm   = skm.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}


# ---------------------------------------------------------------------------
# Trading metrics
# ---------------------------------------------------------------------------

def trading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    threshold_list: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compute prediction-quality metrics relevant to trading.

    hit_rate        : fraction of times predicted 1 is actually 1.
    miss_rate       : fraction of actual 1 missed (predicted 0).
    threshold_sweep : DataFrame of precision/recall/f1 vs threshold.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    pos_mask = y_pred == 1
    hit_rate = float(y_true[pos_mask].mean()) if pos_mask.sum() > 0 else float("nan")

    actual_pos = y_true == 1
    if actual_pos.sum() > 0:
        miss_rate = float((y_true[actual_pos] != y_pred[actual_pos]).mean())
    else:
        miss_rate = float("nan")

    result: Dict[str, Any] = {
        "hit_rate":  hit_rate,
        "miss_rate": miss_rate,
        "n_signals": int(pos_mask.sum()),
        "n_total":   len(y_true),
        "signal_rate": float(pos_mask.mean()),
    }

    # Threshold sweep (only if probabilities provided)
    if y_proba is not None:
        thresholds = threshold_list or list(np.arange(0.3, 0.85, 0.05))
        rows = []
        for thr in thresholds:
            p = (y_proba >= thr).astype(int)
            rows.append({
                "threshold": thr,
                "precision": skm.precision_score(y_true, p, zero_division=0),
                "recall":    skm.recall_score(y_true, p, zero_division=0),
                "f1":        skm.f1_score(y_true, p, zero_division=0),
                "n_signals": int((p == 1).sum()),
            })
        result["threshold_sweep"] = pd.DataFrame(rows)

    return result


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def generate_full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    target_name: str = "target",
) -> Dict[str, Any]:
    """Aggregate all metric groups into one dict."""
    report: Dict[str, Any] = {
        "target": target_name,
        "n_samples": len(y_true),
        "class_distribution": {
            "actual_pos_rate": float(np.mean(y_true)),
            "pred_pos_rate":   float(np.mean(y_pred)),
        },
        "classification": classification_report_df(
            y_true, y_pred, y_proba, target_name
        ),
        "confusion": confusion_summary(y_true, y_pred),
        "trading":   trading_metrics(y_true, y_pred, y_proba),
    }
    if y_proba is not None:
        report["calibration"] = calibration_metrics(y_true, y_proba)

    logger.info("Full report generated for '%s': %d samples.", target_name, len(y_true))
    return report


# ---------------------------------------------------------------------------
# Threshold optimiser
# ---------------------------------------------------------------------------

def optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Find the threshold that maximises ``metric`` on the validation set.

    Parameters
    ----------
    metric : 'f1' | 'precision' | 'recall' | 'accuracy' | 'roc_auc'

    Returns
    -------
    (best_threshold, best_score)
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    best_thr   = 0.5
    best_score = -np.inf

    for thr in thresholds:
        preds = (y_proba >= thr).astype(int)
        if metric == "f1":
            score = skm.f1_score(y_true, preds, zero_division=0)
        elif metric == "precision":
            score = skm.precision_score(y_true, preds, zero_division=0)
        elif metric == "recall":
            score = skm.recall_score(y_true, preds, zero_division=0)
        elif metric == "accuracy":
            score = skm.accuracy_score(y_true, preds)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_thr   = thr

    return float(best_thr), float(best_score)
