"""
src/calibration/conformal_verification.py
==========================================
Task 38 — Conformal Interval Wiring Verification.

Hard gate before paper trading: verifies real (non-stub) conformal
intervals achieve the expected empirical coverage.

Functions
---------
verify_conformal_coverage   Walk-forward coverage evaluation.
coverage_report             Pass/fail dict vs tolerance.
mean_interval_width_by_regime  Width sanity check per regime.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.calibration.conformal import ConformalPredictor
from src.models.linear_models  import RidgeRegressionModel

logger = logging.getLogger(__name__)

REPORT_DIR = Path("output/reports")

# Pass criteria: empirical coverage within ±8 pp of nominal
TOLERANCE_PP = 0.08


# ---------------------------------------------------------------------------
# Core verification
# ---------------------------------------------------------------------------

def verify_conformal_coverage(
    data:         pd.DataFrame,       # rows = dates, columns include "target" + features
    target_col:   str         = "target",
    feature_cols: Optional[List[str]] = None,
    splitter      = None,
    alpha_levels: List[float] = None,  # [0.32, 0.10] → 68%, 90%
    model_class   = None,
    regime_col:   Optional[str] = None,
) -> Dict[str, Any]:
    """
    Walk-forward evaluation of conformal coverage.

    For each out-of-sample fold:
      - Fit model on training data.
      - Calibrate ConformalPredictor on calibration tail (20% of train).
      - Compute intervals on test data.
      - Record empirical coverage.

    Returns a results dict with per-fold and aggregate coverage.
    """
    if alpha_levels is None:
        alpha_levels = [0.32, 0.10]   # 68%, 90%

    if feature_cols is None:
        feature_cols = [c for c in data.columns if c != target_col and c != regime_col]

    X = data[feature_cols].copy().fillna(0)
    y = data[target_col].copy()

    # Default splitter
    if splitter is None:
        from src.targets.splitter import WalkForwardSplitter, SplitConfig
        cfg = SplitConfig(
            min_train_rows = min(200, max(50, len(X) // 4)),
            test_rows      = max(10, len(X) // 10),
            step_rows      = max(10, len(X) // 10),
            gap_rows       = 0,
            expanding      = True,
        )
        splitter = WalkForwardSplitter(cfg)

    if model_class is None:
        model_class = RidgeRegressionModel

    folds = splitter.split(X)
    if not folds:
        logger.warning("verify_conformal_coverage: no folds produced; returning empty.")
        return {"folds": [], "aggregate": {}}

    # Storage: {alpha → {fold → {"n", "cov_high", "cov_low", "width"}}}
    fold_results: Dict[float, List[Dict]] = {a: [] for a in alpha_levels}
    regime_widths: Dict[str, List[float]] = {}

    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        # Calibration split: last 20% of training
        cal_split = int(len(tr_idx) * 0.8)
        X_fit, y_fit = X_tr.iloc[:cal_split], y_tr.iloc[:cal_split]
        X_cal, y_cal = X_tr.iloc[cal_split:], y_tr.iloc[cal_split:]

        if len(X_fit) < 10 or len(X_cal) < 5:
            continue

        try:
            model = model_class(name=f"fold_{fold_i}")
            model.fit(X_fit, y_fit)

            cp = ConformalPredictor(model, alpha_list=[1 - a for a in alpha_levels],
                                    use_mapie=False)
            cp.calibrate(X_cal, y_cal)

            intervals = cp.predict_interval(X_te)

            for alpha in alpha_levels:
                coverage_pct = 1 - alpha
                lv = int(round(coverage_pct * 100))
                lo_col = f"lower_{lv}"
                hi_col = f"upper_{lv}"

                if lo_col not in intervals.columns or hi_col not in intervals.columns:
                    # Try to compute on the fly
                    lo = cp._interval(X_te, alpha / 2)
                    intervals[lo_col] = lo["lower"]
                    intervals[hi_col] = lo["upper"]

                if lo_col not in intervals.columns:
                    fold_results[alpha].append({
                        "fold": fold_i, "n": len(y_te),
                        "coverage": np.nan, "mean_width": np.nan
                    })
                    continue

                covered = (
                    (y_te.values >= intervals[lo_col].values) &
                    (y_te.values <= intervals[hi_col].values)
                )
                mean_width = float((intervals[hi_col] - intervals[lo_col]).mean())

                fold_results[alpha].append({
                    "fold":       fold_i,
                    "n":          len(y_te),
                    "coverage":   float(covered.mean()),
                    "mean_width": mean_width,
                })

                # Regime breakdown
                if regime_col and regime_col in data.columns:
                    regimes = data[regime_col].iloc[te_idx].values
                    for reg in np.unique(regimes):
                        mask = regimes == reg
                        if mask.sum() == 0:
                            continue
                        w = float((intervals[hi_col].values[mask] -
                                   intervals[lo_col].values[mask]).mean())
                        regime_widths.setdefault(str(reg), []).append(w)

        except Exception as exc:
            logger.warning("Fold %d failed: %s", fold_i, exc)
            for alpha in alpha_levels:
                fold_results[alpha].append({
                    "fold": fold_i, "n": 0,
                    "coverage": np.nan, "mean_width": np.nan
                })

    # Aggregate
    aggregate = {}
    for alpha in alpha_levels:
        fr = fold_results[alpha]
        covs = [r["coverage"] for r in fr if not np.isnan(r.get("coverage", np.nan))]
        ws   = [r["mean_width"] for r in fr if not np.isnan(r.get("mean_width", np.nan))]
        aggregate[alpha] = {
            "target_coverage": round(1 - alpha, 2),
            "empirical_mean":  round(float(np.mean(covs)), 4) if covs else np.nan,
            "empirical_std":   round(float(np.std(covs)),  4) if covs else np.nan,
            "mean_width":      round(float(np.mean(ws)),   6) if ws   else np.nan,
            "n_folds":         len(fr),
        }

    return {
        "folds":         fold_results,
        "aggregate":     aggregate,
        "regime_widths": {k: round(float(np.mean(v)), 6)
                          for k, v in regime_widths.items()},
    }


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def coverage_report(
    results: Dict[str, Any],
    tolerance: float = TOLERANCE_PP,
) -> Dict[str, Any]:
    """
    Evaluate pass/fail for each alpha level.

    Pass criterion: |empirical_mean - target_coverage| <= tolerance.
    """
    report = {}
    for alpha, agg in results.get("aggregate", {}).items():
        target  = agg.get("target_coverage", 1 - alpha)
        actual  = agg.get("empirical_mean", np.nan)
        if np.isnan(actual):
            passed = False
        else:
            passed = abs(actual - target) <= tolerance
        report[f"alpha_{alpha}"] = {
            "target":   target,
            "actual":   actual,
            "delta":    round(float(actual - target), 4) if not np.isnan(actual) else None,
            "pass":     passed,
            "tolerance": tolerance,
        }

    all_pass = all(v["pass"] for v in report.values())
    report["overall_pass"] = all_pass
    return report


# ---------------------------------------------------------------------------
# Interval width by regime
# ---------------------------------------------------------------------------

def mean_interval_width_by_regime(
    results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Returns mean interval width per regime.
    Sanity check: RED > YELLOW > GREEN (wider intervals in stress regimes).
    """
    regime_widths = results.get("regime_widths", {})
    if not regime_widths:
        return {"note": "No regime data available."}

    ordered = {k: regime_widths[k]
               for k in ["RED", "YELLOW", "GREEN"] if k in regime_widths}

    # Check ordering
    values = list(ordered.values())
    ordering_correct = all(values[i] >= values[i+1] for i in range(len(values)-1))

    return {
        "widths_by_regime": ordered,
        "ordering_correct": ordering_correct,
        "note": "Expect RED >= YELLOW >= GREEN width",
    }


# ---------------------------------------------------------------------------
# Stub detection
# ---------------------------------------------------------------------------

def detect_stub_intervals(
    results: Dict[str, Any],
) -> bool:
    """
    Returns True if all fold widths are identical (constant intervals = stub).
    """
    for alpha, fr in results.get("folds", {}).items():
        widths = [r.get("mean_width") for r in fr
                  if r.get("mean_width") is not None and not np.isnan(r.get("mean_width", np.nan))]
        if len(widths) > 1:
            if np.std(widths) < 1e-9:
                return True   # constant → stub
    return False


# ---------------------------------------------------------------------------
# Save report
# ---------------------------------------------------------------------------

def save_coverage_report(
    cov_report: Dict[str, Any],
    report_dir: Path = REPORT_DIR,
) -> Path:
    """Write conformal_coverage_YYYYMMDD.json."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    path  = report_dir / f"conformal_coverage_{today}.json"
    payload = {k: v for k, v in cov_report.items()}
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()

    # Make sure numpy types are JSON-serialisable
    def _cast(obj):
        if isinstance(obj, dict):
            return {k: _cast(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_cast(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    path.write_text(json.dumps(_cast(payload), indent=2))
    logger.info("Conformal coverage report saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Verify conformal coverage (hard gate).")
    ap.add_argument("--rows", type=int, default=504,
                    help="Number of synthetic rows if real data absent (default 504).")
    args = ap.parse_args()

    # Try to load real feature data; fall back to synthetic
    feat_path = Path("data/processed/features.parquet")
    if feat_path.exists():
        df = pd.read_parquet(feat_path)
        target_col = "target_high_pct"
        if target_col not in df.columns:
            target_col = df.columns[-1]
    else:
        rng = np.random.default_rng(42)
        n   = args.rows
        df  = pd.DataFrame(
            {"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n),
             "target": rng.normal(0, 0.01, n)},
            index=pd.bdate_range("2022-01-03", periods=n),
        )
        target_col = "target"

    results = verify_conformal_coverage(df, target_col=target_col)
    report  = coverage_report(results)
    path    = save_coverage_report(report)

    print("\n=== Conformal Coverage Verification ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    overall = report.get("overall_pass", False)
    print(f"\n{'✅ GATE PASSED' if overall else '❌ GATE FAILED'}")
    import sys as _sys
    _sys.exit(0 if overall else 2)
