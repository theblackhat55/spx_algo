"""
src/calibration/adaptive_conformal.py
=====================================
Level 2: Adaptive conformal calibration.

Replaces fixed 63-day residual window with:
- Exponential decay weighting (half-life 15 days, recent errors matter more)
- Regime-aware quantiles (separate calibration per GREEN/YELLOW/RED)
- Auto-widen by 20% if recent coverage < 75% for 5 consecutive days
- Minimum interval width = 0.3% of SPX

Diagnostics saved to output/monitoring/calibration_health.json
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HEALTH_PATH = Path("output/monitoring/calibration_health.json")

# Defaults
DEFAULT_HALF_LIFE = 15       # days
DEFAULT_MIN_WIDTH_PCT = 0.003  # 0.3% of SPX
DEFAULT_WIDEN_FACTOR = 1.20   # +20% if under-covering
DEFAULT_COVERAGE_FLOOR = 0.75  # trigger widening below this
DEFAULT_FLOOR_STREAK = 5      # consecutive days below floor


class AdaptiveConformalPredictor:
    """
    Regime-aware, exponentially-weighted conformal prediction intervals.

    Usage:
        acp = AdaptiveConformalPredictor(model)
        acp.calibrate(X_cal, y_cal, regimes, half_life=15)
        intervals = acp.predict_interval(X_new, regime='YELLOW')
    """

    def __init__(
        self,
        model,
        alpha_list: Optional[List[float]] = None,
        half_life: int = DEFAULT_HALF_LIFE,
        min_width_pct: float = DEFAULT_MIN_WIDTH_PCT,
    ):
        self.model = model
        self.alpha_list = alpha_list or [0.68, 0.90]
        self.half_life = half_life
        self.min_width_pct = min_width_pct

        self._residuals: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        self._regimes: Optional[np.ndarray] = None
        self._calibrated = False

        # Tracking for auto-widening
        self._coverage_history: List[float] = []
        self._widen_active = False
        self._widen_factor = 1.0

    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        regimes: Optional[pd.Series] = None,
    ) -> "AdaptiveConformalPredictor":
        """
        Calibrate using exponentially-weighted residuals.

        Parameters
        ----------
        X_cal : calibration features
        y_cal : calibration targets
        regimes : Series of regime labels (GREEN/YELLOW/RED) aligned with X_cal
        """
        preds = self.model.predict(X_cal).astype(float)
        self._residuals = np.abs(y_cal.values - preds)

        # Exponential decay weights (most recent = highest weight)
        n = len(self._residuals)
        decay = np.log(2) / self.half_life
        self._weights = np.exp(-decay * np.arange(n)[::-1])
        self._weights /= self._weights.sum()

        # Store regimes
        if regimes is not None:
            self._regimes = regimes.values
        else:
            self._regimes = np.array(["UNKNOWN"] * n)

        self._calibrated = True
        logger.info(
            "AdaptiveConformal calibrated: %d samples, half_life=%d, regimes=%s",
            n, self.half_life,
            {r: int((self._regimes == r).sum()) for r in np.unique(self._regimes)},
        )
        return self

    def predict_interval(
        self,
        X: pd.DataFrame,
        regime: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate prediction intervals, optionally regime-filtered.

        If regime is provided, uses only residuals from matching regime
        (falls back to all residuals if fewer than 10 regime-matched samples).
        """
        if not self._calibrated:
            raise RuntimeError("Not calibrated. Call calibrate() first.")

        point = self.model.predict(X).astype(float)
        result = pd.DataFrame({"predicted": point}, index=X.index)

        for alpha in self.alpha_list:
            label = int(round(alpha * 100))
            q = self._weighted_quantile(alpha, regime)

            # Apply auto-widening if active
            q *= self._widen_factor

            # Enforce minimum width
            q = max(q, self.min_width_pct / 2)

            result[f"lower_{label}"] = point - q
            result[f"upper_{label}"] = point + q

        return result

    def _weighted_quantile(
        self,
        alpha: float,
        regime: Optional[str] = None,
    ) -> float:
        """
        Compute weighted quantile of absolute residuals.

        Uses regime-filtered residuals if enough samples exist.
        """
        residuals = self._residuals
        weights = self._weights

        # Regime filtering
        if regime and self._regimes is not None:
            mask = self._regimes == regime
            if mask.sum() >= 10:
                residuals = residuals[mask]
                weights = self._weights[mask]
                weights = weights / weights.sum()
            else:
                logger.debug(
                    "Regime %s has only %d samples, using all %d",
                    regime, mask.sum(), len(residuals),
                )

        # Weighted quantile
        sorted_idx = np.argsort(residuals)
        sorted_res = residuals[sorted_idx]
        sorted_wts = weights[sorted_idx]
        cumulative = np.cumsum(sorted_wts)

        # Find the index where cumulative weight crosses alpha
        idx = np.searchsorted(cumulative, alpha)
        idx = min(idx, len(sorted_res) - 1)

        return float(sorted_res[idx])

    def update_coverage(self, actual_covered_90: float):
        """
        Track recent coverage and activate widening if needed.

        Called after each day's reconciliation.
        """
        self._coverage_history.append(actual_covered_90)

        # Keep last 20 days
        if len(self._coverage_history) > 20:
            self._coverage_history = self._coverage_history[-20:]

        # Check if last N days are all below floor
        recent = self._coverage_history[-DEFAULT_FLOOR_STREAK:]
        if len(recent) >= DEFAULT_FLOOR_STREAK and all(c < DEFAULT_COVERAGE_FLOOR for c in recent):
            if not self._widen_active:
                self._widen_factor = DEFAULT_WIDEN_FACTOR
                self._widen_active = True
                logger.warning(
                    "AdaptiveConformal: activating +%.0f%% widening (coverage below %.0f%% for %d days)",
                    (DEFAULT_WIDEN_FACTOR - 1) * 100,
                    DEFAULT_COVERAGE_FLOOR * 100,
                    DEFAULT_FLOOR_STREAK,
                )
        else:
            if self._widen_active:
                self._widen_factor = 1.0
                self._widen_active = False
                logger.info("AdaptiveConformal: deactivating widening (coverage recovered)")

    def diagnostics(self) -> Dict:
        """Return calibration health metrics."""
        if not self._calibrated:
            return {"status": "not_calibrated"}

        diag = {
            "status": "calibrated",
            "n_samples": len(self._residuals),
            "half_life": self.half_life,
            "widen_active": self._widen_active,
            "widen_factor": self._widen_factor,
        }

        for alpha in self.alpha_list:
            label = int(round(alpha * 100))
            q_all = self._weighted_quantile(alpha, None)
            diag[f"q{label}_all"] = round(q_all, 6)

            for regime in ["GREEN", "YELLOW", "RED"]:
                q_r = self._weighted_quantile(alpha, regime)
                diag[f"q{label}_{regime}"] = round(q_r, 6)

        return diag

    def save_health(self, path: Path = HEALTH_PATH):
        """Save diagnostics to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        diag = self.diagnostics()
        diag["date"] = date.today().isoformat()
        with open(path, "w") as f:
            json.dump(diag, f, indent=2)
        logger.info("Calibration health saved → %s", path)
