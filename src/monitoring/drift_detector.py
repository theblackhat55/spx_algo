"""
src/monitoring/drift_detector.py
==================================
Task 34 — Model Drift Monitor.

Detects when predictions diverge from actuals before the weekly retrain.

Classes
-------
DriftDetector   Core drift monitor with 21-day rolling window.

Status enum
-----------
HEALTHY   All metrics within acceptable thresholds.
WARNING   One metric elevated; alert but continue trading.
DEGRADED  Multiple metrics failing; pause trading, request retrain.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DRIFT_LOG_PATH    = Path("output/monitoring/drift_log.csv")
RETRAIN_FLAG_PATH = Path("output/monitoring/retrain_requested.flag")
REPORT_DIR        = Path("output/reports")

DRIFT_LOG_COLUMNS = [
    "date",
    "high_error_pct",    # |actual_high% - predicted_high%|
    "low_error_pct",
    "covered_68",        # 1 if actual inside 68% interval
    "covered_90",
    "condor_win",        # 1=win, 0=loss, -1=skip
    "regime",
    "drift_status",
]


class DriftStatus(str, Enum):
    HEALTHY  = "HEALTHY"
    WARNING  = "WARNING"
    DEGRADED = "DEGRADED"


# ---------------------------------------------------------------------------
# PSI helper
# ---------------------------------------------------------------------------

def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two arrays."""
    lo = min(expected.min(), actual.min())
    hi = max(expected.max(), actual.max())
    if hi == lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    e_pct = np.histogram(expected, bins=edges)[0] / max(len(expected), 1)
    a_pct = np.histogram(actual,   bins=edges)[0] / max(len(actual),   1)
    e_pct = np.where(e_pct == 0, 1e-4, e_pct)
    a_pct = np.where(a_pct == 0, 1e-4, a_pct)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Rolling-window drift detector for the SPX iron-condor signal pipeline.

    Parameters
    ----------
    mae_red_threshold       : float  Rolling-21 MAE (% of index) → DEGRADED.
    mae_yellow_threshold    : float  Rolling-21 MAE → WARNING.
    coverage_floor          : float  Min empirical 68% coverage → WARNING below.
    consecutive_loss_limit  : int    Consecutive condor losses → WARNING.
    psi_warning_threshold   : float  PSI > this → WARNING.
    psi_red_threshold       : float  PSI > this → DEGRADED.
    log_path                : Path   Where drift_log.csv is written.
    retrain_flag_path       : Path   Where the retrain flag file is written.
    window                  : int    Rolling window size in trading days.
    """

    def __init__(
        self,
        mae_red_threshold:      float = 0.007,   # 0.7%
        mae_yellow_threshold:   float = 0.005,   # 0.5%
        coverage_floor:         float = 0.55,    # 55%
        consecutive_loss_limit: int   = 5,
        psi_warning_threshold:  float = 0.20,
        psi_red_threshold:      float = 0.25,
        log_path:               Path  = DRIFT_LOG_PATH,
        retrain_flag_path:      Path  = RETRAIN_FLAG_PATH,
        window:                 int   = 21,
    ):
        self.mae_red_threshold      = mae_red_threshold
        self.mae_yellow_threshold   = mae_yellow_threshold
        self.coverage_floor         = coverage_floor
        self.consecutive_loss_limit = consecutive_loss_limit
        self.psi_warning_threshold  = psi_warning_threshold
        self.psi_red_threshold      = psi_red_threshold
        self.log_path               = Path(log_path)
        self.retrain_flag_path      = Path(retrain_flag_path)
        self.window                 = window

        # In-memory rolling buffer (list of dicts)
        self._buffer: List[Dict[str, Any]] = []
        self._load_log()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        signal_date:    str,
        predicted_high: float,
        predicted_low:  float,
        actual_high:    float,
        actual_low:     float,
        lower_68_high:  float,
        upper_68_high:  float,
        lower_68_low:   float,
        upper_68_low:   float,
        condor_win:     int   = -1,   # 1=win, 0=loss, -1=skip
        regime:         str   = "UNKNOWN",
    ) -> Dict[str, Any]:
        """
        Record one day's prediction vs actuals and update the rolling window.

        Returns the row dict appended.
        """
        high_err = abs(actual_high - predicted_high)
        low_err  = abs(actual_low  - predicted_low)

        cov_68 = int(
            (lower_68_high <= actual_high <= upper_68_high) and
            (lower_68_low  <= actual_low  <= upper_68_low)
        )
        cov_90 = int(cov_68)   # simplified: treat 90 same as 68 unless explicit

        row = {
            "date":          signal_date,
            "high_error_pct": round(high_err, 6),
            "low_error_pct":  round(low_err,  6),
            "covered_68":     cov_68,
            "covered_90":     cov_90,
            "condor_win":     condor_win,
            "regime":         regime,
            "drift_status":   "",    # filled after check_drift
        }

        self._buffer.append(row)
        # Keep only the rolling window
        if len(self._buffer) > self.window:
            self._buffer = self._buffer[-self.window:]

        status = self.check_drift()
        row["drift_status"] = status.value
        self._append_to_log(row)
        return row

    def check_drift(self, feature_array: Optional[np.ndarray] = None,
                    train_array: Optional[np.ndarray] = None) -> DriftStatus:
        """
        Evaluate the rolling window and return HEALTHY / WARNING / DEGRADED.

        Parameters
        ----------
        feature_array   : Optional recent feature distribution for PSI.
        train_array     : Optional training feature distribution for PSI.
        """
        if not self._buffer:
            return DriftStatus.HEALTHY

        buf = self._buffer
        mae_high = np.mean([r["high_error_pct"] for r in buf])
        mae_low  = np.mean([r["low_error_pct"]  for r in buf])
        rolling_mae = max(mae_high, mae_low)

        cov_rate = np.mean([r["covered_68"] for r in buf])

        # Consecutive loss streak (ignore skips)
        active = [r for r in buf if r["condor_win"] != -1]
        consec = 0
        for r in reversed(active):
            if r["condor_win"] == 0:
                consec += 1
            else:
                break

        # PSI if arrays provided
        psi_score = 0.0
        if feature_array is not None and train_array is not None:
            try:
                psi_score = _psi(train_array, feature_array)
            except Exception:
                pass

        # --- Classify ---
        degraded_flags = 0
        warning_flags  = 0

        if rolling_mae > self.mae_red_threshold:
            degraded_flags += 1
        elif rolling_mae > self.mae_yellow_threshold:
            warning_flags += 1

        if cov_rate < self.coverage_floor:
            warning_flags += 1

        if consec >= self.consecutive_loss_limit:
            warning_flags += 1

        if psi_score > self.psi_red_threshold:
            degraded_flags += 1
        elif psi_score > self.psi_warning_threshold:
            warning_flags += 1

        if degraded_flags >= 1:
            return DriftStatus.DEGRADED
        if warning_flags >= 1:
            return DriftStatus.WARNING
        return DriftStatus.HEALTHY

    def trigger_retrain_if_needed(self) -> bool:
        """
        If the last 2 entries are both DEGRADED, write the retrain flag.
        Returns True if flag was written.
        """
        if len(self._buffer) < 2:
            return False
        last_two = [r.get("drift_status", "") for r in self._buffer[-2:]]
        if all(s == DriftStatus.DEGRADED.value for s in last_two):
            self.retrain_flag_path.parent.mkdir(parents=True, exist_ok=True)
            self.retrain_flag_path.write_text(
                datetime.now(timezone.utc).isoformat()
            )
            logger.warning("RETRAIN FLAG written: %s", self.retrain_flag_path)
            return True
        return False

    def daily_report(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary of the current rolling window."""
        if not self._buffer:
            return {"status": DriftStatus.HEALTHY.value, "window_rows": 0}

        buf = self._buffer
        status = self.check_drift()
        return {
            "status":            status.value,
            "window_rows":       len(buf),
            "rolling_mae_high":  round(np.mean([r["high_error_pct"] for r in buf]), 6),
            "rolling_mae_low":   round(np.mean([r["low_error_pct"]  for r in buf]), 6),
            "coverage_68":       round(np.mean([r["covered_68"]     for r in buf]), 4),
            "coverage_90":       round(np.mean([r["covered_90"]     for r in buf]), 4),
            "condor_win_rate":   round(
                np.mean([r["condor_win"] for r in buf if r["condor_win"] != -1] or [0]), 4
            ),
            "generated_at":      datetime.now(timezone.utc).isoformat(),
        }

    def save_report(self, report_dir: Path = REPORT_DIR) -> Path:
        """Save daily_report() as JSON to output/reports/drift_YYYYMMDD.json."""
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        today = date.today().isoformat()
        path  = report_dir / f"drift_{today}.json"
        path.write_text(json.dumps(self.daily_report(), indent=2))
        logger.info("Drift report saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_log(self) -> None:
        """Load existing drift_log.csv into the in-memory buffer."""
        if not self.log_path.exists():
            return
        try:
            df = pd.read_csv(self.log_path)
            tail = df.tail(self.window).to_dict("records")
            self._buffer = tail
            logger.debug("Drift log loaded: %d rows (window=%d)", len(tail), self.window)
        except Exception as exc:
            logger.warning("Could not load drift log: %s", exc)

    def _append_to_log(self, row: Dict[str, Any]) -> None:
        """Append a single row to the CSV log (creates file if absent)."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=DRIFT_LOG_COLUMNS)
            if write_header:
                writer.writeheader()
            # Write only the canonical columns
            writer.writerow({k: row.get(k, "") for k in DRIFT_LOG_COLUMNS})
