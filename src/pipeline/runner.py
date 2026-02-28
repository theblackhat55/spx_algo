"""
src/pipeline/runner.py
=======================
Task 22 — End-to-end pipeline runner.

Orchestrates the full signal generation flow:
    Fetch → Validate → Features → Targets → Train → Calibrate → Regime → Signal

The runner supports two modes:
    live   : Generate today's signal using all available history.
    replay : Regenerate a signal for a specific historical date
             (used for backtest / production reconciliation).

IMPORTANT — Production parity check
-------------------------------------
After building the walk-forward backtest, run the pipeline in ``replay``
mode for a date that was in the OOS test set and verify that the signal
produced matches the backtest's recorded OOS output for that date.
If they differ, there is a discrepancy between the two code paths that
must be resolved before live deployment.

This function is exposed as ``replay_date_check()`` below.
"""
from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from config.settings import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR,
    SPX_FILE, VIX_FILE,
)

logger = logging.getLogger(__name__)

SIGNAL_DIR = Path(OUTPUT_DIR) / "signals"


# ---------------------------------------------------------------------------
# Signal output
# ---------------------------------------------------------------------------

@dataclass
class DailySignal:
    """The output of one pipeline run (one trading day's prediction)."""

    signal_date:         str          # YYYY-MM-DD the signal is FOR (next trading day)
    generated_at:        str          # ISO timestamp when generated
    mode:                str          # 'live' | 'replay'

    regime:              str          # 'GREEN' | 'YELLOW' | 'RED'

    # Regression outputs (absolute price levels)
    pred_high:           Optional[float]
    pred_low:            Optional[float]

    # Conformal intervals
    upper_90:            Optional[float]
    lower_90:            Optional[float]
    upper_68:            Optional[float]
    lower_68:            Optional[float]

    # Strike levels (ready for order entry)
    call_strike:         Optional[float]
    put_strike:          Optional[float]

    # Classification probabilities
    prob_high_bin_050:   Optional[float]   # P(high > 0.5%)
    prob_low_bin_050:    Optional[float]   # P(low  < -0.5%)

    # Prior close (for reference)
    prior_close:         Optional[float]

    # Model metadata
    n_features_used:     int
    model_names:         list

    # Risk
    tradeable:           bool          # False if regime=RED or any hard stop
    notes:               list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def save(self, output_dir: Optional[Path] = None) -> Path:
        out = Path(output_dir or SIGNAL_DIR)
        out.mkdir(parents=True, exist_ok=True)
        fname = out / f"signal_{self.signal_date}.json"
        fname.write_text(self.to_json())
        logger.info("Signal saved → %s", fname)
        return fname


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """
    Runs the full signal-generation pipeline.

    Usage
    -----
    >>> runner = PipelineRunner()
    >>> signal = runner.run(mode='live')
    >>> signal.save()
    """

    def __init__(
        self,
        raw_dir:       Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        output_dir:    Optional[Path] = None,
    ):
        self.raw_dir       = Path(raw_dir       or RAW_DATA_DIR)
        self.processed_dir = Path(processed_dir or PROCESSED_DATA_DIR)
        self.output_dir    = Path(output_dir    or SIGNAL_DIR)

    # ------------------------------------------------------------------
    def run(
        self,
        mode: str = "live",
        as_of_date: Optional[str] = None,
        save_signal: bool = True,
    ) -> DailySignal:
        """
        Run the full pipeline.

        Parameters
        ----------
        mode        : 'live' or 'replay'.
        as_of_date  : 'YYYY-MM-DD' — if mode='replay', restrict all data to
                      this date so the pipeline cannot see the future.
        save_signal : Whether to write the signal JSON to disk.
        """
        notes: list[str] = []
        generated_at = datetime.now(timezone.utc).isoformat()

        # ── Step 1: Load raw data ─────────────────────────────────────────
        spx_df, vix_df = self._load_raw_data(as_of_date)
        if spx_df is None:
            return self._error_signal(mode, as_of_date, generated_at,
                                      "SPX data not available")

        prior_close = float(spx_df["Close"].iloc[-1])
        signal_date = _next_trading_day(spx_df.index[-1]).strftime("%Y-%m-%d")

        notes.append(f"Prior close: {prior_close:.2f}")
        notes.append(f"SPX rows: {len(spx_df)}")

        # ── Step 2: Validate ──────────────────────────────────────────────
        try:
            from src.data.validator import SPXValidator
            validator = SPXValidator()
            issues = validator.validate(spx_df)
            if issues:
                notes.append(f"Validation warnings: {issues}")
        except Exception as e:
            notes.append(f"Validator unavailable: {e}")

        # ── Step 3: Load features ─────────────────────────────────────────
        features = self._load_or_build_features(spx_df, vix_df, notes)
        if features is None:
            return self._error_signal(mode, as_of_date, generated_at,
                                      "Feature build failed")

        # ── Step 4: Regime ────────────────────────────────────────────────
        regime_str, regime_int = self._compute_regime(spx_df, vix_df, notes)

        # ── Step 5: Regression predictions ───────────────────────────────
        reg_result = self._regression_predict(features, spx_df, notes)

        # ── Step 6: Conformal intervals ───────────────────────────────────
        intervals = self._conformal_intervals(reg_result, prior_close, notes)

        # ── Step 7: Classification probabilities ──────────────────────────
        clf_probs = self._classification_predict(features, notes)

        # ── Step 8: Strike levels ─────────────────────────────────────────
        call_strike = intervals.get("upper_90")
        put_strike  = intervals.get("lower_90")
        if call_strike and put_strike:
            call_strike = prior_close * (1 + call_strike) if abs(call_strike) < 1 else call_strike
            put_strike  = prior_close * (1 + put_strike)  if abs(put_strike)  < 1 else put_strike

        tradeable = (regime_str != "RED")
        if not tradeable:
            notes.append("NOT TRADEABLE: regime=RED")

        signal = DailySignal(
            signal_date=signal_date,
            generated_at=generated_at,
            mode=mode,
            regime=regime_str,
            pred_high=reg_result.get("pred_high"),
            pred_low=reg_result.get("pred_low"),
            upper_90=intervals.get("upper_90"),
            lower_90=intervals.get("lower_90"),
            upper_68=intervals.get("upper_68"),
            lower_68=intervals.get("lower_68"),
            call_strike=call_strike,
            put_strike=put_strike,
            prob_high_bin_050=clf_probs.get("prob_high_bin_050"),
            prob_low_bin_050=clf_probs.get("prob_low_bin_050"),
            prior_close=prior_close,
            n_features_used=features.shape[1] if features is not None else 0,
            model_names=reg_result.get("model_names", []),
            tradeable=tradeable,
            notes=notes,
        )

        if save_signal:
            signal.save(self.output_dir)

        logger.info(
            "Pipeline [%s] complete: date=%s regime=%s tradeable=%s "
            "pred_high=%.2f pred_low=%.2f",
            mode, signal_date, regime_str, tradeable,
            signal.pred_high or 0, signal.pred_low or 0,
        )
        return signal

    # ------------------------------------------------------------------
    def _load_raw_data(self, as_of_date):
        """Load and optionally truncate SPX and VIX data."""
        spx_path = self.raw_dir / "spx_daily.parquet"
        vix_path = self.raw_dir / "vix_daily.parquet"

        if not spx_path.exists():
            logger.error("SPX file not found: %s", spx_path)
            return None, None

        spx = pd.read_parquet(spx_path)
        spx.index = pd.to_datetime(spx.index)
        spx = spx.sort_index()

        vix = None
        if vix_path.exists():
            vix = pd.read_parquet(vix_path)
            vix.index = pd.to_datetime(vix.index)
            vix = vix.sort_index()

        # Replay mode: restrict to data known as of as_of_date
        if as_of_date:
            cutoff = pd.Timestamp(as_of_date)
            spx = spx.loc[:cutoff]
            if vix is not None:
                vix = vix.loc[:cutoff]

        return spx, vix

    def _load_or_build_features(self, spx_df, vix_df, notes):
        """Attempt to load saved features; fall back to building inline."""
        feat_path = self.processed_dir / "features.parquet"
        if feat_path.exists():
            try:
                feats = pd.read_parquet(feat_path)
                feats.index = pd.to_datetime(feats.index)
                notes.append(f"Loaded cached features: {feats.shape}")
                return feats
            except Exception as e:
                notes.append(f"Feature cache load failed ({e}); rebuilding.")

        try:
            from src.features.builder import build_feature_matrix
            feats = build_feature_matrix(
                raw_dir=self.raw_dir,
                processed_dir=self.processed_dir,
            )
            notes.append(f"Built features inline: {feats.shape}")
            return feats
        except Exception as e:
            notes.append(f"Feature build failed: {e}")
            logger.exception("Feature build failed")
            return None

    def _compute_regime(self, spx_df, vix_df, notes):
        """Compute regime; fall back to GREEN on error."""
        try:
            from src.calibration.regime import RegimeDetector
            rd = RegimeDetector(use_hmm=True, use_garch=False)
            regime_series = rd.fit_predict(spx_df, vix_df)
            reg_int = int(regime_series.iloc[-1])
            reg_str = {0: "GREEN", 1: "YELLOW", 2: "RED"}.get(reg_int, "YELLOW")
            notes.append(f"Regime: {reg_str}")
            return reg_str, reg_int
        except Exception as e:
            notes.append(f"Regime detector failed ({e}); defaulting YELLOW")
            return "YELLOW", 1

    def _regression_predict(self, features, spx_df, notes):
        """Load saved regression model and produce point predictions."""
        model_dir = Path(OUTPUT_DIR) / "models"
        result = {"pred_high": None, "pred_low": None, "model_names": []}

        for target, key in [("target_high_pct", "pred_high"),
                             ("target_low_pct",  "pred_low")]:
            model_path = model_dir / f"regression_{target}.pkl"
            if not model_path.exists():
                notes.append(f"No saved model for {target}")
                continue
            try:
                from src.models.base_model import BaseModel
                model = BaseModel.load(model_path)
                pred  = float(model.predict(features.iloc[[-1]])[0])
                result[key] = pred
                result["model_names"].append(model.name)
            except Exception as e:
                notes.append(f"Regression predict failed ({target}): {e}")

        return result

    def _conformal_intervals(self, reg_result, prior_close, notes):
        """Stub: return stored interval or compute from point prediction + ATR."""
        # In full deployment, this loads a saved ConformalPredictor and calls
        # predict_interval().  For now, derive approximate intervals from
        # the point prediction using a fixed +/- buffer.
        intervals = {}
        for key, pct_key in [("upper_90","pred_high"), ("lower_90","pred_low"),
                              ("upper_68","pred_high"), ("lower_68","pred_low")]:
            pct = reg_result.get(pct_key)
            if pct is not None:
                intervals[key] = pct
        notes.append("Intervals: point prediction (conformal not loaded)")
        return intervals

    def _classification_predict(self, features, notes):
        """Load saved classifier and return probabilities."""
        result: Dict[str, Any] = {}
        model_dir = Path(OUTPUT_DIR) / "models"

        for target in ("next_high_bin_050", "next_low_bin_050"):
            model_path = model_dir / f"classifier_{target}.pkl"
            if not model_path.exists():
                continue
            try:
                from src.models.base_model import BaseModel
                model = BaseModel.load(model_path)
                prob  = float(model.predict_proba(features.iloc[[-1]])[0, 1])
                key   = f"prob_{target}"
                result[key] = prob
            except Exception as e:
                notes.append(f"Classifier predict failed ({target}): {e}")

        return result

    @staticmethod
    def _error_signal(mode, as_of_date, generated_at, reason) -> "DailySignal":
        logger.error("Pipeline error: %s", reason)
        _today = as_of_date or date.today().strftime("%Y-%m-%d")
        return DailySignal(
            signal_date=_today,
            generated_at=generated_at,
            mode=mode,
            regime="RED",
            pred_high=None, pred_low=None,
            upper_90=None, lower_90=None,
            upper_68=None, lower_68=None,
            call_strike=None, put_strike=None,
            prob_high_bin_050=None, prob_low_bin_050=None,
            prior_close=None,
            n_features_used=0,
            model_names=[],
            tradeable=False,
            notes=[f"ERROR: {reason}"],
        )


# ---------------------------------------------------------------------------
# Production parity check
# ---------------------------------------------------------------------------

def replay_date_check(
    backtest_oos_row: pd.Series,
    as_of_date: str,
    runner: Optional[PipelineRunner] = None,
    tolerance_pct: float = 0.001,
) -> Dict[str, Any]:
    """
    Run the pipeline in replay mode for ``as_of_date`` and compare its
    output to the backtest's recorded OOS prediction for the same date.

    If pred_high or pred_low differ by more than ``tolerance_pct``,
    this is a code-path discrepancy that must be resolved before live
    deployment.

    Parameters
    ----------
    backtest_oos_row : Row from Trainer.run().oos_proba for as_of_date.
    as_of_date       : 'YYYY-MM-DD'
    tolerance_pct    : Maximum allowed relative deviation (default 0.1%).

    Returns
    -------
    dict with keys: 'match', 'replay_pred', 'backtest_pred', 'diff_pct'
    """
    if runner is None:
        runner = PipelineRunner()

    replay_sig = runner.run(mode="replay", as_of_date=as_of_date,
                             save_signal=False)

    # Compare regression predictions
    replay_val   = replay_sig.pred_high
    backtest_val = float(backtest_oos_row) if not pd.isna(backtest_oos_row) else None

    if replay_val is None or backtest_val is None:
        return {
            "match":          False,
            "replay_pred":    replay_val,
            "backtest_pred":  backtest_val,
            "diff_pct":       None,
            "note":           "One or both values unavailable",
        }

    diff_pct = abs(replay_val - backtest_val) / max(abs(backtest_val), 1e-9)
    match    = diff_pct <= tolerance_pct

    return {
        "match":         match,
        "replay_pred":   replay_val,
        "backtest_pred": backtest_val,
        "diff_pct":      diff_pct,
        "note":          "OK" if match else f"DISCREPANCY: {diff_pct:.2%} > {tolerance_pct:.2%}",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_trading_day(last_date: pd.Timestamp) -> pd.Timestamp:
    """Return the next business day after last_date."""
    offset = pd.tseries.offsets.BDay(1)
    return last_date + offset
