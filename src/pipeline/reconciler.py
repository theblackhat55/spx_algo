"""
src/pipeline/reconciler.py
============================
Task 36 — Daily Reconciliation Runner.

At 10:00 AM ET each trading day:
  1. Fetches yesterday's actual OHLCV via live_fetcher.
  2. Loads yesterday's signal from output/signals/.
  3. Computes prediction errors.
  4. Updates the paper-trade log.
  5. Runs the drift detector.
  6. Sends a reconciliation alert via alerting module.

Additional:
  generate_weekly_digest(end_date)  — Friday summary.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.monitoring.drift_detector import DriftDetector, DriftStatus
from src.execution.paper_logger    import PaperTradeLogger
from src.pipeline.alerting         import Alerter, AlertConfig

logger = logging.getLogger(__name__)

SIGNAL_DIR  = Path("output/signals")
OUTPUT_DIR  = Path("output")
REPORT_DIR  = Path("output/reports")


# ---------------------------------------------------------------------------
# Helper: load signal JSON
# ---------------------------------------------------------------------------

def _load_signal(signal_date: str, signal_dir: Path = SIGNAL_DIR) -> Optional[Dict[str, Any]]:
    """Load signal_YYYYMMDD.json; return None if missing."""
    path = signal_dir / f"signal_{signal_date}.json"
    if not path.exists():
        logger.warning("Signal file not found: %s", path)
        return None
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Helper: fetch actual OHLCV for a date
# ---------------------------------------------------------------------------

def _fetch_actual(trade_date: str) -> Optional[Dict[str, float]]:
    """
    Try live_fetcher first; fall back to the parquet file.
    Returns {"High": ..., "Low": ..., "Close": ..., "Open": ...} or None.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("^GSPC")
        hist = ticker.history(start=trade_date, end=str(
            (date.fromisoformat(trade_date) + timedelta(days=3)).isoformat()
        ))
        if not hist.empty and trade_date in [str(d.date()) for d in hist.index]:
            row = hist.iloc[0]
            return {"High": row["High"], "Low": row["Low"],
                    "Close": row["Close"], "Open": row["Open"]}
    except Exception as exc:
        logger.debug("yfinance fetch failed: %s", exc)

    # Parquet fallback
    parquet = Path("data/raw/spx_daily.parquet")
    if parquet.exists():
        try:
            df = pd.read_parquet(parquet)
            df.index = pd.to_datetime(df.index)
            mask = df.index.normalize() == pd.Timestamp(trade_date)
            if mask.any():
                row = df[mask].iloc[0]
                return {"High": float(row["High"]), "Low": float(row["Low"]),
                        "Close": float(row["Close"]), "Open": float(row["Open"])}
        except Exception as exc:
            logger.debug("Parquet fallback failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Helpers: safe float conversion and formatting
# ---------------------------------------------------------------------------

def _safe_float(v) -> float:
    """Return float(v) or np.nan on failure."""
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _fmt(v, decimals: int = 4) -> str:
    """Format a numeric value for display; return '—' if not finite."""
    f = _safe_float(v)
    if np.isnan(f):
        return "—"
    return f"{f:.{decimals}f}"


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------

class Reconciler:
    """
    Daily reconciliation: load yesterday's signal, fetch actuals, update logs.

    Parameters
    ----------
    signal_dir    : where signal JSON files are stored.
    paper_logger  : PaperTradeLogger instance.
    drift_detector: DriftDetector instance.
    alerter       : Alerter instance (None → no alerts sent).
    """

    def __init__(
        self,
        signal_dir:     Path            = SIGNAL_DIR,
        paper_logger:   Optional[PaperTradeLogger]  = None,
        drift_detector: Optional[DriftDetector]     = None,
        alerter:        Optional[Alerter]            = None,
    ):
        self.signal_dir     = Path(signal_dir)
        self.paper_logger   = paper_logger   or PaperTradeLogger()
        self.drift_detector = drift_detector or DriftDetector()
        self.alerter        = alerter

    # ------------------------------------------------------------------

    def reconcile(self, trade_date: str) -> Dict[str, Any]:
        """
        Reconcile yesterday's signal against actual OHLCV.

        FIX Bug C2: predictions are stored as percentages (e.g. 0.003 = +0.3%).
        We must convert to absolute price before comparing to actual OHLCV or
        compare percentage-to-percentage.  We choose the latter: compute
        actual pct move from prior close and compare to predicted_pct.

        FIX Bug C3 (format-string crash): all formatting uses _fmt() which
        returns '—' for None/NaN, preventing TypeError/ValueError.

        FIX Bug H4 (send_text missing): use send_summary() which dispatches
        text through Alerter.send_failure() with an INFO severity wrapper.
        """
        result: Dict[str, Any] = {
            "trade_date":  trade_date,
            "status":      "ok",
            "errors":      [],
        }

        # 1. Load signal
        signal = _load_signal(trade_date, self.signal_dir)
        if signal is None:
            result["status"] = "missing_signal"
            result["errors"].append(f"No signal file for {trade_date}")
            self._alert_failure(result)
            return result

        # 2. Fetch actuals
        actual = _fetch_actual(trade_date)
        if actual is None:
            result["status"] = "missing_actuals"
            result["errors"].append(f"No OHLCV data for {trade_date}")
            self._alert_failure(result)
            return result

        # 3. Update paper-trade log
        self.paper_logger.log_signal(signal)
        self.paper_logger.log_outcome(trade_date, actual)

        # 4. Compute errors — FIX Bug C2
        # Signal stores predictions as % moves (e.g. 0.003 = +0.3% above prior close).
        # We derive prior_close from the signal, then convert actual OHLCV to %.
        # If prior_close is unavailable we fall back to reporting the raw pct error.

        prior_close = _safe_float(signal.get("prior_close"))

        # Predicted values — prefer _pct keys; fall back to interval midpoints
        pred_high_pct = _safe_float(
            signal.get("predicted_high_pct")
            or signal.get("pred_high")
        )
        pred_low_pct = _safe_float(
            signal.get("predicted_low_pct")
            or signal.get("pred_low")
        )

        actual_high = float(actual["High"])
        actual_low  = float(actual["Low"])

        if not np.isnan(prior_close) and prior_close > 0:
            # Convert actual OHLCV to percentage move for like-for-like comparison
            actual_high_pct = (actual_high - prior_close) / prior_close
            actual_low_pct  = (actual_low  - prior_close) / prior_close

            high_err = abs(actual_high_pct - pred_high_pct) if not np.isnan(pred_high_pct) else np.nan
            low_err  = abs(actual_low_pct  - pred_low_pct)  if not np.isnan(pred_low_pct)  else np.nan

            result["actual_high_pct"]   = round(actual_high_pct, 6)
            result["actual_low_pct"]    = round(actual_low_pct,  6)
        else:
            # prior_close unavailable: compare raw pct predictions to raw prices
            # This is a degraded path — flag it in notes.
            high_err = np.nan
            low_err  = np.nan
            result["errors"].append("prior_close missing — pct error not computed")
            logger.warning("prior_close missing for %s — error metrics degraded", trade_date)

        result["high_error_pct"]  = round(float(high_err), 6) if not np.isnan(high_err) else None
        result["low_error_pct"]   = round(float(low_err),  6) if not np.isnan(low_err)  else None
        result["actual"]          = actual
        result["predicted_high"]  = pred_high_pct
        result["predicted_low"]   = pred_low_pct

        # 5. Update drift detector
        regime = signal.get("regime", "UNKNOWN")

        # drift_detector.update() expects raw % move predictions
        drift_row = self.drift_detector.update(
            signal_date    = trade_date,
            predicted_high = float(pred_high_pct) if not np.isnan(pred_high_pct) else 0.0,
            predicted_low  = float(pred_low_pct)  if not np.isnan(pred_low_pct)  else 0.0,
            actual_high    = actual_high_pct if (not np.isnan(prior_close) and prior_close > 0)
                             else actual_high,
            actual_low     = actual_low_pct  if (not np.isnan(prior_close) and prior_close > 0)
                             else actual_low,
            lower_68_high  = _safe_float(signal.get("lower_68_high", 0) or 0),
            upper_68_high  = _safe_float(signal.get("upper_68_high", 1e9) or 1e9),
            lower_68_low   = _safe_float(signal.get("lower_68_low", 0) or 0),
            upper_68_low   = _safe_float(signal.get("upper_68_low", 1e9) or 1e9),
            lower_90_high  = _safe_float(signal.get("lower_90_high") or signal.get("lower_90")),
            upper_90_high  = _safe_float(signal.get("upper_90_high") or signal.get("upper_90")),
            condor_win     = -1,   # will be set from paper_logger if available
            regime         = regime,
        )
        drift_status = self.drift_detector.check_drift()
        result["drift_status"] = drift_status.value

        # 6. Retrain flag
        self.drift_detector.trigger_retrain_if_needed()

        # 7. Save drift report
        try:
            self.drift_detector.save_report()
        except Exception:
            pass

        # 8. Send alert — FIX Bug C3 (safe formatting) + FIX Bug H4 (no send_text)
        summary_text = (
            f"📊 Reconciliation {trade_date}\n"
            f"  Pred High %: {_fmt(pred_high_pct, 4)}  |  "
            f"Actual High: {_fmt(actual_high, 2)}  |  "
            f"Err: {_fmt(high_err, 4)}\n"
            f"  Pred Low  %: {_fmt(pred_low_pct, 4)}  |  "
            f"Actual Low:  {_fmt(actual_low, 2)}  |  "
            f"Err: {_fmt(low_err, 4)}\n"
            f"  Regime: {regime}  |  Drift: {drift_status.value}\n"
        )
        result["summary_text"] = summary_text

        if self.alerter:
            self._send_summary(summary_text)

        logger.info("Reconciliation complete for %s — drift=%s", trade_date, drift_status.value)
        return result

    # ------------------------------------------------------------------

    def generate_weekly_digest(self, end_date: str) -> Dict[str, Any]:
        """
        Aggregate the 5 trading days ending on end_date (inclusive).
        Returns summary dict and sends via alerter.
        """
        end   = date.fromisoformat(end_date)
        dates = []
        d = end
        while len(dates) < 5:
            if d.weekday() < 5:   # Mon-Fri
                dates.insert(0, d.isoformat())
            d -= timedelta(days=1)

        results = [self.reconcile(dt) for dt in dates]

        errs_h = [r["high_error_pct"] for r in results if r.get("high_error_pct") is not None]
        errs_l = [r["low_error_pct"]  for r in results if r.get("low_error_pct")  is not None]
        drift_history = [r.get("drift_status", "UNKNOWN") for r in results]

        stats = self.paper_logger.summary(lookback_days=5)
        digest = {
            "week_ending":     end_date,
            "days_reconciled": len(results),
            "mae_5d_high":     round(float(np.mean(errs_h)), 6) if errs_h else None,
            "mae_5d_low":      round(float(np.mean(errs_l)), 6) if errs_l else None,
            "drift_history":   drift_history,
            "paper_summary":   stats,
            "generated_at":    datetime.now(timezone.utc).isoformat(),
        }

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / f"weekly_digest_{end_date}.json"
        report_path.write_text(json.dumps(digest, indent=2))

        digest_text = (
            f"📅 Weekly Digest — week ending {end_date}\n"
            f"  5-day MAE High: {digest['mae_5d_high']}\n"
            f"  5-day MAE Low:  {digest['mae_5d_low']}\n"
            f"  Drift history:  {', '.join(drift_history)}\n"
            f"  Win rate:       {stats.get('condor_win_rate', 'N/A')}\n"
            f"  Total P&L:      ${stats.get('condor_total_pnl', 0):,.2f}\n"
        )
        if self.alerter:
            self._send_summary(digest_text)

        logger.info("Weekly digest saved: %s", report_path)
        return digest

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _send_summary(self, text: str) -> None:
        """
        Route informational summaries through send_text() (no 🚨 prefix).
        Falls back to send_failure() if send_text is not available.
        Swallows exceptions so a missing webhook never kills reconciliation.
        """
        if not self.alerter:
            return
        try:
            if hasattr(self.alerter, "send_text"):
                self.alerter.send_text(text)
            else:
                self.alerter.send_failure(f"[INFO] {text}")
        except Exception as exc:
            logger.warning("Summary alert failed: %s", exc)

    def _alert_failure(self, result: Dict[str, Any]) -> None:
        if self.alerter:
            try:
                self.alerter.send_failure(
                    f"Reconciliation failed for {result['trade_date']}: "
                    + "; ".join(result.get("errors", []))
                )
            except Exception:
                pass
