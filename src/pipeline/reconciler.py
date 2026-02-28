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

        Parameters
        ----------
        trade_date : YYYY-MM-DD of the trading day whose signal we are checking.

        Returns a results dict with errors, drift status, and alert text.
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

        # 4. Compute errors
        pred_high = signal.get("predicted_high_pct") or signal.get("upper_68_high", np.nan)
        pred_low  = signal.get("predicted_low_pct")  or signal.get("lower_68_low",  np.nan)
        try:
            high_err = abs(float(actual["High"]) - float(pred_high))
            low_err  = abs(float(actual["Low"])  - float(pred_low))
        except (TypeError, ValueError):
            high_err = low_err = np.nan

        result["high_error_pct"] = round(float(high_err), 6) if not np.isnan(high_err) else None
        result["low_error_pct"]  = round(float(low_err),  6) if not np.isnan(low_err)  else None
        result["actual"]         = actual
        result["predicted_high"] = pred_high
        result["predicted_low"]  = pred_low

        # 5. Update drift detector
        regime = signal.get("regime", "UNKNOWN")
        drift_row = self.drift_detector.update(
            signal_date    = trade_date,
            predicted_high = float(pred_high) if pred_high else 0.0,
            predicted_low  = float(pred_low)  if pred_low  else 0.0,
            actual_high    = float(actual["High"]),
            actual_low     = float(actual["Low"]),
            lower_68_high  = float(signal.get("lower_68_high", 0) or 0),
            upper_68_high  = float(signal.get("upper_68_high", 1e9) or 1e9),
            lower_68_low   = float(signal.get("lower_68_low", 0) or 0),
            upper_68_low   = float(signal.get("upper_68_low", 1e9) or 1e9),
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

        # 8. Send alert
        summary_text = (
            f"📊 Reconciliation {trade_date}\n"
            f"  Pred High: {pred_high:.4f}  |  Actual High: {actual['High']:.2f}  "
            f"  Err: {high_err:.4f}\n"
            f"  Pred Low:  {pred_low:.4f}  |  Actual Low:  {actual['Low']:.2f}  "
            f"  Err: {low_err:.4f}\n"
            f"  Regime: {regime}  |  Drift: {drift_status.value}\n"
        )
        result["summary_text"] = summary_text

        if self.alerter:
            try:
                self.alerter.send_text(summary_text)
            except Exception as exc:
                logger.warning("Alert failed: %s", exc)

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
            try:
                self.alerter.send_text(digest_text)
            except Exception as exc:
                logger.warning("Weekly digest alert failed: %s", exc)

        logger.info("Weekly digest saved: %s", report_path)
        return digest

    # ------------------------------------------------------------------

    def _alert_failure(self, result: Dict[str, Any]) -> None:
        if self.alerter:
            try:
                self.alerter.send_failure(
                    f"Reconciliation failed for {result['trade_date']}: "
                    + "; ".join(result.get("errors", []))
                )
            except Exception:
                pass
