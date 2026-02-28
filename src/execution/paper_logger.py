"""
src/execution/paper_logger.py
================================
Task 35 — Paper-Trade Execution Logger.

Records every daily signal and its next-day outcome in a structured CSV log.
Provides summary statistics and HTML report generation.

Classes
-------
PaperTradeLogger   Core logger with signal/outcome tracking.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PAPER_LOG_PATH = Path("output/trades/paper_trade_log.csv")

# All columns in the CSV, in order
LOG_COLUMNS = [
    # Signal fields
    "date",
    "predicted_high",
    "predicted_low",
    "predicted_direction",
    "regime",
    "call_strike",
    "put_strike",
    "long_call_strike",
    "long_put_strike",
    "lower_68_high",
    "upper_68_high",
    "lower_68_low",
    "upper_68_low",
    "lower_90_high",
    "upper_90_high",
    "lower_90_low",
    "upper_90_low",
    "data_quality",
    "model_version_hash",
    "prior_close",
    # Outcome fields (filled next day)
    "actual_high",
    "actual_low",
    "actual_close",
    "high_error_pct",
    "low_error_pct",
    "direction_correct",
    "condor_result",        # "win" / "loss" / "skip"
    "condor_pnl",
    "interval_68_covered_high",
    "interval_68_covered_low",
    "interval_90_covered_high",
    "interval_90_covered_low",
]


class PaperTradeLogger:
    """
    Structured log of paper-trade signals and their next-day outcomes.

    Parameters
    ----------
    log_path : Path   Path to CSV log file.  Created on first write.
    """

    def __init__(self, log_path: Path = PAPER_LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core write operations
    # ------------------------------------------------------------------

    def log_signal(self, signal: Dict[str, Any]) -> None:
        """
        Append a signal row.  If the date already exists, the row is
        updated (idempotent: logging the same date twice does not duplicate).
        """
        date_str = str(signal.get("signal_date") or signal.get("date", ""))
        row = {col: "" for col in LOG_COLUMNS}
        row.update({
            "date":                str(date_str),
            "predicted_high":      signal.get("predicted_high_pct", "") or signal.get("pred_high_pct", ""),
            "predicted_low":       signal.get("predicted_low_pct",  "") or signal.get("pred_low_pct",  ""),
            "predicted_direction": signal.get("direction", ""),
            "regime":              signal.get("regime", ""),
            # FIX Bug N5: FullSignal uses ic_short_call/ic_long_call etc.; DailySignal uses call_strike/put_strike.
            "call_strike":         signal.get("ic_short_call",   "") or signal.get("call_strike",      ""),
            "put_strike":          signal.get("ic_short_put",    "") or signal.get("put_strike",       ""),
            "long_call_strike":    signal.get("ic_long_call",    "") or signal.get("long_call_strike", ""),
            "long_put_strike":     signal.get("ic_long_put",     "") or signal.get("long_put_strike",  ""),
            # FIX Bug N5: FullSignal uses conf_68_high_lo / conf_68_high_hi etc.
            "lower_68_high":       signal.get("conf_68_high_lo", "") or signal.get("lower_68_high",   ""),
            "upper_68_high":       signal.get("conf_68_high_hi", "") or signal.get("upper_68_high",   ""),
            "lower_68_low":        signal.get("conf_68_low_lo",  "") or signal.get("lower_68_low",    ""),
            "upper_68_low":        signal.get("conf_68_low_hi",  "") or signal.get("upper_68_low",    ""),
            "lower_90_high":       signal.get("conf_90_high_lo", "") or signal.get("lower_90_high",   ""),
            "upper_90_high":       signal.get("conf_90_high_hi", "") or signal.get("upper_90_high",   ""),
            "lower_90_low":        signal.get("conf_90_low_lo",  "") or signal.get("lower_90_low",    ""),
            "upper_90_low":        signal.get("conf_90_low_hi",  "") or signal.get("upper_90_low",    ""),
            "data_quality":        signal.get("data_quality", "ok"),
            "model_version_hash":  signal.get("model_version_hash", ""),
            "prior_close":         signal.get("prior_close", ""),
        })
        self._upsert(date_str, row)

    def log_outcome(self, date_str: str, actual_ohlcv: Dict[str, float]) -> None:
        """
        Fill in the outcome columns for the given date.
        The signal row must already exist.
        """
        df = self._read_df()
        if df.empty or date_str not in df["date"].values:
            logger.warning("log_outcome: no signal row found for %s", date_str)
            return

        idx = df.index[df["date"] == date_str][0]

        actual_high  = float(actual_ohlcv.get("High",  np.nan))
        actual_low   = float(actual_ohlcv.get("Low",   np.nan))
        actual_close = float(actual_ohlcv.get("Close", np.nan))

        # Parse signal values
        def _f(col):
            try:
                return float(df.at[idx, col])
            except (ValueError, TypeError):
                return np.nan

        pred_high = _f("predicted_high")
        pred_low  = _f("predicted_low")
        pred_dir  = df.at[idx, "predicted_direction"]

        high_err  = abs(actual_high - pred_high) if not np.isnan(pred_high) else np.nan
        low_err   = abs(actual_low  - pred_low)  if not np.isnan(pred_low)  else np.nan

        dir_correct = (
            bool((pred_dir == "UP" and actual_close > _f("predicted_low")) or
                 (pred_dir == "DOWN" and actual_close < _f("predicted_high")))
            if pred_dir else np.nan
        )

        # Condor result: win if both short strikes unbreached
        # P&L formula (Bug 7 & 8 fix):
        #   Credit approximation uses same formula as BacktestEngine:
        #     credit_pts = (vix_estimate/100) * prior_close / sqrt(252) * 0.12
        #   Without live VIX in the logger we use a conservative fixed estimate.
        #   Win  P&L = credit_pts - friction (not a spurious interval subtraction)
        #   Loss P&L = credit_pts - intrusion - friction  (credit offsets the loss)
        call_k = _f("call_strike")
        put_k  = _f("put_strike")
        if np.isnan(call_k) or np.isnan(put_k) or df.at[idx, "regime"] == "RED":
            condor_result = "skip"
            condor_pnl    = 0.0
        else:
            # Estimate credit: use prior close from signal if available,
            # otherwise fall back to a generic 0.5% credit (conservative).
            # Use actual prior_close persisted from the signal.
            # Fallback chain: prior_close column → actual_close from prior row
            # → generic 0.5% credit (5.0 pts) if neither is available.
            prior_close = _f("prior_close")
            if np.isnan(prior_close) or prior_close <= 0:
                # Second fallback: use the previous row's actual_close if present
                row_pos = df.index.get_loc(idx)
                if row_pos > 0:
                    prev_close = pd.to_numeric(
                        df.iloc[row_pos - 1].get("actual_close", np.nan), errors="coerce"
                    )
                    prior_close = float(prev_close) if not np.isnan(prev_close) else np.nan
            if np.isnan(prior_close) or prior_close <= 0:
                # Conservative fallback: 0.5% of a typical SPX level
                credit_pts = 5.0
            else:
                # credit ~ 12% of 1-day expected move @ VIX=18 (conservative)
                credit_pts = (18 / 100.0) * prior_close / (252 ** 0.5) * 0.12
            friction = 0.40   # $0.10 per leg × 4 legs

            if actual_high <= call_k and actual_low >= put_k:
                condor_result = "win"
                condor_pnl    = (credit_pts - friction) * 100   # per-contract multiplier
            else:
                # Cap each leg intrusion at wing width (max loss per leg = wing_width - credit)
                long_call = _f("long_call_strike")
                long_put  = _f("long_put_strike")
                wing_width = 20.0   # default 20-pt wing
                if not np.isnan(long_call) and long_call > call_k:
                    wing_width = min(wing_width, long_call - call_k)
                if not np.isnan(long_put) and long_put < put_k:
                    wing_width = min(wing_width, put_k - long_put)
                call_intrusion = min(max(actual_high - call_k, 0.0), wing_width)
                put_intrusion  = min(max(put_k - actual_low,  0.0), wing_width)
                # FIX Bug N3: match engine.py — intrusions are ADDITIVE (both legs can
                # breach on the same day). Using max() understated losses on double-breach days.
                total_intrusion = call_intrusion + put_intrusion
                condor_result   = "loss"
                condor_pnl      = (credit_pts - total_intrusion - friction) * 100
                # Max loss = both wings fully breached (2 × wing_width − credit)
                max_loss        = -(2 * wing_width - credit_pts) * 100
                condor_pnl      = max(condor_pnl, max_loss)

        # Coverage flags
        def _cov(lo_col, hi_col, val):
            lo, hi = _f(lo_col), _f(hi_col)
            if np.isnan(lo) or np.isnan(hi):
                return np.nan
            return int(lo <= val <= hi)

        df.at[idx, "actual_high"]             = actual_high
        df.at[idx, "actual_low"]              = actual_low
        df.at[idx, "actual_close"]            = actual_close
        df.at[idx, "high_error_pct"]          = round(high_err,  6) if not np.isnan(high_err)  else ""
        df.at[idx, "low_error_pct"]           = round(low_err,   6) if not np.isnan(low_err)   else ""
        df.at[idx, "direction_correct"]       = dir_correct if not isinstance(dir_correct, float) else ""
        df.at[idx, "condor_result"]           = condor_result
        df.at[idx, "condor_pnl"]              = round(condor_pnl, 2)
        df.at[idx, "interval_68_covered_high"] = _cov("lower_68_high", "upper_68_high", actual_high)
        df.at[idx, "interval_68_covered_low"]  = _cov("lower_68_low",  "upper_68_low",  actual_low)
        df.at[idx, "interval_90_covered_high"] = _cov("lower_90_high", "upper_90_high", actual_high)
        df.at[idx, "interval_90_covered_low"]  = _cov("lower_90_low",  "upper_90_low",  actual_low)

        self._write_df(df)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def summary(self, lookback_days: int = 63) -> Dict[str, Any]:
        """Return a dict of performance statistics over the last N days."""
        df = self._read_df()
        if df.empty:
            return {"rows": 0}

        # Filter to rows with outcome
        has_out = df["actual_high"] != ""
        df_out  = df[has_out].tail(lookback_days).copy()

        if df_out.empty:
            return {"rows": 0, "lookback_days": lookback_days}

        def _col_float(col):
            return pd.to_numeric(df_out[col], errors="coerce")

        mae_high = _col_float("high_error_pct").mean()
        mae_low  = _col_float("low_error_pct").mean()
        dir_acc  = (df_out["direction_correct"] == "True").mean()
        wins     = (df_out["condor_result"] == "win").sum()
        losses   = (df_out["condor_result"] == "loss").sum()
        skips    = (df_out["condor_result"] == "skip").sum()
        win_rate = wins / max(wins + losses, 1)
        total_pnl = _col_float("condor_pnl").sum()

        cov_68h = _col_float("interval_68_covered_high").mean()
        cov_68l = _col_float("interval_68_covered_low").mean()
        cov_90h = _col_float("interval_90_covered_high").mean()
        cov_90l = _col_float("interval_90_covered_low").mean()

        regime_counts = df_out["regime"].value_counts().to_dict()

        return {
            "rows":              len(df_out),
            "lookback_days":     lookback_days,
            "MAE_high":          round(float(mae_high), 6) if not np.isnan(mae_high) else None,
            "MAE_low":           round(float(mae_low),  6) if not np.isnan(mae_low)  else None,
            "directional_accuracy": round(float(dir_acc), 4),
            "condor_win_rate":   round(float(win_rate),  4),
            "condor_wins":       int(wins),
            "condor_losses":     int(losses),
            "condor_total_pnl":  round(float(total_pnl), 2),
            "coverage_68_high":  round(float(cov_68h), 4) if not np.isnan(cov_68h) else None,
            "coverage_68_low":   round(float(cov_68l), 4) if not np.isnan(cov_68l) else None,
            "coverage_90_high":  round(float(cov_90h), 4) if not np.isnan(cov_90h) else None,
            "coverage_90_low":   round(float(cov_90l), 4) if not np.isnan(cov_90l) else None,
            "days_skipped":      int(skips),
            "regime_counts":     regime_counts,
            "generated_at":      datetime.now(timezone.utc).isoformat(),
        }

    def export_html_report(self, lookback_days: int = 63) -> str:
        """
        Generate a self-contained HTML report with equity curve and summary table.
        Returns the HTML string.
        """
        stats  = self.summary(lookback_days)
        df     = self._read_df()
        has_out = df["actual_high"] != ""
        df_out  = df[has_out].tail(lookback_days).copy()

        pnl_values = pd.to_numeric(df_out["condor_pnl"], errors="coerce").fillna(0)
        cum_pnl    = pnl_values.cumsum().tolist()
        dates_str  = df_out["date"].tolist()

        # Inline SVG equity curve
        svg = _make_svg_equity_curve(dates_str, cum_pnl)

        stats_rows = "".join(
            f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
            for k, v in stats.items()
            if k not in ("generated_at", "regime_counts")
        )

        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>SPX Algo Paper Trade Report</title>
<style>
  body {{ font-family: sans-serif; max-width: 900px; margin: 40px auto; color: #333; }}
  h1   {{ color: #1a5276; }}
  table{{ border-collapse: collapse; width: 100%; }}
  td,th{{ border: 1px solid #ccc; padding: 8px 12px; }}
  th   {{ background: #1a5276; color: white; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
</style></head><body>
<h1>SPX Iron-Condor — Paper Trade Report</h1>
<p>Generated: {datetime.now(timezone.utc).isoformat()}</p>
<h2>Equity Curve ({lookback_days}-day lookback)</h2>
{svg}
<h2>Summary Statistics</h2>
<table><tr><th>Metric</th><th>Value</th></tr>
{stats_rows}
</table>
</body></html>"""
        return html

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _read_df(self) -> pd.DataFrame:
        if not self.log_path.exists():
            return pd.DataFrame(columns=LOG_COLUMNS)
        try:
            return pd.read_csv(self.log_path, dtype=str).fillna("")
        except Exception as exc:
            logger.warning("Could not read paper log: %s", exc)
            return pd.DataFrame(columns=LOG_COLUMNS)

    def _write_df(self, df: pd.DataFrame) -> None:
        df.to_csv(self.log_path, index=False)

    def _upsert(self, date_str: str, row: Dict[str, Any]) -> None:
        """Insert or update a row by date key."""
        df = self._read_df()
        if not df.empty and date_str in df["date"].values:
            idx = df.index[df["date"] == date_str][0]
            for col in LOG_COLUMNS:
                if row.get(col, "") != "":
                    df.at[idx, col] = row[col]
        else:
            new_row = pd.DataFrame([{col: row.get(col, "") for col in LOG_COLUMNS}])
            df = pd.concat([df, new_row], ignore_index=True)
        self._write_df(df)


# ---------------------------------------------------------------------------
# SVG equity curve helper
# ---------------------------------------------------------------------------

def _make_svg_equity_curve(dates: List[str], cum_pnl: List[float]) -> str:
    if not cum_pnl:
        return "<p>No data</p>"
    W, H, PAD = 800, 250, 40
    mn, mx = min(cum_pnl), max(cum_pnl)
    span = mx - mn or 1
    n = len(cum_pnl)

    def _x(i):
        return PAD + (i / max(n - 1, 1)) * (W - 2 * PAD)

    def _y(v):
        return H - PAD - ((v - mn) / span) * (H - 2 * PAD)

    pts = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(cum_pnl))
    colour = "#2ecc71" if cum_pnl[-1] >= 0 else "#e74c3c"
    zero_y = _y(0)
    return (
        f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">'
        f'<line x1="{PAD}" y1="{zero_y:.1f}" x2="{W-PAD}" y2="{zero_y:.1f}" '
        f'stroke="#aaa" stroke-dasharray="4"/>'
        f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"/>'
        f'</svg>'
    )
