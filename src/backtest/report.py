"""
src/backtest/report.py
=======================
Task 21 — Backtest performance reporting.

Metrics produced
----------------
Returns & P&L
    total_pnl_dollars, total_trades, win_rate, avg_win, avg_loss,
    profit_factor, expected_value_per_trade

Risk-adjusted
    annualised_return, annualised_vol, sharpe_ratio, sortino_ratio,
    calmar_ratio, max_drawdown_pct, max_drawdown_dollars

Per-regime
    breakdown by GREEN / YELLOW / RED with independent stats

Monthly
    monthly P&L table (pivot)

Baseline comparison
    comparison against all three naive baselines from baselines.py
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.calibration.regime import REGIME_LABELS

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Core performance metrics
# ---------------------------------------------------------------------------

def compute_performance(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute all performance metrics from a trades DataFrame
    (output of IronCondorEngine.run()).

    Returns
    -------
    dict  with scalar metrics + sub-DataFrames (monthly, regime_breakdown).
    """
    active = trades[~trades["skipped"]].copy()

    if len(active) == 0:
        logger.warning("No active trades — empty performance dict.")
        return {"error": "no_active_trades"}

    pnl   = active["net_pnl_dollars"]
    n     = len(pnl)

    # ── Basic ─────────────────────────────────────────────────────────────
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    total_pnl   = float(pnl.sum())
    win_rate    = float(len(wins) / n)
    avg_win     = float(wins.mean())  if len(wins)   > 0 else 0.0
    avg_loss    = float(losses.mean()) if len(losses) > 0 else 0.0

    gross_profit = float(wins.sum())
    gross_loss   = float(abs(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    ev_per_trade  = float(pnl.mean())

    # ── Drawdown ──────────────────────────────────────────────────────────
    cum_pnl   = pnl.cumsum()
    running_max = cum_pnl.cummax()
    drawdowns   = cum_pnl - running_max

    max_dd_dollars = float(drawdowns.min())
    if running_max.max() != 0:
        max_dd_pct = float((drawdowns / running_max.replace(0, np.nan)).min())
    else:
        max_dd_pct = float(drawdowns.min() / max(1.0, abs(total_pnl)) * -1)

    # ── Annualised stats ──────────────────────────────────────────────────
    n_years = n / TRADING_DAYS_PER_YEAR
    ann_return = total_pnl / max(n_years, 1e-6)

    daily_pnl  = pnl
    ann_vol    = float(daily_pnl.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe     = ann_return / ann_vol if ann_vol > 0 else 0.0

    downside_std = float(
        daily_pnl[daily_pnl < 0].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    ) if (daily_pnl < 0).any() else 1.0
    sortino = ann_return / downside_std if downside_std > 0 else 0.0

    calmar  = ann_return / abs(max_dd_dollars) if max_dd_dollars != 0 else 0.0

    # ── Per-regime breakdown ──────────────────────────────────────────────
    regime_rows = []
    for reg_val, label in REGIME_LABELS.items():
        sub = active[active["regime"] == reg_val]["net_pnl_dollars"]
        if len(sub) == 0:
            continue
        regime_rows.append({
            "regime":       label,
            "n_trades":     len(sub),
            "win_rate":     float((sub > 0).mean()),
            "total_pnl":    float(sub.sum()),
            "avg_pnl":      float(sub.mean()),
        })
    regime_df = pd.DataFrame(regime_rows).set_index("regime") if regime_rows else pd.DataFrame()

    # ── Monthly P&L ───────────────────────────────────────────────────────
    monthly = (
        active["net_pnl_dollars"]
        .resample("ME").sum()
        .rename("pnl")
        .to_frame()
    )
    monthly["year"]  = monthly.index.year
    monthly["month"] = monthly.index.month
    monthly_pivot = monthly.pivot(index="year", columns="month", values="pnl")
    monthly_pivot.columns = [
        ["Jan","Feb","Mar","Apr","May","Jun",
         "Jul","Aug","Sep","Oct","Nov","Dec"][m - 1]
        for m in monthly_pivot.columns
    ]
    monthly_pivot["YTD"] = monthly_pivot.sum(axis=1)

    # ── Call vs Put breakdown ─────────────────────────────────────────────
    call_breach_rate = float((active["call_intrusion"] > 0).mean())
    put_breach_rate  = float((active["put_intrusion"]  > 0).mean())
    both_breach_rate = float(
        ((active["call_intrusion"] > 0) & (active["put_intrusion"] > 0)).mean()
    )

    return {
        # scalars
        "total_pnl_dollars":       total_pnl,
        "total_trades":            n,
        "win_rate":                win_rate,
        "avg_win_dollars":         avg_win,
        "avg_loss_dollars":        avg_loss,
        "profit_factor":           profit_factor,
        "ev_per_trade_dollars":    ev_per_trade,
        "annualised_return_dollars": ann_return,
        "annualised_vol_dollars":  ann_vol,
        "sharpe_ratio":            sharpe,
        "sortino_ratio":           sortino,
        "calmar_ratio":            calmar,
        "max_drawdown_dollars":    max_dd_dollars,
        "max_drawdown_pct":        max_dd_pct,
        "call_breach_rate":        call_breach_rate,
        "put_breach_rate":         put_breach_rate,
        "both_breach_rate":        both_breach_rate,
        # DataFrames
        "regime_breakdown":        regime_df,
        "monthly_pnl":             monthly_pivot,
        "cumulative_pnl":          cum_pnl,
    }


# ---------------------------------------------------------------------------
# Skipped-day analysis
# ---------------------------------------------------------------------------

def skip_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise why days were skipped — useful for verifying regime filter
    is working correctly in the backtest.
    """
    skipped = trades[trades["skipped"]]
    if len(skipped) == 0:
        return pd.DataFrame(columns=["skip_reason", "count", "pct"])

    counts = (
        skipped["skip_reason"]
        .value_counts()
        .rename_axis("skip_reason")
        .reset_index(name="count")
    )
    counts["pct"] = counts["count"] / len(trades)
    return counts


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(perf: Dict[str, Any], title: str = "Backtest Results") -> None:
    """Print a clean human-readable performance summary."""
    if "error" in perf:
        print(f"[{title}] ERROR: {perf['error']}")
        return

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Total P&L     : ${perf['total_pnl_dollars']:>10,.2f}")
    print(f"  Trades        : {perf['total_trades']:>10,d}")
    print(f"  Win Rate      : {perf['win_rate']:>10.1%}")
    print(f"  Avg Win       : ${perf['avg_win_dollars']:>10,.2f}")
    print(f"  Avg Loss      : ${perf['avg_loss_dollars']:>10,.2f}")
    print(f"  Profit Factor : {perf['profit_factor']:>10.2f}")
    print(f"  EV / Trade    : ${perf['ev_per_trade_dollars']:>10,.2f}")
    print(sep)
    print(f"  Sharpe Ratio  : {perf['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio : {perf['sortino_ratio']:>10.2f}")
    print(f"  Calmar Ratio  : {perf['calmar_ratio']:>10.2f}")
    print(f"  Max Drawdown  : ${perf['max_drawdown_dollars']:>10,.2f}  "
          f"({perf['max_drawdown_pct']:.1%})")
    print(sep)
    print(f"  Call Breach % : {perf['call_breach_rate']:>10.1%}")
    print(f"  Put  Breach % : {perf['put_breach_rate']:>10.1%}")
    print(f"  Both Breach % : {perf['both_breach_rate']:>10.1%}")
    print(sep)

    if not perf["regime_breakdown"].empty:
        print("\n  Per-Regime:")
        print(perf["regime_breakdown"].to_string())
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def compare_to_baselines(
    perf: Dict[str, Any],
    baseline_metrics: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Compare the model-driven iron condor results against naive baselines.

    Parameters
    ----------
    perf             : Output of compute_performance().
    baseline_metrics : {name: {mae_high, mae_low}} from baselines.evaluate().

    Returns
    -------
    DataFrame summarising model vs. each baseline.
    """
    rows = [{"strategy": "model",
             "win_rate":  perf.get("win_rate", np.nan),
             "sharpe":    perf.get("sharpe_ratio", np.nan),
             "total_pnl": perf.get("total_pnl_dollars", np.nan),
             "mae_high":  np.nan,
             "mae_low":   np.nan}]

    for name, bm in baseline_metrics.items():
        rows.append({
            "strategy": name,
            "win_rate":  np.nan,
            "sharpe":    np.nan,
            "total_pnl": np.nan,
            "mae_high":  bm.get("mae_high", np.nan),
            "mae_low":   bm.get("mae_low",  np.nan),
        })

    return pd.DataFrame(rows).set_index("strategy")
