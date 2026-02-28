"""
src/features/calendar_features.py
===================================
Task 7 — Calendar and event-proximity features.

LOOKAHEAD-BIAS CONTRACT
-----------------------
All columns use only the date itself (or the calendar built from a
known schedule).  Event schedules (FOMC, OpEx) are forward-looking by
nature but are *publicly known in advance* — the Fed publishes its
schedule a year ahead.  No *realised* data from the future is used.

Functions
---------
compute_calendar_features   — Assemble all calendar columns
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.calendar import load_calendar


def compute_calendar_features(
    df: pd.DataFrame,
    calendar_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute all calendar and event-proximity features.

    Parameters
    ----------
    df:
        OHLCV DataFrame whose index defines the set of trading dates.
    calendar_df:
        Pre-built event calendar (from :func:`src.data.calendar.build_trading_calendar`).
        If None, it is loaded automatically (rebuilt if the CSV is missing).

    Returns
    -------
    DataFrame with the same DatetimeIndex as *df* and the following columns:

    Day-of-week (one-hot, 5 columns):
      is_monday, is_tuesday, is_wednesday, is_thursday, is_friday

    Cyclic seasonal encoding:
      month_sin, month_cos   — sin/cos(2π × month / 12)
      week_sin,  week_cos    — sin/cos(2π × week_of_year / 52)

    Event flags (from calendar):
      is_fomc_day, is_day_before_fomc, is_day_after_fomc
      is_monthly_opex, is_quarterly_opex, is_opex_week

    Countdown integers:
      days_to_next_fomc, days_to_next_opex

    Month/quarter structure:
      is_month_end, is_quarter_end, is_first_trading_day_of_month
      trading_days_remaining_month
    """
    out = pd.DataFrame(index=df.index)

    # ── Day of week (one-hot) ─────────────────────────────────────────────────
    dow = df.index.dayofweek   # 0=Mon … 4=Fri
    out["is_monday"]    = (dow == 0).astype(int)
    out["is_tuesday"]   = (dow == 1).astype(int)
    out["is_wednesday"] = (dow == 2).astype(int)
    out["is_thursday"]  = (dow == 3).astype(int)
    out["is_friday"]    = (dow == 4).astype(int)

    # ── Cyclic month encoding ─────────────────────────────────────────────────
    month = df.index.month.astype(float)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # ── Cyclic week-of-year encoding ──────────────────────────────────────────
    # pandas isocalendar().week returns values 1–53; normalise to 52
    week = df.index.isocalendar().week.astype(float).values
    out["week_sin"] = np.sin(2 * np.pi * week / 52.0)
    out["week_cos"] = np.cos(2 * np.pi * week / 52.0)

    # ── Merge event calendar ──────────────────────────────────────────────────
    if calendar_df is None:
        calendar_df = load_calendar()

    # Reindex calendar to our trading dates (forward-fill for any missing dates)
    event_cols = [
        "is_fomc_day", "is_day_before_fomc", "is_day_after_fomc",
        "is_monthly_opex", "is_quarterly_opex", "is_opex_week",
        "days_to_next_fomc", "days_to_next_opex",
        "is_month_end", "is_quarter_end",
        "is_first_trading_day_of_month",
        "trading_days_remaining_month",
    ]

    # Only merge columns that exist in calendar_df
    available = [c for c in event_cols if c in calendar_df.columns]
    cal_aligned = calendar_df[available].reindex(df.index, method="ffill")

    for col in available:
        out[col] = cal_aligned[col].values

    # Fill any remaining NaN with safe defaults (0 for flags, 25 for countdowns)
    for col in available:
        if col in ("days_to_next_fomc", "days_to_next_opex", "trading_days_remaining_month"):
            out[col] = out[col].fillna(25)
        else:
            out[col] = out[col].fillna(0)

    return out
