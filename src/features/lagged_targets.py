"""
src/features/lagged_targets.py
================================
Task 9 — Features derived from how the prior day's high/low played out.

LOOKAHEAD-BIAS CONTRACT — CRITICAL
------------------------------------
Every feature value on date T uses ONLY data from T-1 or earlier.
The shift(1) pattern is used throughout and must NEVER be removed.
These features are explicitly about "what happened yesterday" as
predictors of "what will happen today."

Functions
---------
_direction_streak        — Compute signed consecutive direction streaks
compute_lagged_target_features — All lagged-target features
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _direction_streak(series: pd.Series) -> pd.Series:
    """Count consecutive days where each value is greater than the prior value.

    Returns a signed integer series:
      +N means the series has been rising for N consecutive days
      -N means the series has been falling for N consecutive days
       0 means unchanged

    Parameters
    ----------
    series:
        Any numeric Series (e.g. daily High values).

    Returns
    -------
    Integer Series of streak counts.

    Notes
    -----
    The computation uses only past values — no lookahead.
    """
    diff = series.diff()                       # NaN on first row
    direction = np.sign(diff).fillna(0).astype(int)

    streaks = pd.Series(0, index=series.index, dtype=int)
    current_streak = 0

    for i, d in enumerate(direction):
        if d > 0:
            current_streak = max(current_streak + 1, 1)
        elif d < 0:
            current_streak = min(current_streak - 1, -1)
        else:
            current_streak = 0
        streaks.iloc[i] = current_streak

    return streaks


def compute_lagged_target_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all lagged-target features.

    Parameters
    ----------
    df:
        OHLCV DataFrame with columns Open, High, Low, Close.
        Index is a sorted DatetimeIndex.

    Returns
    -------
    DataFrame with the same index and the following columns
    (all computed from T-1 data, so valid for predicting T):

    prev_high_exceedance
        How far above the prior close did the prior high reach:
        (High_{t-1} - Close_{t-1}) / Close_{t-1}

    prev_low_exceedance
        How far below the prior close did the prior low reach:
        (Close_{t-1} - Low_{t-1}) / Close_{t-1}

    range_change
        Whether the daily range is expanding or contracting:
        (Range_{t-1} - Range_{t-2}) / Range_{t-2}

    high_direction_streak
        Signed count of consecutive days where High has been rising
        (positive) or falling (negative), as of T-1.

    low_direction_streak
        Same for Low values.

    inside_day_flag
        1 if yesterday was an inside day:
        High_{t-1} < High_{t-2}  AND  Low_{t-1} > Low_{t-2}

    inside_day_streak
        Count of consecutive inside days ending at T-1.

    outside_day_flag
        1 if yesterday was an outside day:
        High_{t-1} > High_{t-2}  AND  Low_{t-1} < Low_{t-2}
    """
    out = pd.DataFrame(index=df.index)

    # ── Yesterday's exceedances ───────────────────────────────────────────────
    prev_close = df["Close"].shift(1)
    prev_high  = df["High"].shift(1)
    prev_low   = df["Low"].shift(1)

    out["prev_high_exceedance"] = (prev_high - prev_close) / prev_close.replace(0, np.nan)
    out["prev_low_exceedance"]  = (prev_close - prev_low)  / prev_close.replace(0, np.nan)

    # ── Range change ──────────────────────────────────────────────────────────
    prev_range    = (df["High"] - df["Low"]).shift(1)
    prev_range_m1 = (df["High"] - df["Low"]).shift(2)
    out["range_change"] = (
        (prev_range - prev_range_m1) / prev_range_m1.replace(0, np.nan)
    )

    # ── Direction streaks — computed on full series then shifted ──────────────
    # We compute streaks on the raw series (no lookahead in the streak logic
    # itself), then shift(1) so the streak value at row T reflects the streak
    # as of yesterday.
    high_streak = _direction_streak(df["High"])
    low_streak  = _direction_streak(df["Low"])

    out["high_direction_streak"] = high_streak.shift(1)
    out["low_direction_streak"]  = low_streak.shift(1)

    # ── Inside day flag (T-1 vs T-2 data) ────────────────────────────────────
    prev_high_t2 = df["High"].shift(2)
    prev_low_t2  = df["Low"].shift(2)

    inside = ((prev_high < prev_high_t2) & (prev_low > prev_low_t2)).astype(int)
    out["inside_day_flag"] = inside

    # ── Inside day streak ─────────────────────────────────────────────────────
    # Reuse _direction_streak on the inside-day boolean series:
    # A "streak" here is simply a cumulative run of consecutive 1s.
    inside_streak = pd.Series(0, index=df.index, dtype=int)
    run = 0
    for i in range(len(inside)):
        if inside.iloc[i] == 1:
            run += 1
        else:
            run = 0
        inside_streak.iloc[i] = run
    out["inside_day_streak"] = inside_streak

    # ── Outside day flag ──────────────────────────────────────────────────────
    out["outside_day_flag"] = (
        (prev_high > prev_high_t2) & (prev_low < prev_low_t2)
    ).astype(int)

    return out
