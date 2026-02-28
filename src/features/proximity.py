"""
src/features/proximity.py
=========================
Task 4 — Proximity variables (inspired by Jones 2015).

These features capture how far the prior day(s) high/low reached relative
to the closing price and how the intraday range has been evolving.

LOOKAHEAD-BIAS CONTRACT
-----------------------
Every feature value on date T is computed using only OHLCV data from
dates T-1 and earlier — EXCEPT ``open_gap_pct`` which uses today's Open
(requires_open=True in the feature registry).

Functions
---------
compute_proximity_features    — All proximity features in one call
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import PROXIMITY_LAG_PERIODS


def compute_proximity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all proximity features and return as a new DataFrame.

    Parameters
    ----------
    df:
        OHLCV DataFrame with columns Open, High, Low, Close (Volume optional).
        Index must be a sorted DatetimeIndex.

    Returns
    -------
    DataFrame with the same DatetimeIndex and the following columns:

    Lagged high/low relative to prior close (N ∈ {1,2,3,5}):
      ``prev_high_pct_N``  = (High_{t-N} - Close_{t-N}) / Close_{t-N}
      ``prev_low_pct_N``   = (Close_{t-N} - Low_{t-N}) / Close_{t-N}

    Gap and range:
      ``open_gap_pct``        = (Open_t   - Close_{t-1}) / Close_{t-1}
      ``prev_range_pct``      = (High_{t-1} - Low_{t-1}) / Close_{t-1}
      ``rolling_avg_range_5`` = 5-day rolling mean of prev_range_pct
      ``rolling_avg_range_20``= 20-day rolling mean of prev_range_pct

    Notes
    -----
    * All shift operations shift *forward* in time (i.e. shift(N) places
      T-N data on row T), which is the correct direction for prediction.
    * Rows where insufficient history exists will be NaN; the Feature
      Builder (Task 10) drops them.
    """
    out = pd.DataFrame(index=df.index)

    # ── Per-lag high / low pct distance ──────────────────────────────────────
    for lag in PROXIMITY_LAG_PERIODS:
        shifted_close = df["Close"].shift(lag)
        shifted_high  = df["High"].shift(lag)
        shifted_low   = df["Low"].shift(lag)

        out[f"prev_high_pct_{lag}"] = (shifted_high - shifted_close) / shifted_close
        out[f"prev_low_pct_{lag}"]  = (shifted_close - shifted_low)  / shifted_close

    # ── Overnight gap (requires today's Open — post-open only) ───────────────
    prev_close = df["Close"].shift(1)
    out["open_gap_pct"] = (df["Open"] - prev_close) / prev_close

    # ── Prior-day range ───────────────────────────────────────────────────────
    prev_high  = df["High"].shift(1)
    prev_low   = df["Low"].shift(1)
    out["prev_range_pct"] = (prev_high - prev_low) / prev_close

    # ── Rolling average range (expanding over historical data) ────────────────
    # We base the rolling window on prev_range_pct so the roll itself is
    # always over already-lagged data — no lookahead possible.
    out["rolling_avg_range_5"]  = (
        out["prev_range_pct"].rolling(window=5,  min_periods=3).mean()
    )
    out["rolling_avg_range_20"] = (
        out["prev_range_pct"].rolling(window=20, min_periods=10).mean()
    )

    return out
