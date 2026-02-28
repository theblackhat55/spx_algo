"""
src/features/technical.py
==========================
Task 6 — Standard technical indicators: trend, momentum, Bollinger, volume.

All indicators are implemented from scratch using pandas / numpy so that
no optional TA library (pandas-ta, ta-lib) is required at runtime.

LOOKAHEAD-BIAS CONTRACT
-----------------------
All computations use only data available at or before date T (the close
of day T).  EMA / rolling operations are applied directly to the price
series without any shifting; this means the feature value for row T is
computed from data through T, which is correct for predicting the *next*
day (T+1).

Functions
---------
_ema                      — pandas EWM helper
compute_trend_features    — EMA distances + binary direction flags
compute_rsi               — RSI for a single window
compute_momentum_features — RSI, ROC, MACD histogram
compute_bollinger_features— BB position and width
compute_volume_features   — Volume ratios, OBV, OBV ROC
compute_all_technical_features — Master assembler
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import EMA_WINDOWS, RSI_WINDOWS, VOLUME_RATIO_WINDOWS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential moving average with span=window, adjust=False."""
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Trend / EMA features
# ─────────────────────────────────────────────────────────────────────────────

def compute_trend_features(
    df: pd.DataFrame,
    ema_windows: list[int] | None = None,
) -> pd.DataFrame:
    """EMA distances and trend-direction binary flags.

    Parameters
    ----------
    df:
        OHLCV DataFrame.  Must contain ``Close``.
    ema_windows:
        List of EMA periods.  Defaults to ``EMA_WINDOWS`` from settings.

    Returns
    -------
    DataFrame with columns:
      ``ema_dist_N`` = (Close - EMA_N) / EMA_N   for each N
      ``trend_short``   = 1 if EMA_9  > EMA_21 else 0
      ``trend_medium``  = 1 if EMA_21 > EMA_50 else 0
    """
    if ema_windows is None:
        ema_windows = EMA_WINDOWS

    close = df["Close"]
    out   = pd.DataFrame(index=df.index)
    emas: dict[int, pd.Series] = {}

    for w in ema_windows:
        ema_val = _ema(close, w)
        emas[w]  = ema_val
        out[f"ema_dist_{w}"] = (close - ema_val) / ema_val.replace(0, np.nan)

    # Binary trend flags
    if 9 in emas and 21 in emas:
        out["trend_short"]  = (emas[9] > emas[21]).astype(int)
    if 21 in emas and 50 in emas:
        out["trend_medium"] = (emas[21] > emas[50]).astype(int)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# RSI (Wilder's smoothed RSI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, window: int) -> pd.Series:
    """Wilder's RSI using exponential smoothing.

    Parameters
    ----------
    series:
        Price series (Close).
    window:
        Lookback period (e.g. 14).

    Returns
    -------
    Series of RSI values in [0, 100].
    """
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)

    # Wilder smoothing: EWM with alpha = 1/window
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.name = f"rsi_{window}"
    return rsi


# ─────────────────────────────────────────────────────────────────────────────
# Momentum features
# ─────────────────────────────────────────────────────────────────────────────

def compute_momentum_features(
    df: pd.DataFrame,
    rsi_windows: list[int] | None = None,
    roc_windows: list[int] | None = None,
) -> pd.DataFrame:
    """RSI, Rate of Change, and MACD histogram.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    rsi_windows:
        RSI lookback periods.  Defaults to ``RSI_WINDOWS`` from settings.
    roc_windows:
        ROC lookback periods.  Defaults to [5, 10, 20].

    Returns
    -------
    DataFrame with RSI, ROC, and MACD histogram columns.
    """
    if rsi_windows is None:
        rsi_windows = RSI_WINDOWS
    if roc_windows is None:
        roc_windows = [5, 10, 20]

    close = df["Close"]
    out   = pd.DataFrame(index=df.index)

    # ── RSI ───────────────────────────────────────────────────────────────────
    for w in rsi_windows:
        out[f"rsi_{w}"] = compute_rsi(close, w)

    # ── Rate of Change ────────────────────────────────────────────────────────
    for w in roc_windows:
        out[f"roc_{w}"] = close.pct_change(periods=w)

    # ── MACD histogram (12, 26, 9) ────────────────────────────────────────────
    ema_fast  = _ema(close, 12)
    ema_slow  = _ema(close, 26)
    macd_line = ema_fast - ema_slow
    signal    = _ema(macd_line, 9)
    out["macd_histogram"] = macd_line - signal

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Bollinger Bands
# ─────────────────────────────────────────────────────────────────────────────

def compute_bollinger_features(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """20-period Bollinger Bands — position and width.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    window:
        Rolling period (default 20).
    num_std:
        Number of standard deviations (default 2.0).

    Returns
    -------
    DataFrame with:
      ``bb_position`` = (Close - Lower) / (Upper - Lower)
         • 0 = at lower band, 1 = at upper band
         • Outside [0,1] when price is beyond the bands
      ``bb_width``    = (Upper - Lower) / Middle
    """
    close    = df["Close"]
    roll     = close.rolling(window=window, min_periods=window // 2)
    middle   = roll.mean()
    std      = roll.std()

    upper = middle + num_std * std
    lower = middle - num_std * std
    band_width = (upper - lower).replace(0, np.nan)

    out = pd.DataFrame(index=df.index)
    out["bb_position"] = (close - lower) / band_width
    out["bb_width"]    = band_width / middle.replace(0, np.nan)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Volume features
# ─────────────────────────────────────────────────────────────────────────────

def compute_volume_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Volume ratios, On-Balance Volume (OBV), and OBV momentum.

    Parameters
    ----------
    df:
        OHLCV DataFrame.  Must contain ``Volume`` column.
        If ``Volume`` is missing, returns an empty DataFrame.
    windows:
        Rolling windows for volume ratio.  Defaults to
        ``VOLUME_RATIO_WINDOWS`` from settings.

    Returns
    -------
    DataFrame with volume features.
    """
    if windows is None:
        windows = VOLUME_RATIO_WINDOWS

    if "Volume" not in df.columns:
        return pd.DataFrame(index=df.index)

    vol = df["Volume"]
    close = df["Close"]
    out   = pd.DataFrame(index=df.index)

    # ── Volume ratios ─────────────────────────────────────────────────────────
    for w in windows:
        roll_mean = vol.rolling(window=w, min_periods=w // 2).mean()
        out[f"volume_ratio_{w}"] = vol / roll_mean.replace(0, np.nan)

    # ── On-Balance Volume ─────────────────────────────────────────────────────
    direction = np.sign(close.diff())           # +1 up, -1 down, 0 unchanged
    direction.iloc[0] = 0
    obv = (vol * direction).cumsum()

    # ── OBV rate of change (10-day) ───────────────────────────────────────────
    out["obv_roc_10"] = obv.pct_change(periods=10)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Master assembler
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and merge all technical features.

    Parameters
    ----------
    df:
        SPX OHLCV DataFrame (daily, DatetimeIndex, sorted ascending).

    Returns
    -------
    Single merged DataFrame with all technical feature columns.
    """
    frames = [
        compute_trend_features(df),
        compute_momentum_features(df),
        compute_bollinger_features(df),
        compute_volume_features(df),
    ]
    return pd.concat(frames, axis=1)
