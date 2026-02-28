"""
src/features/options_features.py
==================================
Task 8 — Options-market-derived features.

This module is designed for graceful degradation:
  • Core VIX-derived features always work (VIX is free data).
  • GEX data and put/call ratios are optional — if unavailable the
    functions return empty DataFrames and the feature builder skips them.

LOOKAHEAD-BIAS CONTRACT
-----------------------
All rolling windows use data up to and including day T.  VIX data for
day T is the closing VIX level — available after 4 PM ET on day T,
which is valid for predicting day T+1.

Functions
---------
compute_iv_features         — IV Rank, IV Percentile, VIX term structure
compute_putcall_features    — Put/call ratio (optional)
compute_gex_features        — GEX features (optional)
compute_all_options_features— Master assembler
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_iv_features(vix_df: pd.DataFrame) -> pd.DataFrame:
    """IV Rank, IV Percentile, and VIX term-structure proxy.

    Parameters
    ----------
    vix_df:
        DataFrame with a ``Close`` column (the VIX level).

    Returns
    -------
    DataFrame with:
      ``iv_rank_252``            — (VIX - 252d min) / (252d max - 252d min)
      ``iv_percentile_252``      — % of last 252 VIX readings < current VIX
      ``vix_term_structure_proxy``— VIX 5-day change (rising = backwardation)
    """
    vix = vix_df["Close"]
    out = pd.DataFrame(index=vix_df.index)

    roll = vix.rolling(window=252, min_periods=126)
    roll_min = roll.min()
    roll_max = roll.max()
    spread   = (roll_max - roll_min).replace(0, np.nan)

    out["iv_rank_252"] = (vix - roll_min) / spread

    # IV Percentile: fraction of prior 252 readings strictly below current
    out["iv_percentile_252"] = (
        roll.apply(
            lambda x: (x[:-1] < x[-1]).sum() / max(len(x) - 1, 1),
            raw=True,
        )
    )

    # Term structure proxy: positive = VIX rising (backwardation tendency)
    out["vix_term_structure_proxy"] = vix.diff(5)

    return out


def compute_putcall_features(
    df: pd.DataFrame,
    putcall_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Put/call ratio features (optional).

    If ``putcall_df`` is not provided, returns an empty DataFrame.

    Parameters
    ----------
    df:
        OHLCV DataFrame (used only for index alignment).
    putcall_df:
        DataFrame with a ``put_call_ratio`` column, indexed by date.

    Returns
    -------
    DataFrame with put/call features, or empty DataFrame if unavailable.
    """
    if putcall_df is None or putcall_df.empty:
        logger.debug("No put/call data — skipping put_call features.")
        return pd.DataFrame(index=df.index)

    pc = putcall_df["put_call_ratio"].reindex(df.index, method="ffill")
    out = pd.DataFrame(index=df.index)
    out["putcall_ratio"]         = pc
    out["putcall_sma_5"]         = pc.rolling(5, min_periods=3).mean()
    out["putcall_percentile_252"]= (
        pc.rolling(252, min_periods=126)
        .apply(lambda x: (x[:-1] < x[-1]).sum() / max(len(x) - 1, 1), raw=True)
    )
    return out


def compute_gex_features(
    gex_df: pd.DataFrame | None,
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Gamma Exposure (GEX) features (optional).

    If ``gex_df`` is None, returns an empty DataFrame.

    Parameters
    ----------
    gex_df:
        DataFrame with a ``gex`` column (raw gamma exposure), or None.
    index:
        DatetimeIndex for alignment.  Required if gex_df is None.

    Returns
    -------
    DataFrame with GEX features, or empty DataFrame if unavailable.
    """
    fallback_index = index if index is not None else pd.DatetimeIndex([])

    if gex_df is None or gex_df.empty:
        logger.debug("No GEX data — skipping GEX features.")
        return pd.DataFrame(index=fallback_index)

    idx = gex_df.index if index is None else index
    gex = gex_df["gex"].reindex(idx, method="ffill")

    out = pd.DataFrame(index=idx)
    out["gex_raw"]  = gex
    out["gex_sign"] = (gex > 0).astype(int)

    abs_mean_20 = gex.abs().rolling(20, min_periods=10).mean().replace(0, np.nan)
    out["gex_normalized"] = gex / abs_mean_20

    out["gex_percentile_252"] = (
        gex.rolling(252, min_periods=126)
        .apply(lambda x: (x[:-1] < x[-1]).sum() / max(len(x) - 1, 1), raw=True)
    )

    return out


def compute_all_options_features(
    ohlcv_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    gex_df: pd.DataFrame | None = None,
    putcall_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Master assembler for all options-derived features.

    Parameters
    ----------
    ohlcv_df:
        SPX OHLCV DataFrame.
    vix_df:
        VIX OHLCV DataFrame.
    gex_df:
        Optional GEX DataFrame.
    putcall_df:
        Optional put/call ratio DataFrame.

    Returns
    -------
    Merged DataFrame.  Optional sources that are missing contribute
    zero columns (not errors).
    """
    vix_aligned = vix_df.reindex(ohlcv_df.index, method="ffill")
    frames: list[pd.DataFrame] = [
        compute_iv_features(vix_aligned),
        compute_putcall_features(ohlcv_df, putcall_df),
        compute_gex_features(gex_df, index=ohlcv_df.index),
    ]
    # Filter out empty frames before concat
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(index=ohlcv_df.index)
    return pd.concat(non_empty, axis=1)
