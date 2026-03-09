"""
src/features/options_features.py
==================================
Task 8 — Options-market-derived features.

This module is designed for graceful degradation:
  • Core VIX-derived features always work (VIX is free data).
  • GEX data, put/call ratios, and summarized options features are optional.
  • Missing optional data returns empty DataFrames and the feature builder
    skips them gracefully.

LOOKAHEAD-BIAS CONTRACT
-----------------------
All rolling windows use data up to and including day T. VIX / options summary
data for day T are assumed to be end-of-day values, valid for predicting T+1.

Functions
---------
compute_iv_features             — IV Rank, IV Percentile, VIX term structure
compute_putcall_features        — Put/call ratio (optional)
compute_gex_features            — GEX features (optional)
compute_options_summary_features— external summarized options features (optional)
compute_all_options_features    — Master assembler
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


OPTIONS_SUMMARY_COLUMNS = [
    "atm_implied_move_1d",
    "front_iv_atm",
    "front_iv_change_1d",
    "put_call_skew_25d",
    "term_structure_slope_front_second",
]


def compute_iv_features(vix_df: pd.DataFrame) -> pd.DataFrame:
    """IV Rank, IV Percentile, and VIX term-structure proxy."""
    vix = vix_df["Close"]
    out = pd.DataFrame(index=vix_df.index)

    roll = vix.rolling(window=252, min_periods=126)
    roll_min = roll.min()
    roll_max = roll.max()
    spread = (roll_max - roll_min).replace(0, np.nan)

    out["iv_rank_252"] = (vix - roll_min) / spread

    out["iv_percentile_252"] = (
        roll.apply(
            lambda x: (x[:-1] < x[-1]).sum() / max(len(x) - 1, 1),
            raw=True,
        )
    )

    # Proxy only; richer term-structure features may come from options_daily.csv
    out["vix_term_structure_proxy"] = vix.diff(5)

    return out


def compute_putcall_features(
    df: pd.DataFrame,
    putcall_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Put/call ratio features (optional)."""
    if putcall_df is None or putcall_df.empty:
        logger.debug("No put/call data — skipping put_call features.")
        return pd.DataFrame(index=df.index)

    pc = putcall_df["put_call_ratio"].reindex(df.index, method="ffill")
    out = pd.DataFrame(index=df.index)
    out["putcall_ratio"] = pc
    out["putcall_sma_5"] = pc.rolling(5, min_periods=3).mean()
    out["putcall_percentile_252"] = (
        pc.rolling(252, min_periods=126)
        .apply(lambda x: (x[:-1] < x[-1]).sum() / max(len(x) - 1, 1), raw=True)
    )
    return out


def compute_gex_features(
    gex_df: pd.DataFrame | None,
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Gamma Exposure (GEX) features (optional)."""
    fallback_index = index if index is not None else pd.DatetimeIndex([])

    if gex_df is None or gex_df.empty:
        logger.debug("No GEX data — skipping GEX features.")
        return pd.DataFrame(index=fallback_index)

    idx = gex_df.index if index is None else index
    gex = gex_df["gex"].reindex(idx, method="ffill")

    out = pd.DataFrame(index=idx)
    out["gex_raw"] = gex
    out["gex_sign"] = (gex > 0).astype(int)

    abs_mean_20 = gex.abs().rolling(20, min_periods=10).mean().replace(0, np.nan)
    out["gex_normalized"] = gex / abs_mean_20

    out["gex_percentile_252"] = (
        gex.rolling(252, min_periods=126)
        .apply(lambda x: (x[:-1] < x[-1]).sum() / max(len(x) - 1, 1), raw=True)
    )

    return out


def compute_options_summary_features(
    ohlcv_df: pd.DataFrame,
    options_summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Optional summarized options-derived daily features.

    Expected columns in options_summary_df:
      - atm_implied_move_1d
      - front_iv_atm
      - front_iv_change_1d
      - put_call_skew_25d
      - term_structure_slope_front_second

    Missing columns are ignored. Missing file/data yields an empty frame.
    """
    if options_summary_df is None or options_summary_df.empty:
        logger.debug("No options summary data — skipping summarized options features.")
        return pd.DataFrame(index=ohlcv_df.index)

    df = options_summary_df.copy()
    df.index = pd.to_datetime(df.index)

    available = [c for c in OPTIONS_SUMMARY_COLUMNS if c in df.columns]
    if not available:
        logger.debug("Options summary data present but no recognized columns found.")
        return pd.DataFrame(index=ohlcv_df.index)

    out = df[available].reindex(ohlcv_df.index, method="ffill")
    return out


def compute_all_options_features(
    ohlcv_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    gex_df: pd.DataFrame | None = None,
    putcall_df: pd.DataFrame | None = None,
    options_summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Master assembler for all options-derived features."""
    vix_aligned = vix_df.reindex(ohlcv_df.index, method="ffill")

    frames: list[pd.DataFrame] = [
        compute_iv_features(vix_aligned),
        compute_putcall_features(ohlcv_df, putcall_df),
        compute_gex_features(gex_df, index=ohlcv_df.index),
        compute_options_summary_features(ohlcv_df, options_summary_df),
    ]

    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(index=ohlcv_df.index)
    return pd.concat(non_empty, axis=1)
