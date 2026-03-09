"""
src/targets/ohlc_targets.py
===========================

OHLC component target builder for next-day SPX forecasting.

This module creates structurally meaningful next-day targets:

1. target_gap_ret
   next open relative to prior close

2. target_high_from_open
   next high relative to next open

3. target_low_from_open
   next low relative to next open (positive number)

4. target_close_from_open
   next close relative to next open

These targets are designed to be easier to model than raw OHLC levels and
more reusable across downstream strategies.

No existing production path is modified by this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_OHLC_COLUMNS = ("Open", "High", "Low", "Close")


def _validate_ohlc_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_OHLC_COLUMNS) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")


def build_ohlc_component_targets(
    price_df: pd.DataFrame,
    dropna: bool = False,
) -> pd.DataFrame:
    """
    Build next-day OHLC component targets from a daily OHLC dataframe.

    Parameters
    ----------
    price_df
        DataFrame indexed by trading date with columns Open, High, Low, Close.
    dropna
        If True, drop rows with NaN targets (typically the final row).

    Returns
    -------
    pd.DataFrame
        Same index as input with columns:
        - target_gap_ret
        - target_high_from_open
        - target_low_from_open
        - target_close_from_open
        - target_range_from_open
        - target_close_loc_in_range
    """
    if price_df is None or len(price_df) == 0:
        raise ValueError("price_df is empty")

    _validate_ohlc_columns(price_df)

    df = price_df.copy()
    out = pd.DataFrame(index=df.index)

    next_open = df["Open"].shift(-1)
    next_high = df["High"].shift(-1)
    next_low = df["Low"].shift(-1)
    next_close = df["Close"].shift(-1)
    prev_close = df["Close"]

    out["target_gap_ret"] = next_open / prev_close - 1.0
    out["target_high_from_open"] = next_high / next_open - 1.0
    out["target_low_from_open"] = 1.0 - (next_low / next_open)
    out["target_close_from_open"] = next_close / next_open - 1.0
    out["target_range_from_open"] = (next_high - next_low) / next_open

    intraday_range = next_high - next_low
    close_loc = (next_close - next_low) / intraday_range
    close_loc = close_loc.where(intraday_range > 0)
    out["target_close_loc_in_range"] = close_loc

    if dropna:
        out = out.dropna()

    return out


def save_ohlc_component_targets(
    price_df: pd.DataFrame,
    output_path: Path,
    dropna: bool = False,
) -> pd.DataFrame:
    """
    Build and save OHLC component targets to parquet.
    """
    targets = build_ohlc_component_targets(price_df=price_df, dropna=dropna)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(output_path)
    return targets


def load_ohlc_component_targets(path: Path) -> pd.DataFrame:
    """
    Load previously saved OHLC component targets.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Targets file not found: {path}")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df
