"""
src/features/events.py
======================

Lightweight event-feature builder for SPX next-day forecasting.

This module adds explicit event flags that are useful for daily OHLC/range
forecasting and intentionally stays simple:

- CPI day
- FOMC day
- NFP day
- OPEX day
- month-end
- quarter-end
- holiday-adjacent session

The user-maintained CSV can be sparse. Missing dates default to 0.
Derived flags (month-end / quarter-end / holiday-adjacent) are computed from
the trading-date index directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_EVENT_COLUMNS = [
    "is_cpi_day",
    "is_nfp_day",
    "is_opex_day",
    "is_holiday_adjacent",
]


def load_event_calendar(csv_path: Path) -> pd.DataFrame:
    """
    Load event calendar CSV.

    Expected columns:
    - date
    - optional event flag columns from DEFAULT_EVENT_COLUMNS

    Missing flag columns are created and filled with 0.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        df = pd.DataFrame(columns=["date", *DEFAULT_EVENT_COLUMNS])
        return df

    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError(f"Event calendar missing required 'date' column: {csv_path}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    for col in DEFAULT_EVENT_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    for col in DEFAULT_EVENT_COLUMNS:
        df[col] = df[col].fillna(0).astype(int)

    return df[DEFAULT_EVENT_COLUMNS]


def compute_event_features(
    trading_index: pd.DatetimeIndex,
    event_calendar_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute event features aligned to the provided trading index.

    Parameters
    ----------
    trading_index
        Trading-day DatetimeIndex.
    event_calendar_path
        Path to user-maintained event calendar CSV.

    Returns
    -------
    pd.DataFrame
        Event feature dataframe indexed by trading date.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(trading_index)).sort_values().unique()
    out = pd.DataFrame(index=idx)

    for col in DEFAULT_EVENT_COLUMNS:
        out[col] = 0

    if event_calendar_path is not None:
        event_df = load_event_calendar(Path(event_calendar_path))
        if not event_df.empty:
            common = out.index.intersection(event_df.index)
            if len(common) > 0:
                for col in DEFAULT_EVENT_COLUMNS:
                    out.loc[common, col] = event_df.loc[common, col].astype(int)

    # Derived deterministic flags from trading calendar itself
    idx_series = pd.Series(idx, index=idx)

    next_trade = idx_series.shift(-1)
    prev_trade = idx_series.shift(1)

    # holiday-adjacent: large gap between trading sessions (typically >= 3 calendar days)
    next_gap = (next_trade - idx_series).dt.days
    prev_gap = (idx_series - prev_trade).dt.days

    holiday_adjacent = (
        next_gap.fillna(1).ge(3) |
        prev_gap.fillna(1).ge(3)
    )
    out["is_holiday_adjacent"] = holiday_adjacent.astype(int)

    return out[DEFAULT_EVENT_COLUMNS]


if __name__ == "__main__":
    sample_idx = pd.date_range("2026-01-01", periods=10, freq="B")
    df = compute_event_features(sample_idx)
    print(df)
