"""
src/data/calendar.py
====================
Task 3 — Build the event calendar (FOMC dates, OpEx Fridays, holidays).

The calendar is a pandas DataFrame indexed by trading date with boolean
columns for every market-moving scheduled event.  It is saved to
data/raw/calendar_events.csv and is consumed by Task 7
(src/features/calendar_features.py).

Functions
---------
get_fomc_dates          — Return a Series of historical FOMC announcement dates
get_opex_fridays        — Return a Series of monthly options expiration Fridays
build_trading_calendar  — Construct the master event calendar DataFrame
load_calendar           — Load from CSV (or rebuild if missing)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    DATA_START_DATE,
    CALENDAR_FILE,
    SPX_FILE,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FOMC dates
# ─────────────────────────────────────────────────────────────────────────────

# Hard-coded historical FOMC announcement dates (second day of 2-day meetings)
# through early 2026.  The Fed publishes the schedule for the next calendar
# year, so this list can be updated annually.
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
_FOMC_DATES_RAW: list[str] = [
    # 2000
    "2000-02-02", "2000-03-21", "2000-05-16", "2000-06-28",
    "2000-08-22", "2000-10-03", "2000-11-15", "2000-12-19",
    # 2001
    "2001-01-31", "2001-03-20", "2001-05-15", "2001-06-27",
    "2001-08-21", "2001-10-02", "2001-11-06", "2001-12-11",
    # 2002
    "2002-01-30", "2002-03-19", "2002-05-07", "2002-06-26",
    "2002-08-13", "2002-09-24", "2002-11-06", "2002-12-10",
    # 2003
    "2003-01-29", "2003-03-18", "2003-05-06", "2003-06-25",
    "2003-08-12", "2003-09-16", "2003-10-28", "2003-12-09",
    # 2004
    "2004-01-28", "2004-03-16", "2004-05-04", "2004-06-30",
    "2004-08-10", "2004-09-21", "2004-11-10", "2004-12-14",
    # 2005
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30",
    "2005-08-09", "2005-09-20", "2005-11-01", "2005-12-13",
    # 2006
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29",
    "2006-08-08", "2006-09-20", "2006-10-25", "2006-12-12",
    # 2007
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28",
    "2007-08-07", "2007-09-18", "2007-10-31", "2007-12-11",
    # 2008
    "2008-01-30", "2008-03-18", "2008-04-30", "2008-06-25",
    "2008-08-05", "2008-09-16", "2008-10-29", "2008-12-16",
    # 2009
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24",
    "2009-08-12", "2009-09-23", "2009-11-04", "2009-12-16",
    # 2010
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23",
    "2010-08-10", "2010-09-21", "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22",
    "2011-08-09", "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20",
    "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19",
    "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18",
    "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05",
    "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (scheduled)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]


def get_fomc_dates(
    start: str = DATA_START_DATE,
    end: str | None = None,
) -> pd.DatetimeIndex:
    """Return a DatetimeIndex of historical FOMC announcement dates.

    Parameters
    ----------
    start:
        Earliest date to include.
    end:
        Latest date to include (defaults to today).

    Returns
    -------
    Sorted DatetimeIndex of FOMC announcement dates.
    """
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today()
    dates = pd.DatetimeIndex(pd.to_datetime(_FOMC_DATES_RAW))
    mask = (dates >= pd.Timestamp(start)) & (dates <= end_ts)
    return dates[mask].sort_values()


# ─────────────────────────────────────────────────────────────────────────────
# Options expiration dates
# ─────────────────────────────────────────────────────────────────────────────

def _third_friday(year: int, month: int) -> pd.Timestamp:
    """Return the third Friday of (year, month) — monthly OpEx date."""
    # First day of the month
    first = pd.Timestamp(year=year, month=month, day=1)
    # Day-of-week offset to the first Friday (Friday = 4 in pandas/Python)
    dow = first.dayofweek   # 0=Mon … 6=Sun
    days_to_friday = (4 - dow) % 7
    first_friday = first + pd.Timedelta(days=days_to_friday)
    return first_friday + pd.Timedelta(weeks=2)


def get_opex_fridays(
    start: str = DATA_START_DATE,
    end: str | None = None,
) -> pd.DatetimeIndex:
    """Return a DatetimeIndex of monthly options expiration Fridays.

    OpEx is the third Friday of each month.  If that Friday is a
    US market holiday, OpEx moves to the preceding Thursday (handled
    conservatively here — the calendar_features module can refine this
    with actual holiday data).

    Parameters
    ----------
    start, end:
        Date range boundaries.

    Returns
    -------
    Sorted DatetimeIndex of OpEx Fridays.
    """
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today() + pd.DateOffset(months=3)
    start_ts = pd.Timestamp(start)

    dates = []
    current = pd.Timestamp(year=start_ts.year, month=start_ts.month, day=1)

    while current <= end_ts:
        opex = _third_friday(current.year, current.month)
        dates.append(opex)
        # Move to next month
        if current.month == 12:
            current = pd.Timestamp(year=current.year + 1, month=1, day=1)
        else:
            current = pd.Timestamp(year=current.year, month=current.month + 1, day=1)

    idx = pd.DatetimeIndex(dates)
    return idx[(idx >= start_ts) & (idx <= end_ts)].sort_values()


def _is_quarterly_opex(dt: pd.Timestamp) -> bool:
    """Return True if *dt* is a quarterly triple-witching expiration."""
    return dt.month in {3, 6, 9, 12}


# ─────────────────────────────────────────────────────────────────────────────
# Master calendar builder
# ─────────────────────────────────────────────────────────────────────────────

def build_trading_calendar(
    trading_dates: pd.DatetimeIndex | None = None,
    spx_file: Path = SPX_FILE,
    save_path: Path = CALENDAR_FILE,
) -> pd.DataFrame:
    """Build the master event calendar DataFrame.

    The calendar is aligned to the actual SPX trading day index
    (loaded from *spx_file*).  If the SPX file doesn't exist yet,
    a US business-day calendar is used as a fallback.

    Columns produced
    ----------------
    is_fomc_day, is_day_before_fomc, is_day_after_fomc,
    is_monthly_opex, is_quarterly_opex, is_opex_week,
    days_to_next_fomc, days_to_next_opex,
    is_month_end, is_quarter_end, is_first_trading_day_of_month,
    trading_days_remaining_month

    Parameters
    ----------
    trading_dates:
        Explicit DatetimeIndex of trading days.  If None, derived from
        the SPX Parquet file.
    spx_file:
        Path to spx_daily.parquet (used to determine actual trading days).
    save_path:
        Where to write the CSV.

    Returns
    -------
    DataFrame indexed by trading date with the event columns.
    """
    # ── Get actual trading day calendar ───────────────────────────────────────
    if trading_dates is None:
        if spx_file.exists():
            spx_df = pd.read_parquet(spx_file)
            trading_dates = spx_df.index
            logger.info("Using %d trading days from SPX file.", len(trading_dates))
        else:
            logger.warning(
                "SPX file not found; falling back to US business day calendar."
            )
            trading_dates = pd.bdate_range(
                start=DATA_START_DATE,
                end=pd.Timestamp.today() + pd.DateOffset(months=6),
            )

    trading_dates = pd.DatetimeIndex(trading_dates).sort_values()

    # ── Base DataFrame ────────────────────────────────────────────────────────
    cal = pd.DataFrame(index=trading_dates)
    cal.index.name = "Date"

    # ── FOMC columns ──────────────────────────────────────────────────────────
    fomc_dates = set(get_fomc_dates(end=str(trading_dates[-1].date())))
    cal["is_fomc_day"] = cal.index.isin(fomc_dates).astype(int)

    # Days before / after FOMC (shift the boolean mask)
    fomc_series = cal["is_fomc_day"].astype(bool)
    cal["is_day_before_fomc"] = fomc_series.shift(-1, fill_value=False).astype(int)
    cal["is_day_after_fomc"]  = fomc_series.shift(1,  fill_value=False).astype(int)

    # Trading days to next FOMC for each date
    fomc_idx = sorted(fomc_dates)
    def _days_to_next_fomc(dt: pd.Timestamp) -> int:
        future_fomc = [f for f in fomc_idx if f > dt]
        if not future_fomc:
            return 50   # placeholder for beyond-known-schedule dates
        next_f = future_fomc[0]
        # Count trading days between dt and next_fomc
        between = trading_dates[(trading_dates > dt) & (trading_dates <= next_f)]
        return len(between)

    cal["days_to_next_fomc"] = [
        _days_to_next_fomc(d) for d in cal.index
    ]

    # ── OpEx columns ──────────────────────────────────────────────────────────
    opex_dates = set(get_opex_fridays(end=str(trading_dates[-1].date())))
    cal["is_monthly_opex"]   = cal.index.isin(opex_dates).astype(int)
    cal["is_quarterly_opex"] = (
        cal.index.isin(opex_dates) & cal.index.map(_is_quarterly_opex)
    ).astype(int)

    # OpEx week = any day in the same ISO week as an OpEx Friday
    opex_year_weeks = {
        (d.isocalendar()[0], d.isocalendar()[1]) for d in opex_dates
    }
    cal["is_opex_week"] = cal.index.map(
        lambda d: (d.isocalendar()[0], d.isocalendar()[1]) in opex_year_weeks
    ).astype(int)

    # Trading days to next OpEx
    opex_list = sorted(opex_dates)
    def _days_to_next_opex(dt: pd.Timestamp) -> int:
        future_opex = [o for o in opex_list if o > dt]
        if not future_opex:
            return 25
        next_o = future_opex[0]
        between = trading_dates[(trading_dates > dt) & (trading_dates <= next_o)]
        return len(between)

    cal["days_to_next_opex"] = [
        _days_to_next_opex(d) for d in cal.index
    ]

    # ── Month / quarter structure ─────────────────────────────────────────────
    # Last trading day of each calendar month
    month_ends = (
        cal.groupby([cal.index.year, cal.index.month])
        .apply(lambda g: g.index[-1])
        .values
    )
    cal["is_month_end"]    = cal.index.isin(month_ends).astype(int)

    # Last trading day of each calendar quarter
    quarter_ends = (
        cal.groupby([cal.index.year, cal.index.quarter])
        .apply(lambda g: g.index[-1])
        .values
    )
    cal["is_quarter_end"]  = cal.index.isin(quarter_ends).astype(int)

    # First trading day of each calendar month
    month_starts = (
        cal.groupby([cal.index.year, cal.index.month])
        .apply(lambda g: g.index[0])
        .values
    )
    cal["is_first_trading_day_of_month"] = cal.index.isin(month_starts).astype(int)

    # Trading days remaining in the month (inclusive of current day)
    cal["trading_days_remaining_month"] = (
        cal.groupby([cal.index.year, cal.index.month])
        .cumcount(ascending=False) + 1
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cal.to_csv(save_path)
    logger.info(
        "Calendar saved → %s  [%d rows, %d columns]",
        save_path.name, len(cal), cal.shape[1],
    )

    return cal


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_calendar(
    calendar_path: Path = CALENDAR_FILE,
    rebuild_if_missing: bool = True,
) -> pd.DataFrame:
    """Load the event calendar from CSV, rebuilding if not present.

    Parameters
    ----------
    calendar_path:
        Path to the CSV produced by :func:`build_trading_calendar`.
    rebuild_if_missing:
        If True and the file is absent, build the calendar first.

    Returns
    -------
    DataFrame with DatetimeIndex and event columns.
    """
    if not calendar_path.exists():
        if rebuild_if_missing:
            logger.info("Calendar CSV not found — building now …")
            return build_trading_calendar(save_path=calendar_path)
        else:
            raise FileNotFoundError(f"Calendar file not found: {calendar_path}")

    cal = pd.read_csv(calendar_path, index_col="Date", parse_dates=True)
    logger.info("Calendar loaded from %s  [%d rows]", calendar_path.name, len(cal))
    return cal


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    cal = build_trading_calendar()
    print(cal.head(10))
    print("\nFOMC days (first 5):")
    print(cal[cal["is_fomc_day"] == 1].head())
    print("\nQuarterly OpEx days (first 5):")
    print(cal[cal["is_quarterly_opex"] == 1].head())
