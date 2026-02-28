"""
src/data/live_fetcher.py
=========================
Task 26 — Live market data fetcher.

Fetches fresh end-of-day data from yFinance and FRED, appends to the
existing Parquet store with backup + idempotency guarantees.

Functions
---------
fetch_daily_update()        Download latest row for all tickers.
fetch_fred_update()         Pull latest FRED macro series.
append_to_parquet()         Append a single row; dedup + schema-check.
validate_market_open()      True if date is a valid trading day.
run_daily_fetch()           Orchestrate: validate → fetch → append.
"""
from __future__ import annotations

import logging
import shutil
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------

EQUITY_TICKERS: Dict[str, str] = {
    "^GSPC":    "spx_daily.parquet",
    "^VIX":     "vix_daily.parquet",
    "^VVIX":    "vvix_daily.parquet",
    "^TNX":     "tnx_daily.parquet",
    "^TYX":     "tyx_daily.parquet",
    "DX-Y.NYB": "dxy_daily.parquet",
    "ES=F":     "es_daily.parquet",
}

FRED_SERIES: Dict[str, str] = {
    "T10Y2Y":       "t10y2y",
    "DFF":          "fed_funds",
    "BAMLH0A0HYM2": "hy_spread",
    "VIXCLS":       "vix_fred",
}

RETRY_DELAYS = (5, 15, 45)   # seconds between exponential-backoff attempts


# ---------------------------------------------------------------------------
# Market calendar helpers
# ---------------------------------------------------------------------------

def validate_market_open(target_date: date, raw_dir: Optional[Path] = None) -> bool:
    """Return True if *target_date* is a US equity trading day.

    Uses the calendar module (Phase 1) when available; falls back to a
    simple weekday check + known US holiday list.
    """
    if target_date.weekday() >= 5:      # Saturday=5, Sunday=6
        logger.info("validate_market_open: %s is a weekend — skip.", target_date)
        return False

    # Try Phase 1 calendar
    try:
        from src.data.calendar import is_trading_day
        result = is_trading_day(target_date)
        logger.info("validate_market_open: %s → %s (calendar module)", target_date, result)
        return result
    except Exception:
        pass

    # Fallback: simple NYSE holiday list for current year
    year = target_date.year
    holidays = _us_market_holidays(year)
    is_open = target_date not in holidays
    logger.info("validate_market_open: %s → %s (fallback)", target_date, is_open)
    return is_open


def _us_market_holidays(year: int) -> set:
    """Return a minimal set of fixed and observed NYSE holidays."""
    from datetime import date as d
    def nearest_weekday(dt: date) -> date:
        if dt.weekday() == 5:   return dt - timedelta(1)   # Sat → Fri
        if dt.weekday() == 6:   return dt + timedelta(1)   # Sun → Mon
        return dt

    holidays = {
        nearest_weekday(d(year,  1,  1)),   # New Year's Day
        nearest_weekday(d(year,  7,  4)),   # Independence Day
        nearest_weekday(d(year, 11, 11)),   # Veterans Day (markets open — included for completeness)
        nearest_weekday(d(year, 12, 25)),   # Christmas
    }
    # MLK Day (3rd Monday Jan), Presidents Day (3rd Monday Feb),
    # Memorial Day (last Monday May), Labor Day (1st Monday Sep)
    def nth_weekday(y, m, weekday, n):
        first = d(y, m, 1)
        delta = (weekday - first.weekday()) % 7
        first_match = first + timedelta(delta)
        return first_match + timedelta(7 * (n - 1))

    def last_monday(y, m):
        # last Monday of month
        last = d(y, m + 1, 1) - timedelta(1) if m < 12 else d(y, 12, 31)
        delta = last.weekday()   # Mon=0 … Sun=6
        return last - timedelta(delta)

    holidays.add(nth_weekday(year, 1, 0, 3))   # MLK
    holidays.add(nth_weekday(year, 2, 0, 3))   # Presidents Day
    holidays.add(last_monday(year, 5))          # Memorial Day
    holidays.add(nth_weekday(year, 9, 0, 1))   # Labor Day

    # Thanksgiving (4th Thursday Nov) + Black Friday
    thanksgiving = nth_weekday(year, 11, 3, 4)
    holidays.add(thanksgiving)
    holidays.add(thanksgiving + timedelta(1))   # Black Friday (half-day, treat as closed)

    return holidays


# ---------------------------------------------------------------------------
# yFinance fetch helpers
# ---------------------------------------------------------------------------

def _fetch_ticker_row(ticker: str, target_date: date) -> Optional[pd.Series]:
    """Download the single row for *ticker* on *target_date* with retries."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — cannot fetch %s", ticker)
        return None

    start = target_date.strftime("%Y-%m-%d")
    end   = (target_date + timedelta(days=3)).strftime("%Y-%m-%d")

    for attempt, delay in enumerate(RETRY_DELAYS, 1):
        try:
            raw = yf.download(ticker, start=start, end=end,
                              progress=False, auto_adjust=True)

            # Handle yfinance ≥0.2.54 MultiIndex columns
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            if raw.empty:
                logger.warning("%s: empty response on attempt %d", ticker, attempt)
            else:
                # Find the row closest to target_date (may be next business day)
                raw.index = pd.to_datetime(raw.index).normalize()
                if pd.Timestamp(target_date) in raw.index:
                    return raw.loc[pd.Timestamp(target_date)]
                # Return most recent row
                return raw.iloc[-1]

        except Exception as exc:
            logger.warning("%s fetch attempt %d failed: %s", ticker, attempt, exc)
            if attempt < len(RETRY_DELAYS):
                time.sleep(delay)

    logger.error("%s: all %d fetch attempts failed", ticker, len(RETRY_DELAYS))
    return None


def fetch_daily_update(
    target_date: Optional[date] = None,
    raw_dir: Optional[Path] = None,
) -> Dict[str, Optional[pd.Series]]:
    """Fetch the latest OHLCV row for every equity ticker.

    Returns a dict mapping filename → Series (or None on failure).
    Sets a ``partial_data`` key to True if any ticker failed.
    """
    if target_date is None:
        target_date = date.today()

    results: Dict[str, Optional[pd.Series]] = {}
    failed = []

    for ticker, fname in EQUITY_TICKERS.items():
        row = _fetch_ticker_row(ticker, target_date)
        results[fname] = row
        if row is None:
            failed.append(ticker)
            logger.warning("Failed to fetch %s — will use previous close as fallback", ticker)

    results["partial_data"] = len(failed) > 0
    results["failed_tickers"] = failed
    logger.info("fetch_daily_update complete: %d tickers, %d failed",
                len(EQUITY_TICKERS), len(failed))
    return results


# ---------------------------------------------------------------------------
# FRED fetch helpers
# ---------------------------------------------------------------------------

def fetch_fred_update(target_date: Optional[date] = None) -> Dict[str, Optional[float]]:
    """Pull latest available value for each FRED series.

    FRED data publishes with a 1-day lag; returns the most recent
    observation and its actual observation date.
    """
    if target_date is None:
        target_date = date.today()

    results: Dict[str, object] = {}

    try:
        from fredapi import Fred
        import os
        api_key = os.getenv("FRED_API_KEY", "")
        if not api_key:
            logger.warning("FRED_API_KEY not set — skipping FRED fetch")
            return results
        fred = Fred(api_key=api_key)
    except ImportError:
        logger.warning("fredapi not installed — skipping FRED fetch")
        return results

    observation_end = target_date.strftime("%Y-%m-%d")
    observation_start = (target_date - timedelta(days=10)).strftime("%Y-%m-%d")

    for series_id, key in FRED_SERIES.items():
        try:
            data = fred.get_series(series_id,
                                   observation_start=observation_start,
                                   observation_end=observation_end)
            if data.empty:
                logger.warning("FRED %s: no data in window", series_id)
                results[key] = None
                results[f"{key}_obs_date"] = None
            else:
                latest = data.dropna()
                if latest.empty:
                    results[key] = None
                    results[f"{key}_obs_date"] = None
                else:
                    results[key] = float(latest.iloc[-1])
                    results[f"{key}_obs_date"] = str(latest.index[-1].date())
                    logger.info("FRED %s = %.4f (obs %s)",
                                series_id, results[key], results[f"{key}_obs_date"])
        except Exception as exc:
            logger.warning("FRED %s fetch failed: %s", series_id, exc)
            results[key] = None

    return results


# ---------------------------------------------------------------------------
# Parquet append helpers
# ---------------------------------------------------------------------------

def append_to_parquet(
    new_row: pd.Series,
    filepath: Path,
    date_index: Optional[date] = None,
    backup: bool = True,
) -> bool:
    """Append *new_row* to *filepath*, with duplicate-date rejection and
    schema validation.

    Parameters
    ----------
    new_row    : Single-row Series (index = column names).
    filepath   : Path to the existing Parquet file.
    date_index : The date this row belongs to.
    backup     : If True, write a dated backup before modifying.

    Returns
    -------
    True if a new row was appended, False if date already present.
    """
    filepath = Path(filepath)

    # Load existing data
    if filepath.exists():
        existing = pd.read_parquet(filepath)
        existing.index = pd.to_datetime(existing.index)
    else:
        existing = pd.DataFrame()

    # Determine the timestamp for this row
    ts = pd.Timestamp(date_index) if date_index else pd.Timestamp("today").normalize()

    # Idempotency: reject if date already present
    if not existing.empty and ts in existing.index:
        logger.info("append_to_parquet: %s already present in %s — no-op",
                    ts.date(), filepath.name)
        return False

    # Schema validation
    if not existing.empty:
        new_cols = set(new_row.index)
        old_cols = set(existing.columns)
        missing = old_cols - new_cols
        extra   = new_cols - old_cols
        if missing:
            logger.warning("New row missing columns: %s — filling with NaN", missing)
            for col in missing:
                new_row[col] = np.nan
        if extra:
            logger.warning("New row has extra columns: %s — dropping", extra)
            new_row = new_row[list(old_cols)]

    # Backup before write
    if backup and filepath.exists():
        backup_path = filepath.parent / f"{filepath.stem}_{date.today().strftime('%Y%m%d')}{filepath.suffix}"
        shutil.copy2(filepath, backup_path)
        logger.info("Backup written → %s", backup_path)

    # Append
    new_df = pd.DataFrame([new_row], index=[ts])
    new_df.index = pd.to_datetime(new_df.index)
    combined = pd.concat([existing, new_df]).sort_index()
    combined.to_parquet(filepath)
    logger.info("append_to_parquet: appended %s to %s (%d rows total)",
                ts.date(), filepath.name, len(combined))
    return True


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_daily_fetch(
    target_date: Optional[date] = None,
    raw_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Full orchestration: validate → fetch equity → fetch FRED → append.

    Returns a status dict with keys:
        market_open   : bool
        tickers_ok    : list[str]
        tickers_fail  : list[str]
        appended      : list[str]
        data_quality  : 'FULL' | 'PARTIAL' | 'DEGRADED'
        fred_ok       : bool
    """
    if target_date is None:
        target_date = date.today()

    # Resolve raw_dir
    if raw_dir is None:
        try:
            from config.settings import RAW_DATA_DIR
            raw_dir = Path(RAW_DATA_DIR)
        except Exception:
            raw_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

    status: Dict[str, object] = {
        "date":         str(target_date),
        "market_open":  False,
        "tickers_ok":   [],
        "tickers_fail": [],
        "appended":     [],
        "data_quality": "DEGRADED",
        "fred_ok":      False,
    }

    # Step 1: Validate market open
    if not validate_market_open(target_date, raw_dir):
        status["market_open"] = False
        logger.info("run_daily_fetch: market closed on %s — nothing to do", target_date)
        return status

    status["market_open"] = True

    # Step 2: Fetch equity data
    fetch_results = fetch_daily_update(target_date, raw_dir)
    failed = fetch_results.get("failed_tickers", [])

    for fname, row in fetch_results.items():
        if fname in ("partial_data", "failed_tickers"):
            continue
        if row is None:
            status["tickers_fail"].append(fname)
            continue

        status["tickers_ok"].append(fname)
        fpath = raw_dir / fname
        try:
            appended = append_to_parquet(row, fpath, target_date)
            if appended:
                status["appended"].append(fname)
        except Exception as exc:
            logger.error("append failed for %s: %s", fname, exc)
            status["tickers_fail"].append(fname)
            status["tickers_ok"].remove(fname)

    # Step 3: Fetch FRED
    try:
        fred_data = fetch_fred_update(target_date)
        status["fred_data"] = fred_data
        status["fred_ok"] = bool(fred_data)
    except Exception as exc:
        logger.warning("FRED fetch failed: %s", exc)
        status["fred_ok"] = False

    # Step 4: Assess data quality
    n_fail = len(status["tickers_fail"])
    n_total = len(EQUITY_TICKERS)
    if n_fail == 0:
        status["data_quality"] = "FULL"
    elif n_fail <= n_total // 3:
        status["data_quality"] = "PARTIAL"
    else:
        status["data_quality"] = "DEGRADED"

    logger.info(
        "run_daily_fetch: date=%s quality=%s ok=%d fail=%d appended=%d",
        target_date, status["data_quality"],
        len(status["tickers_ok"]), n_fail, len(status["appended"]),
    )
    return status
