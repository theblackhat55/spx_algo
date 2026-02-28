"""
src/data/fetcher.py
===================
Task 1 — Download all raw market data and store as Parquet files.

CONSTRAINT: This module only writes to data/raw/.  It never modifies
existing files — it overwrites them on a fresh download.  All
downstream code reads from those Parquet files.

Functions
---------
fetch_yahoo_data          — Download a single Yahoo Finance ticker
fetch_all_yahoo_data      — Iterate the YAHOO_TICKERS dict
fetch_fred_data           — Download all FRED series
fetch_all_data            — Master orchestrator
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# ── project imports ───────────────────────────────────────────────────────────
from config.settings import (
    DATA_START_DATE,
    YAHOO_TICKERS,
    FRED_SERIES,
    RAW_DATA_DIR,
    MACRO_FILE,
    FRED_API_KEY,
)

logger = logging.getLogger(__name__)

# ── per-ticker filename map (human name → path) ───────────────────────────────
_TICKER_FILE_MAP: dict[str, Path] = {
    name: RAW_DATA_DIR / f"{name}_daily.parquet"
    for name in YAHOO_TICKERS
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yahoo_data(
    ticker: str,
    start_date: str,
    save_path: Path,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> Optional[pd.DataFrame]:
    """Download daily OHLCV data for *ticker* from Yahoo Finance.

    Parameters
    ----------
    ticker:
        Yahoo Finance symbol (e.g. "^GSPC").
    start_date:
        ISO-8601 date string, e.g. "2000-01-01".
    save_path:
        Destination Parquet file path.
    max_retries:
        Maximum number of attempts before giving up.
    base_delay:
        Starting back-off delay in seconds (doubles each retry).

    Returns
    -------
    DataFrame with DatetimeIndex and OHLCV columns, or ``None`` on failure.

    Notes
    -----
    yfinance has been experiencing intermittent ``YFRateLimitError``
    since mid-2025.  This function handles that exception with
    exponential back-off and returns ``None`` (rather than raising) if
    all retries are exhausted.
    """
    delay = base_delay
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Fetching %s (attempt %d/%d) ...", ticker, attempt, max_retries)
            raw = yf.download(
                ticker,
                start=start_date,
                progress=False,
                auto_adjust=True,
                multi_level_index=False,
            )

            if raw is None or raw.empty:
                raise ValueError(f"yfinance returned empty DataFrame for {ticker}")

            # Normalise column names — yfinance sometimes uses Title case
            raw.columns = [c.capitalize() for c in raw.columns]

            # Keep only OHLCV and ensure DatetimeIndex
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
            df = raw[keep].copy()
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"

            # Drop rows where all prices are NaN (can happen for futures roll dates)
            df = df.dropna(subset=["Open", "High", "Low", "Close"])

            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(save_path)
            logger.info(
                "  ✓ %s saved → %s  [%d rows, %s → %s]",
                ticker, save_path.name, len(df),
                df.index[0].date(), df.index[-1].date(),
            )
            return df

        except Exception as exc:
            # Catch rate-limit and other transient errors
            exc_name = type(exc).__name__
            if "RateLimit" in exc_name or "TooManyRequests" in exc_name:
                logger.warning(
                    "  ⚠ Rate limit hit for %s (attempt %d).  "
                    "Waiting %.0fs …", ticker, attempt, delay
                )
            else:
                logger.warning(
                    "  ⚠ Error fetching %s (attempt %d): %s – %s",
                    ticker, attempt, exc_name, exc,
                )
            last_error = exc

            if attempt < max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 60)   # cap at 60 s

    logger.error(
        "  ✗ Failed to fetch %s after %d attempts. Last error: %s",
        ticker, max_retries, last_error,
    )
    return None


def fetch_all_yahoo_data(
    tickers_dict: dict[str, str] | None = None,
    start_date: str = DATA_START_DATE,
    raw_dir: Path = RAW_DATA_DIR,
    pause_between: float = 1.0,
) -> dict[str, Optional[pd.DataFrame]]:
    """Download every ticker in *tickers_dict* and save as Parquet files.

    Parameters
    ----------
    tickers_dict:
        Mapping of ``{human_name: yf_symbol}``.  Defaults to
        ``YAHOO_TICKERS`` from settings.
    start_date:
        Historical start date for all downloads.
    raw_dir:
        Directory in which Parquet files are saved.
    pause_between:
        Sleep time in seconds between consecutive downloads.

    Returns
    -------
    Dict mapping human-readable name to the downloaded DataFrame (or
    ``None`` on failure).
    """
    if tickers_dict is None:
        tickers_dict = YAHOO_TICKERS

    results: dict[str, Optional[pd.DataFrame]] = {}

    for name, symbol in tickers_dict.items():
        save_path = raw_dir / f"{name}_daily.parquet"
        df = fetch_yahoo_data(symbol, start_date, save_path)
        results[name] = df
        if df is not None:
            time.sleep(pause_between)

    return results


def fetch_fred_data(
    series_dict: dict[str, str] | None = None,
    start_date: str = DATA_START_DATE,
    save_path: Path = MACRO_FILE,
    api_key: str | None = None,
) -> Optional[pd.DataFrame]:
    """Download FRED macroeconomic series and store as a single Parquet.

    Parameters
    ----------
    series_dict:
        Mapping of ``{FRED_series_id: human_name}``.  Defaults to
        ``FRED_SERIES`` from settings.
    start_date:
        Historical start date.
    save_path:
        Destination Parquet file.
    api_key:
        FRED API key.  Falls back to ``FRED_API_KEY`` from settings.

    Returns
    -------
    DataFrame with DatetimeIndex and one column per FRED series, or
    ``None`` if the fredapi package is unavailable or the key is missing.
    """
    try:
        from fredapi import Fred  # soft dependency check
    except ImportError:
        logger.error("fredapi is not installed.  Run: pip install fredapi")
        return None

    key = api_key or FRED_API_KEY
    if not key:
        logger.error(
            "FRED_API_KEY is not set.  Add it to your .env file.  "
            "Skipping FRED download."
        )
        return None

    if series_dict is None:
        series_dict = FRED_SERIES

    fred = Fred(api_key=key)
    frames: list[pd.Series] = []

    for series_id, col_name in series_dict.items():
        try:
            logger.info("Fetching FRED series %s (%s) …", series_id, col_name)
            s = fred.get_series(series_id, observation_start=start_date)
            s.name = col_name
            frames.append(s)
            logger.info("  ✓ %s — %d observations", col_name, len(s))
        except Exception as exc:
            logger.warning("  ⚠ Failed to fetch %s: %s", series_id, exc)

    if not frames:
        logger.error("No FRED series were downloaded successfully.")
        return None

    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path)
    logger.info(
        "FRED data saved → %s  [%d rows, %d series, %s → %s]",
        save_path.name, len(df), len(frames),
        df.index[0].date(), df.index[-1].date(),
    )
    return df


def fetch_all_data(
    start_date: str = DATA_START_DATE,
    raw_dir: Path = RAW_DATA_DIR,
) -> dict[str, Optional[pd.DataFrame]]:
    """Master function: download all Yahoo Finance and FRED data.

    Orchestrates the full data acquisition pipeline, prints a summary
    of what was downloaded, and returns a consolidated dict.

    Parameters
    ----------
    start_date:
        Historical start date (ISO-8601).
    raw_dir:
        Target directory for all raw Parquet files.

    Returns
    -------
    Dict containing:
      ``"yahoo"``  → ``{name: DataFrame | None}``
      ``"fred"``   → ``DataFrame | None``
    """
    logger.info("=" * 60)
    logger.info("SPX Algo — Data Fetcher")
    logger.info("Start date: %s", start_date)
    logger.info("=" * 60)

    yahoo_results = fetch_all_yahoo_data(
        start_date=start_date, raw_dir=raw_dir
    )

    fred_result = fetch_fred_data(
        start_date=start_date, save_path=raw_dir / "macro_fred.parquet"
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    successes = 0
    failures = 0

    for name, df in yahoo_results.items():
        if df is not None:
            print(
                f"  ✓ {name:10s}  {len(df):6,d} rows  "
                f"{df.index[0].date()} → {df.index[-1].date()}"
            )
            successes += 1
        else:
            print(f"  ✗ {name:10s}  FAILED")
            failures += 1

    if fred_result is not None:
        print(
            f"  ✓ {'FRED':10s}  {len(fred_result):6,d} rows  "
            f"{fred_result.index[0].date()} → {fred_result.index[-1].date()}  "
            f"({fred_result.shape[1]} series)"
        )
        successes += 1
    else:
        print("  ✗ FRED       SKIPPED / FAILED")

    print("-" * 60)
    print(f"  {successes} succeeded  |  {failures} failed")
    print("=" * 60 + "\n")

    return {"yahoo": yahoo_results, "fred": fred_result}


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    fetch_all_data()
