"""
src/data/validator.py
=====================
Task 2 — Comprehensive data quality validation.

Every function returns a list of error strings (empty = all good).
validate_ohlcv *raises* ValueError on critical constraint violations
(nulls, OHLCV relationship errors) so that callers fail loudly.

Functions
---------
validate_ohlcv              — OHLCV sanity checks
validate_date_alignment     — Cross-ticker date consistency
validate_no_weekend_dates   — No Saturday / Sunday rows
validate_completeness       — Trading-day count within 2% of expectation
run_all_validations         — Load all Parquets and run every check
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    RAW_DATA_DIR,
    DATA_START_DATE,
    YAHOO_TICKERS,
)

logger = logging.getLogger(__name__)

# ── approximate number of US trading days per year ───────────────────────────
_TRADING_DAYS_PER_YEAR = 252
_TOLERANCE = 0.02          # allow 2% deviation from expected count


def validate_ohlcv(
    df: pd.DataFrame,
    name: str = "DataFrame",
) -> list[str]:
    """Validate OHLCV integrity for a single instrument DataFrame.

    Checks performed
    ----------------
    1. No NaN values in OHLCV columns.
    2. High >= Low for every row.
    3. High >= max(Open, Close) for every row.
    4. Low  <= min(Open, Close) for every row.
    5. Close > 0 for every row.
    6. Volume >= 0 for every row (some instruments lack volume data).
    7. Index is a DatetimeIndex.
    8. Index is sorted ascending (monotonic).
    9. No duplicate dates.

    Parameters
    ----------
    df:
        DataFrame with columns Open, High, Low, Close (Volume optional).
    name:
        Human-readable label used in error messages.

    Returns
    -------
    List of error strings.  Empty list means all checks passed.

    Raises
    ------
    ValueError
        If OHLCV null values are found or if any High < Low violation
        exists (these are data-corruption-level errors).
    """
    errors: list[str] = []

    # ── Column existence ──────────────────────────────────────────────────────
    required_cols = ["Open", "High", "Low", "Close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        msg = f"[{name}] Missing columns: {missing}"
        errors.append(msg)
        raise ValueError(msg)   # Can't continue without these columns

    # ── 1. Null check (critical) ──────────────────────────────────────────────
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        msg = (
            f"[{name}] NULL values found in OHLCV columns: "
            + null_counts[null_counts > 0].to_dict().__repr__()
        )
        errors.append(msg)
        raise ValueError(msg)

    # ── 2. High >= Low (critical) ─────────────────────────────────────────────
    violations = (df["High"] < df["Low"]).sum()
    if violations:
        msg = f"[{name}] {violations} rows where High < Low — data corruption."
        errors.append(msg)
        raise ValueError(msg)

    # ── 3. High >= max(Open, Close) ───────────────────────────────────────────
    bad_high = (df["High"] < df[["Open", "Close"]].max(axis=1)).sum()
    if bad_high:
        errors.append(
            f"[{name}] {bad_high} rows where High < max(Open, Close)."
        )

    # ── 4. Low <= min(Open, Close) ────────────────────────────────────────────
    bad_low = (df["Low"] > df[["Open", "Close"]].min(axis=1)).sum()
    if bad_low:
        errors.append(
            f"[{name}] {bad_low} rows where Low > min(Open, Close)."
        )

    # ── 5. Close > 0 ──────────────────────────────────────────────────────────
    non_positive = (df["Close"] <= 0).sum()
    if non_positive:
        errors.append(
            f"[{name}] {non_positive} rows where Close <= 0."
        )

    # ── 6. Volume >= 0 ────────────────────────────────────────────────────────
    if "Volume" in df.columns:
        neg_vol = (df["Volume"] < 0).sum()
        if neg_vol:
            errors.append(f"[{name}] {neg_vol} rows with negative Volume.")

    # ── 7. DatetimeIndex ──────────────────────────────────────────────────────
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"[{name}] Index is not a DatetimeIndex.")

    # ── 8. Monotonic increasing ───────────────────────────────────────────────
    if not df.index.is_monotonic_increasing:
        errors.append(f"[{name}] Date index is not monotonically increasing.")

    # ── 9. No duplicate dates ─────────────────────────────────────────────────
    dups = df.index.duplicated().sum()
    if dups:
        errors.append(f"[{name}] {dups} duplicate date entries found.")

    return errors


def validate_date_alignment(
    dfs_dict: dict[str, pd.DataFrame],
) -> dict[str, list[str]]:
    """Check that all DataFrames share the same trading-day calendar.

    Parameters
    ----------
    dfs_dict:
        Mapping of ``{name: DataFrame}`` for instruments to compare.

    Returns
    -------
    Dict mapping instrument-pair names to lists of misalignment notes.
    An empty inner list means the pair is fully aligned.
    """
    report: dict[str, list[str]] = {}
    names = list(dfs_dict.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a_idx = dfs_dict[a_name].index
            b_idx = dfs_dict[b_name].index

            only_in_a = a_idx.difference(b_idx)
            only_in_b = b_idx.difference(a_idx)

            notes: list[str] = []
            # Allow up to 5 mismatched dates (holidays / data gaps)
            if len(only_in_a) > 5:
                notes.append(
                    f"  {len(only_in_a)} dates in {a_name} not in {b_name}"
                )
            if len(only_in_b) > 5:
                notes.append(
                    f"  {len(only_in_b)} dates in {b_name} not in {a_name}"
                )

            key = f"{a_name}↔{b_name}"
            report[key] = notes

    return report


def validate_no_weekend_dates(
    df: pd.DataFrame,
    name: str = "DataFrame",
) -> list[str]:
    """Verify no rows fall on a weekend (Saturday=5, Sunday=6).

    Parameters
    ----------
    df:
        DataFrame with a DatetimeIndex.
    name:
        Label used in error messages.

    Returns
    -------
    List of error strings.  Empty list means no weekend dates.
    """
    errors: list[str] = []

    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"[{name}] Index is not a DatetimeIndex — cannot check weekends.")
        return errors

    weekend_mask = df.index.dayofweek >= 5
    n_weekend = weekend_mask.sum()
    if n_weekend:
        examples = df.index[weekend_mask][:3].date.tolist()
        errors.append(
            f"[{name}] {n_weekend} weekend dates found (first 3: {examples})."
        )

    return errors


def validate_completeness(
    df: pd.DataFrame,
    expected_start: str,
    expected_end: str,
    name: str = "DataFrame",
    tolerance: float = _TOLERANCE,
) -> list[str]:
    """Check that the row count is within *tolerance* of the expected number.

    Parameters
    ----------
    df:
        DataFrame with a DatetimeIndex.
    expected_start, expected_end:
        ISO-8601 date strings bounding the expected date range.
    name:
        Label used in error messages.
    tolerance:
        Fractional tolerance, e.g. 0.02 = within 2%.

    Returns
    -------
    List of error strings.  Empty list means completeness is satisfied.
    """
    errors: list[str] = []

    start = pd.Timestamp(expected_start)
    end   = pd.Timestamp(expected_end)

    calendar_days = (end - start).days
    years_approx  = calendar_days / 365.25
    expected_rows = int(years_approx * _TRADING_DAYS_PER_YEAR)

    # Count rows within the expected range
    in_range = df.loc[start:end]
    actual_rows = len(in_range)

    if expected_rows == 0:
        return errors

    deviation = abs(actual_rows - expected_rows) / expected_rows
    if deviation > tolerance:
        errors.append(
            f"[{name}] Row count {actual_rows:,} deviates "
            f"{deviation:.1%} from expected ~{expected_rows:,} "
            f"(tolerance {tolerance:.0%})."
        )

    return errors


def run_all_validations(
    raw_dir: Path = RAW_DATA_DIR,
    expected_start: str = DATA_START_DATE,
) -> bool:
    """Load all Parquet files from *raw_dir* and run every validation.

    Parameters
    ----------
    raw_dir:
        Directory containing the downloaded Parquet files.
    expected_start:
        The earliest date we expect data to cover.

    Returns
    -------
    ``True`` if every check passes; ``False`` if any errors are found.
    """
    all_errors: list[str] = []
    dfs: dict[str, pd.DataFrame] = {}

    expected_end = pd.Timestamp.today().strftime("%Y-%m-%d")

    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)

    # ── Load and individually validate every Parquet ──────────────────────────
    parquet_files = sorted(raw_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"  ✗ No Parquet files found in {raw_dir}")
        return False

    for path in parquet_files:
        name = path.stem
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            all_errors.append(f"[{name}] Could not read Parquet: {exc}")
            print(f"  ✗ {name}: cannot read file — {exc}")
            continue

        dfs[name] = df

        # OHLCV checks (only for price files)
        ohlcv_cols = ["Open", "High", "Low", "Close"]
        if all(c in df.columns for c in ohlcv_cols):
            try:
                errs = validate_ohlcv(df, name)
                errs += validate_no_weekend_dates(df, name)
                errs += validate_completeness(
                    df, expected_start, expected_end, name
                )
            except ValueError as exc:
                errs = [str(exc)]

            if errs:
                for e in errs:
                    all_errors.append(e)
                    print(f"  ✗ {e}")
            else:
                print(
                    f"  ✓ {name:20s}  {len(df):7,d} rows  "
                    f"{df.index[0].date()} → {df.index[-1].date()}"
                )
        else:
            # Non-OHLCV file (FRED macro etc.)
            print(
                f"  ✓ {name:20s}  {len(df):7,d} rows  "
                f"{df.shape[1]} columns  (non-OHLCV)"
            )

    # ── Cross-ticker date alignment ───────────────────────────────────────────
    ohlcv_dfs = {
        k: v for k, v in dfs.items()
        if all(c in v.columns for c in ["Open", "High", "Low", "Close"])
    }
    if len(ohlcv_dfs) > 1:
        alignment_report = validate_date_alignment(ohlcv_dfs)
        for pair, notes in alignment_report.items():
            for note in notes:
                msg = f"[Alignment {pair}] {note}"
                all_errors.append(msg)
                print(f"  ⚠ {msg}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("-" * 60)
    if all_errors:
        print(f"  RESULT: FAIL — {len(all_errors)} error(s) found.")
        print("=" * 60 + "\n")
        return False
    else:
        print("  RESULT: PASS — All validations passed.")
        print("=" * 60 + "\n")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run_all_validations()
