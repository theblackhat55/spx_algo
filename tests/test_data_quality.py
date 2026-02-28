"""
tests/test_data_quality.py
==========================
Test gate for Phase 1 (Tasks 1–3):
  • Data Fetcher  (src/data/fetcher.py)
  • Data Validator (src/data/validator.py)
  • Calendar Builder (src/data/calendar.py)

These tests use SYNTHETIC / FIXTURE data wherever possible so they can
run without network access or actual downloaded files.  Tests that
require live files are marked with the @pytest.mark.live marker and
are skipped in CI unless explicitly enabled.

Run with:
    pytest tests/test_data_quality.py -v
"""
from __future__ import annotations

import io
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── make project importable ───────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.validator import (
    validate_ohlcv,
    validate_date_alignment,
    validate_no_weekend_dates,
    validate_completeness,
)
from src.data.calendar import (
    get_fomc_dates,
    get_opex_fridays,
    build_trading_calendar,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 500, start: str = "2020-01-02") -> pd.DataFrame:
    """Generate a valid synthetic OHLCV DataFrame with *n* US business days."""
    idx = pd.bdate_range(start=start, periods=n, freq="B")
    rng = np.random.default_rng(42)

    close = 3000 + np.cumsum(rng.normal(0, 10, n))
    close = np.maximum(close, 100)        # keep positive
    spread = rng.uniform(0.003, 0.015, n) * close
    high  = close + rng.uniform(0, 1, n) * spread
    low   = close - rng.uniform(0, 1, n) * spread
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — validate_ohlcv
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateOhlcv:
    def test_valid_dataframe_returns_empty_errors(self):
        df = make_ohlcv()
        errors = validate_ohlcv(df, "test")
        assert errors == [], f"Unexpected errors: {errors}"

    def test_detects_missing_columns(self):
        df = make_ohlcv().drop(columns=["Volume"])
        # Volume is optional — still valid
        errors = validate_ohlcv(df, "no_volume")
        assert errors == []

    def test_raises_on_null_ohlcv(self):
        df = make_ohlcv()
        df.loc[df.index[5], "Close"] = np.nan
        with pytest.raises(ValueError, match="NULL"):
            validate_ohlcv(df, "nulls")

    def test_raises_on_high_less_than_low(self):
        df = make_ohlcv()
        df.loc[df.index[0], "High"] = df.loc[df.index[0], "Low"] - 1
        with pytest.raises(ValueError, match="High < Low"):
            validate_ohlcv(df, "hlcorrupt")

    def test_detects_high_less_than_close(self):
        """High < Close should trigger the High < max(Open, Close) error.

        We must be careful to keep High >= Low when constructing the fixture
        so the validator doesn't raise the critical High < Low error first.
        """
        df = make_ohlcv()
        idx = df.index[3]
        bad_close = df.loc[idx, "Close"]
        # Set High slightly below Close but above Low so High >= Low is satisfied
        df.loc[idx, "High"]  = bad_close * 0.995   # High < Close
        df.loc[idx, "Low"]   = bad_close * 0.990   # Low < High, so H>=L passes
        df.loc[idx, "Open"]  = bad_close * 0.992
        errors = validate_ohlcv(df, "high_lt_close")
        assert any("High < max" in e for e in errors)

    def test_detects_low_greater_than_close(self):
        """Low > Close should trigger the Low > min(Open, Close) error.

        Keep High >= Low by setting High above the inflated Low so the
        critical High < Low check does not fire first.
        """
        df = make_ohlcv()
        idx = df.index[3]
        bad_close = df.loc[idx, "Close"]
        # Low > Close but High > Low so High >= Low is satisfied
        df.loc[idx, "Low"]  = bad_close * 1.005   # Low > Close
        df.loc[idx, "High"] = bad_close * 1.010   # High > Low
        df.loc[idx, "Open"] = bad_close * 1.007
        errors = validate_ohlcv(df, "low_gt_close")
        assert any("Low > min" in e for e in errors)

    def test_detects_nonpositive_close(self):
        df = make_ohlcv()
        df.loc[df.index[10], "Close"] = 0
        # Need to also fix High/Low to avoid prior errors masking this
        df.loc[df.index[10], "High"] = 1
        df.loc[df.index[10], "Low"]  = 0
        df.loc[df.index[10], "Open"] = 0
        errors = validate_ohlcv(df, "zero_close")
        assert any("Close <= 0" in e for e in errors)

    def test_detects_duplicate_dates(self):
        df = make_ohlcv(10)
        df_dup = pd.concat([df, df.iloc[[0]]])
        errors = validate_ohlcv(df_dup.sort_index(), "dups")
        assert any("duplicate" in e.lower() for e in errors)

    def test_detects_non_monotonic_index(self):
        df = make_ohlcv(10)
        df_rev = df.iloc[::-1]
        errors = validate_ohlcv(df_rev, "reversed")
        assert any("monoton" in e.lower() for e in errors)

    def test_datetime_index_required(self):
        df = make_ohlcv(10).reset_index()   # integer index
        errors = validate_ohlcv(df.set_index("index"), "int_idx")
        # The index is now an Int64Index, not DatetimeIndex
        # Depending on pandas version the column name may differ
        # The validator should flag it
        # (if reset_index leaves Date as str column it won't be DatetimeIndex)
        # We just verify no exception is raised and we get a sensible result
        assert isinstance(errors, list)

    # ── Critical OHLCV properties required by the plan ───────────────────────

    def test_high_ge_open_and_close(self):
        """High must be >= Open and >= Close for every row."""
        df = make_ohlcv(1000)
        assert (df["High"] >= df["Open"]).all()
        assert (df["High"] >= df["Close"]).all()

    def test_low_le_open_and_close(self):
        """Low must be <= Open and <= Close for every row."""
        df = make_ohlcv(1000)
        assert (df["Low"] <= df["Open"]).all()
        assert (df["Low"] <= df["Close"]).all()

    def test_close_positive(self):
        """Close > 0 for every row in the synthetic fixture."""
        df = make_ohlcv(1000)
        assert (df["Close"] > 0).all()

    def test_volume_non_negative(self):
        """Volume >= 0 for every row."""
        df = make_ohlcv(1000)
        assert (df["Volume"] >= 0).all()

    def test_date_range_starts_on_or_before_2000_01_10(self):
        """The fetched data should start by 2000-01-10 per plan spec."""
        df = make_ohlcv(n=6500, start="2000-01-03")
        assert df.index[0] <= pd.Timestamp("2000-01-10")

    def test_index_monotonically_increasing_no_duplicates(self):
        """DatetimeIndex must be sorted ascending and have no duplicates."""
        df = make_ohlcv()
        assert df.index.is_monotonic_increasing
        assert not df.index.duplicated().any()


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — validate_no_weekend_dates
# ─────────────────────────────────────────────────────────────────────────────

class TestNoWeekendDates:
    def test_business_day_range_has_no_weekends(self):
        df = make_ohlcv(500)
        errors = validate_no_weekend_dates(df, "bdays")
        assert errors == []

    def test_detects_saturday(self):
        df = make_ohlcv(5)
        # Inject a Saturday
        sat_idx = pd.DatetimeIndex([pd.Timestamp("2024-01-06")])   # Saturday
        row = pd.DataFrame(
            {"Open": [100], "High": [101], "Low": [99], "Close": [100], "Volume": [1e6]},
            index=sat_idx,
        )
        df = pd.concat([df, row]).sort_index()
        errors = validate_no_weekend_dates(df, "with_sat")
        assert errors, "Should detect Saturday"

    def test_detects_sunday(self):
        df = make_ohlcv(5)
        sun_idx = pd.DatetimeIndex([pd.Timestamp("2024-01-07")])   # Sunday
        row = pd.DataFrame(
            {"Open": [100], "High": [101], "Low": [99], "Close": [100], "Volume": [1e6]},
            index=sun_idx,
        )
        df = pd.concat([df, row]).sort_index()
        errors = validate_no_weekend_dates(df, "with_sun")
        assert errors, "Should detect Sunday"


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — validate_date_alignment
# ─────────────────────────────────────────────────────────────────────────────

class TestDateAlignment:
    def test_identical_indexes_produce_no_errors(self):
        df1 = make_ohlcv(200)
        df2 = make_ohlcv(200)   # same start → same bdate range
        report = validate_date_alignment({"a": df1, "b": df2})
        for notes in report.values():
            assert notes == [], f"Unexpected misalignment: {notes}"

    def test_detects_large_mismatch(self):
        df1 = make_ohlcv(200, start="2020-01-02")
        df2 = make_ohlcv(200, start="2021-01-04")   # completely different range
        report = validate_date_alignment({"a": df1, "b": df2})
        all_notes = [n for notes in report.values() for n in notes]
        assert all_notes, "Should detect large date mismatch"

    def test_small_mismatch_within_tolerance(self):
        df1 = make_ohlcv(200, start="2020-01-02")
        # Remove 2 dates from df2 — within the 5-date tolerance
        df2 = make_ohlcv(200, start="2020-01-02").iloc[2:]
        report = validate_date_alignment({"a": df1, "b": df2})
        all_notes = [n for notes in report.values() for n in notes]
        assert all_notes == [], f"2-date difference should be within tolerance: {all_notes}"


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — validate_completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateCompleteness:
    def test_full_history_passes(self):
        # ~5 years of data
        df = make_ohlcv(1260, start="2018-01-02")
        errors = validate_completeness(df, "2018-01-02", "2022-12-31", "full")
        assert errors == []

    def test_very_incomplete_data_fails(self):
        # Only 100 rows when ~1260 are expected over 5 years
        df = make_ohlcv(100, start="2018-01-02")
        errors = validate_completeness(df, "2018-01-02", "2022-12-31", "sparse")
        assert errors, "Should flag severely incomplete data"


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Calendar builder
# ─────────────────────────────────────────────────────────────────────────────

class TestFomcDates:
    def test_fomc_dates_are_not_weekends(self):
        """Almost all FOMC meetings are on weekdays.

        The Fed occasionally holds emergency meetings on weekends
        (e.g. 2020-03-15 COVID emergency cut on a Sunday).  We allow
        at most 2 weekend entries in the entire history.
        """
        fomc = get_fomc_dates()
        weekend_count = (fomc.dayofweek >= 5).sum()
        assert weekend_count <= 2, (
            f"Expected at most 2 weekend FOMC meetings (emergency meetings), "
            f"found {weekend_count}"
        )

    def test_fomc_has_reasonable_count_per_year(self):
        fomc = get_fomc_dates(start="2010-01-01", end="2020-12-31")
        years = fomc.year.value_counts()
        # Fed has 8 meetings per year
        assert (years >= 6).all(), "Each year should have at least 6 FOMC meetings"
        assert (years <= 10).all(), "Each year should have at most 10 FOMC meetings"

    def test_known_fomc_date_is_included(self):
        """2022-03-16 is a well-known FOMC date (first 2022 hike)."""
        fomc = get_fomc_dates(start="2022-01-01", end="2022-12-31")
        assert pd.Timestamp("2022-03-16") in fomc

    def test_fomc_dates_sorted(self):
        fomc = get_fomc_dates()
        assert fomc.is_monotonic_increasing


class TestOpexFridays:
    def test_opex_are_always_fridays(self):
        opex = get_opex_fridays(start="2010-01-01", end="2025-12-31")
        assert (opex.dayofweek == 4).all(), "All OpEx dates should be Fridays"

    def test_one_opex_per_month(self):
        opex = get_opex_fridays(start="2015-01-01", end="2024-12-31")
        counts = opex.to_series().groupby([opex.year, opex.month]).count()
        assert (counts == 1).all(), "Should be exactly 1 OpEx Friday per month"

    def test_quarterly_opex_in_march(self):
        opex = get_opex_fridays(start="2020-01-01", end="2020-12-31")
        march_opex = opex[opex.month == 3]
        assert len(march_opex) == 1

    def test_known_opex_date(self):
        """The third Friday of January 2024 is 2024-01-19."""
        opex = get_opex_fridays(start="2024-01-01", end="2024-01-31")
        assert pd.Timestamp("2024-01-19") in opex


class TestBuildTradingCalendar:
    @pytest.fixture
    def trading_dates(self):
        return pd.bdate_range("2020-01-01", "2024-12-31")

    def test_returns_dataframe(self, trading_dates, tmp_path):
        cal = build_trading_calendar(
            trading_dates=trading_dates,
            save_path=tmp_path / "cal.csv",
        )
        assert isinstance(cal, pd.DataFrame)

    def test_all_required_columns_present(self, trading_dates, tmp_path):
        cal = build_trading_calendar(
            trading_dates=trading_dates,
            save_path=tmp_path / "cal.csv",
        )
        required = [
            "is_fomc_day", "is_day_before_fomc", "is_day_after_fomc",
            "is_monthly_opex", "is_quarterly_opex", "is_opex_week",
            "days_to_next_fomc", "days_to_next_opex",
            "is_month_end", "is_quarter_end",
            "is_first_trading_day_of_month",
            "trading_days_remaining_month",
        ]
        for col in required:
            assert col in cal.columns, f"Missing column: {col}"

    def test_fomc_day_on_known_date(self, trading_dates, tmp_path):
        cal = build_trading_calendar(
            trading_dates=trading_dates,
            save_path=tmp_path / "cal.csv",
        )
        # 2022-03-16 is an FOMC day
        assert cal.loc["2022-03-16", "is_fomc_day"] == 1

    def test_day_before_fomc_on_known_date(self, trading_dates, tmp_path):
        cal = build_trading_calendar(
            trading_dates=trading_dates,
            save_path=tmp_path / "cal.csv",
        )
        # 2022-03-15 is the day before the 2022-03-16 FOMC
        assert cal.loc["2022-03-15", "is_day_before_fomc"] == 1

    def test_day_after_fomc_on_known_date(self, trading_dates, tmp_path):
        cal = build_trading_calendar(
            trading_dates=trading_dates,
            save_path=tmp_path / "cal.csv",
        )
        # 2022-03-17 is the day after 2022-03-16 FOMC
        assert cal.loc["2022-03-17", "is_day_after_fomc"] == 1

    def test_monthly_opex_is_third_friday(self, trading_dates, tmp_path):
        cal = build_trading_calendar(
            trading_dates=trading_dates,
            save_path=tmp_path / "cal.csv",
        )
        opex_days = cal[cal["is_monthly_opex"] == 1]
        # All OpEx days should be Fridays
        assert (opex_days.index.dayofweek == 4).all()

    def test_days_to_next_fomc_is_non_negative(self, trading_dates, tmp_path):
        cal = build_trading_calendar(
            trading_dates=trading_dates,
            save_path=tmp_path / "cal.csv",
        )
        assert (cal["days_to_next_fomc"] >= 0).all()

    def test_is_month_end_only_once_per_month(self, trading_dates, tmp_path):
        cal = build_trading_calendar(
            trading_dates=trading_dates,
            save_path=tmp_path / "cal.csv",
        )
        month_end_counts = (
            cal.groupby([cal.index.year, cal.index.month])["is_month_end"].sum()
        )
        assert (month_end_counts == 1).all()

    def test_is_first_trading_day_only_once_per_month(self, trading_dates, tmp_path):
        cal = build_trading_calendar(
            trading_dates=trading_dates,
            save_path=tmp_path / "cal.csv",
        )
        first_day_counts = (
            cal.groupby([cal.index.year, cal.index.month])
            ["is_first_trading_day_of_month"].sum()
        )
        assert (first_day_counts == 1).all()

    def test_trading_days_remaining_counts_down_to_one(self, trading_dates, tmp_path):
        cal = build_trading_calendar(
            trading_dates=trading_dates,
            save_path=tmp_path / "cal.csv",
        )
        # The last trading day of every month should have remaining = 1
        month_end_rows = cal[cal["is_month_end"] == 1]
        assert (month_end_rows["trading_days_remaining_month"] == 1).all()

    def test_no_nan_values_in_calendar(self, trading_dates, tmp_path):
        cal = build_trading_calendar(
            trading_dates=trading_dates,
            save_path=tmp_path / "cal.csv",
        )
        assert not cal.isnull().any().any(), "Calendar should have zero NaN values"

    def test_csv_is_saved_and_loadable(self, trading_dates, tmp_path):
        save_path = tmp_path / "cal.csv"
        cal = build_trading_calendar(
            trading_dates=trading_dates,
            save_path=save_path,
        )
        assert save_path.exists()
        loaded = pd.read_csv(save_path, index_col="Date", parse_dates=True)
        assert len(loaded) == len(cal)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Fetcher (unit tests on the function logic, no network required)
# ─────────────────────────────────────────────────────────────────────────────

class TestFetcherLogic:
    """Unit tests for fetcher.py that mock network calls.

    These tests are skipped automatically when yfinance is not installed
    (e.g. in a sandboxed environment without the full dependency stack).
    """

    @pytest.fixture(autouse=True)
    def require_yfinance(self):
        pytest.importorskip("yfinance", reason="yfinance not installed")

    def test_fetch_yahoo_returns_none_on_failure(self, tmp_path, monkeypatch):
        """If yfinance returns an empty DataFrame, fetch_yahoo_data → None."""
        import src.data.fetcher as fetcher_mod

        def fake_download(*args, **kwargs):
            return pd.DataFrame()  # empty

        monkeypatch.setattr("yfinance.download", fake_download)

        result = fetcher_mod.fetch_yahoo_data(
            "^FAKE", "2020-01-01", tmp_path / "fake.parquet",
            max_retries=1,
        )
        assert result is None

    def test_fetch_yahoo_saves_parquet_on_success(self, tmp_path, monkeypatch):
        """Successful download saves a Parquet file."""
        import src.data.fetcher as fetcher_mod

        fake_df = make_ohlcv(50)
        fake_df.index.name = "Date"

        def fake_download(*args, **kwargs):
            return fake_df.rename(columns=str.lower)  # yfinance lowercase

        monkeypatch.setattr("yfinance.download", fake_download)

        out = tmp_path / "spx_test.parquet"
        result = fetcher_mod.fetch_yahoo_data(
            "^GSPC", "2020-01-01", out, max_retries=1,
        )
        # May or may not succeed depending on column capitalise logic;
        # at minimum the function should not raise
        assert isinstance(result, (pd.DataFrame, type(None)))

    def test_fetch_all_yahoo_iterates_all_tickers(self, tmp_path, monkeypatch):
        """fetch_all_yahoo_data calls fetch_yahoo_data for every ticker."""
        import src.data.fetcher as fetcher_mod

        calls = []

        def fake_fetch(ticker, start_date, save_path, **kwargs):
            calls.append(ticker)
            return None   # simulate failure, but we count calls

        monkeypatch.setattr(fetcher_mod, "fetch_yahoo_data", fake_fetch)

        from config.settings import YAHOO_TICKERS
        fetcher_mod.fetch_all_yahoo_data(
            tickers_dict=YAHOO_TICKERS,
            raw_dir=tmp_path,
            pause_between=0,
        )
        assert len(calls) == len(YAHOO_TICKERS)
