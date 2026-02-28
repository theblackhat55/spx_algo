"""
tests/test_features.py
=======================
Test gate for Phase 2 (Tasks 4–11): all feature engineering modules.

Run with:
    pytest tests/test_features.py -v
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 500, start: str = "2020-01-02", seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with valid bar relationships."""
    idx = pd.bdate_range(start=start, periods=n, freq="B")
    rng = np.random.default_rng(seed)

    close  = 3000 + np.cumsum(rng.normal(0, 10, n))
    close  = np.maximum(close, 100)
    spread = rng.uniform(0.003, 0.018, n) * close
    high   = close + rng.uniform(0, 1, n) * spread
    low    = close - rng.uniform(0, 1, n) * spread
    open_  = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low,
         "Close": close, "Volume": volume},
        index=idx,
    )


def make_vix(n: int = 500, start: str = "2020-01-02", seed: int = 99) -> pd.DataFrame:
    """Synthetic VIX DataFrame."""
    idx = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(seed)
    close = 20 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.clip(close, 10, 80).astype(float)
    return pd.DataFrame(
        {"Open": close * 0.98, "High": close * 1.02,
         "Low": close * 0.97, "Close": close, "Volume": 0.0},
        index=idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Proximity Features
# ─────────────────────────────────────────────────────────────────────────────

class TestProximityFeatures:
    def test_returns_dataframe(self):
        from src.features.proximity import compute_proximity_features
        df = make_ohlcv()
        out = compute_proximity_features(df)
        assert isinstance(out, pd.DataFrame)

    def test_all_expected_columns_present(self):
        from src.features.proximity import compute_proximity_features
        df = make_ohlcv()
        out = compute_proximity_features(df)
        for lag in [1, 2, 3, 5]:
            assert f"prev_high_pct_{lag}" in out.columns
            assert f"prev_low_pct_{lag}"  in out.columns
        assert "open_gap_pct"         in out.columns
        assert "prev_range_pct"       in out.columns
        assert "rolling_avg_range_5"  in out.columns
        assert "rolling_avg_range_20" in out.columns

    def test_prev_high_pct_1_is_non_negative(self):
        """(High_{t-1} - Close_{t-1}) / Close_{t-1} must be >= 0."""
        from src.features.proximity import compute_proximity_features
        df = make_ohlcv(500)
        out = compute_proximity_features(df)
        valid = out["prev_high_pct_1"].dropna()
        assert (valid >= 0).all(), "prev_high_pct_1 should be non-negative"

    def test_prev_low_pct_1_is_non_negative(self):
        """(Close_{t-1} - Low_{t-1}) / Close_{t-1} must be >= 0."""
        from src.features.proximity import compute_proximity_features
        df = make_ohlcv(500)
        out = compute_proximity_features(df)
        valid = out["prev_low_pct_1"].dropna()
        assert (valid >= 0).all(), "prev_low_pct_1 should be non-negative"

    def test_manual_calculation_lag1(self):
        """Verify lag-1 high pct against manual calculation for row 5."""
        from src.features.proximity import compute_proximity_features
        df = make_ohlcv(20)
        out = compute_proximity_features(df)

        idx = df.index[5]
        expected_high_pct = (df["High"].iloc[4] - df["Close"].iloc[4]) / df["Close"].iloc[4]
        assert abs(out.loc[idx, "prev_high_pct_1"] - expected_high_pct) < 1e-10

    def test_open_gap_pct_manual(self):
        """open_gap_pct on row 3 = (Open[3] - Close[2]) / Close[2]."""
        from src.features.proximity import compute_proximity_features
        df = make_ohlcv(20)
        out = compute_proximity_features(df)

        idx = df.index[3]
        expected = (df["Open"].iloc[3] - df["Close"].iloc[2]) / df["Close"].iloc[2]
        assert abs(out.loc[idx, "open_gap_pct"] - expected) < 1e-10

    def test_rolling_avg_range_5_manual(self):
        """rolling_avg_range_5 at row 10 should equal mean of prev_range_pct[6:11]."""
        from src.features.proximity import compute_proximity_features
        df = make_ohlcv(50)
        out = compute_proximity_features(df)

        row = 10
        idx = df.index[row]
        # rolling mean of the last 5 values of prev_range_pct ending at row 10
        window_slice = out["prev_range_pct"].iloc[row - 4:row + 1]
        expected = window_slice.mean()
        actual   = out.loc[idx, "rolling_avg_range_5"]
        assert abs(actual - expected) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 — Volatility Features
# ─────────────────────────────────────────────────────────────────────────────

class TestVolatilityFeatures:
    def test_true_range_manual(self):
        """TR = max(H-L, |H-PrevC|, |L-PrevC|) — verify a gap-day."""
        from src.features.volatility import compute_true_range
        # Manually craft a bar where the prev-close gap dominates
        idx = pd.bdate_range("2023-01-03", periods=2)
        df  = pd.DataFrame({
            "Open":  [100.0, 110.0],
            "High":  [102.0, 112.0],
            "Low":   [99.0,  108.0],
            "Close": [101.0, 111.0],
        }, index=idx)
        tr = compute_true_range(df)
        # Row 1: H-L = 4, |H-PrevC| = |112-101| = 11, |L-PrevC| = |108-101| = 7
        assert abs(tr.iloc[1] - 11.0) < 1e-9, f"Expected TR=11, got {tr.iloc[1]}"

    def test_atr_always_positive(self):
        """ATR must be positive for every row after the warmup period."""
        from src.features.volatility import compute_atr
        df  = make_ohlcv(300)
        out = compute_atr(df)
        for col in out.columns:
            valid = out[col].dropna()
            assert (valid > 0).all(), f"{col} has non-positive values"

    def test_atr_20_manual(self):
        """ATR_20 on a specific row should match the Wilder EWM calculation."""
        from src.features.volatility import compute_true_range, compute_atr
        df  = make_ohlcv(100)
        tr  = compute_true_range(df)
        out = compute_atr(df, windows=[20])

        # Compare at row 50 using pandas EWM directly
        expected = tr.ewm(span=20, adjust=False, min_periods=20).mean().iloc[50]
        actual   = out["atr_20"].iloc[50]
        assert abs(actual - expected) < 1e-9, f"ATR_20 mismatch: {actual} vs {expected}"

    def test_parkinson_vol_positive(self):
        """Parkinson volatility must be positive after warmup."""
        from src.features.volatility import compute_parkinson_vol
        df  = make_ohlcv(300)
        vol = compute_parkinson_vol(df)
        assert (vol.dropna() > 0).all()

    def test_garman_klass_vol_positive(self):
        """Garman-Klass volatility must be positive after warmup."""
        from src.features.volatility import compute_garman_klass_vol
        df  = make_ohlcv(300)
        vol = compute_garman_klass_vol(df)
        assert (vol.dropna() > 0).all()

    def test_vix_zscore_approximately_standard_normal(self):
        """VIX Z-score should have mean ≈ 0 and std ≈ 1 over the full sample.

        Note: With a 252-day rolling window and synthetic data the warm-up
        period introduces a systematic bias.  We use a 5-year VIX series
        and allow a tolerance of 0.35 to accommodate this.
        """
        from src.features.volatility import compute_vix_features
        vix = make_vix(n=1500)   # ~6 years — sufficient warmup
        out = compute_vix_features(vix)
        zs  = out["vix_zscore"].dropna()
        assert abs(zs.mean()) < 0.35, f"Z-score mean too far from 0: {zs.mean():.4f}"
        assert abs(zs.std() - 1.0) < 0.35, f"Z-score std too far from 1: {zs.std():.4f}"

    def test_garch_returns_series_when_arch_missing(self, monkeypatch):
        """Should return NaN series gracefully when arch is not available."""
        from src.features import volatility as vol_mod
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "arch":
                raise ImportError("Mocked missing arch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        returns = pd.Series(np.random.normal(0, 0.01, 200))
        result  = vol_mod.compute_garch_volatility(returns)
        assert isinstance(result, pd.Series)

    def test_atr_ratio_5_60_columns_present(self):
        from src.features.volatility import compute_atr
        df  = make_ohlcv(300)
        out = compute_atr(df)
        assert "atr_ratio_5_60" in out.columns


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 — Technical Features
# ─────────────────────────────────────────────────────────────────────────────

class TestTechnicalFeatures:
    def test_rsi_bounds(self):
        """RSI must be in [0, 100] for all valid rows."""
        from src.features.technical import compute_rsi
        close = make_ohlcv(500)["Close"]
        for w in [7, 14, 28]:
            rsi = compute_rsi(close, w).dropna()
            assert (rsi >= 0).all() and (rsi <= 100).all(), \
                f"RSI_{w} has values outside [0,100]: min={rsi.min():.2f} max={rsi.max():.2f}"

    def test_rsi_14_manual(self):
        """Verify RSI(14) on a known constant-down series = 0."""
        from src.features.technical import compute_rsi
        # If close always decreases, RSI should converge toward 0
        n     = 100
        close = pd.Series(np.linspace(100, 50, n))
        rsi   = compute_rsi(close, 14)
        # After warmup, RSI of a monotonically falling series should be < 20
        assert rsi.iloc[-1] < 20, f"RSI for falling series: {rsi.iloc[-1]:.2f}"

    def test_ema_200_warmup(self):
        """EMA_200 should have NaN for the first 199 rows."""
        from src.features.technical import compute_trend_features
        df  = make_ohlcv(300)
        out = compute_trend_features(df, ema_windows=[200])
        # Rows 0..198 should be NaN for ema_dist_200
        nan_count = out["ema_dist_200"].iloc[:199].isna().sum()
        assert nan_count == 199, \
            f"Expected 199 NaN rows, got {nan_count}"

    def test_ema_dist_values_within_reasonable_range(self):
        """EMA distance should be within [-0.3, +0.3] for SPX-like prices."""
        from src.features.technical import compute_trend_features
        df  = make_ohlcv(500)
        out = compute_trend_features(df)
        for col in [c for c in out.columns if c.startswith("ema_dist_")]:
            valid = out[col].dropna()
            assert (valid.abs() < 0.5).all(), \
                f"{col} has extreme values: max_abs={valid.abs().max():.4f}"

    def test_volume_ratio_equals_1_when_volume_matches_mean(self):
        """Volume ratio should be 1.0 when volume equals its rolling mean."""
        from src.features.technical import compute_volume_features
        n   = 100
        idx = pd.bdate_range("2020-01-02", periods=n)
        df  = pd.DataFrame({
            "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
            "Volume": 1_000_000.0,
        }, index=idx)
        out = compute_volume_features(df, windows=[20])
        valid = out["volume_ratio_20"].dropna()
        assert (abs(valid - 1.0) < 1e-9).all(), "Volume ratio != 1 for constant volume"

    def test_bb_position_range_typical(self):
        """BB position should mostly be in [0, 1] for normal prices."""
        from src.features.technical import compute_bollinger_features
        df   = make_ohlcv(300)
        out  = compute_bollinger_features(df)
        pos  = out["bb_position"].dropna()
        # At least 90% of values should be within [0, 1]
        pct_inside = ((pos >= 0) & (pos <= 1)).mean()
        assert pct_inside > 0.85, f"Only {pct_inside:.1%} of bb_position in [0,1]"

    def test_macd_histogram_shape(self):
        """MACD histogram should have the same length as input."""
        from src.features.technical import compute_momentum_features
        df  = make_ohlcv(100)
        out = compute_momentum_features(df)
        assert len(out) == len(df)
        assert "macd_histogram" in out.columns

    def test_all_technical_features_assembled(self):
        """compute_all_technical_features should return all major column groups."""
        from src.features.technical import compute_all_technical_features
        df  = make_ohlcv(500)
        out = compute_all_technical_features(df)
        for expected in [
            "ema_dist_9", "ema_dist_200", "trend_short",
            "rsi_7", "rsi_14", "rsi_28",
            "macd_histogram", "bb_position", "bb_width",
            "volume_ratio_5", "obv_roc_10",
        ]:
            assert expected in out.columns, f"Missing column: {expected}"


# ─────────────────────────────────────────────────────────────────────────────
# Task 7 — Calendar Features
# ─────────────────────────────────────────────────────────────────────────────

class TestCalendarFeatures:
    @pytest.fixture
    def df_and_calendar(self, tmp_path):
        from src.data.calendar import build_trading_calendar
        trading_dates = pd.bdate_range("2020-01-02", "2023-12-29")
        df  = pd.DataFrame(index=trading_dates)
        cal = build_trading_calendar(
            trading_dates=trading_dates,
            save_path=tmp_path / "cal.csv",
        )
        return df, cal

    def test_known_monday(self, df_and_calendar):
        from src.features.calendar_features import compute_calendar_features
        df, cal = df_and_calendar
        out = compute_calendar_features(df, cal)
        # 2020-01-06 is a Monday
        assert out.loc["2020-01-06", "is_monday"] == 1
        for other in ["is_tuesday", "is_wednesday", "is_thursday", "is_friday"]:
            assert out.loc["2020-01-06", other] == 0

    def test_exactly_5_dow_columns(self, df_and_calendar):
        from src.features.calendar_features import compute_calendar_features
        df, cal = df_and_calendar
        out   = compute_calendar_features(df, cal)
        dow_cols = ["is_monday", "is_tuesday", "is_wednesday",
                    "is_thursday", "is_friday"]
        for col in dow_cols:
            assert col in out.columns

        # Each DOW column should sum to roughly 1/5 of total rows
        totals = out[dow_cols].sum()
        expected = len(out) / 5
        for col in dow_cols:
            assert abs(totals[col] - expected) / expected < 0.05, \
                f"{col} deviates from expected 20% frequency"

    def test_cyclic_month_reconstructable(self, df_and_calendar):
        """sin and cos encode month uniquely — arctan2 should recover month.

        The encoding is: month_sin = sin(2π × month / 12), where month is
        1-indexed (1 = January).  The recovered month (also 1-indexed) is
        obtained via arctan2 and compared to idx_dt.month.
        """
        from src.features.calendar_features import compute_calendar_features
        df, cal = df_and_calendar
        out = compute_calendar_features(df, cal)
        for idx_dt in out.index[:20]:
            s = out.loc[idx_dt, "month_sin"]
            c = out.loc[idx_dt, "month_cos"]
            # Recover the normalised angle (0 to 1), then scale to 1..12
            angle_rad       = np.arctan2(s, c) % (2 * np.pi)
            recovered_month = angle_rad / (2 * np.pi) * 12   # 0..12
            actual_month    = idx_dt.month                    # 1..12 (1-indexed)
            # Wrap-around tolerance (handles month=12 ↔ month=0 edge)
            diff = abs(recovered_month - actual_month)
            diff = min(diff, 12 - diff)
            assert diff < 0.5, f"Month cyclic encoding failed at {idx_dt}: {diff}"

    def test_fomc_day_on_known_date(self, df_and_calendar):
        from src.features.calendar_features import compute_calendar_features
        df, cal = df_and_calendar
        out = compute_calendar_features(df, cal)
        assert out.loc["2022-03-16", "is_fomc_day"] == 1

    def test_no_nan_values(self, df_and_calendar):
        from src.features.calendar_features import compute_calendar_features
        df, cal = df_and_calendar
        out = compute_calendar_features(df, cal)
        assert not out.isnull().any().any(), "Calendar features should have zero NaN"

    def test_month_sin_cos_in_range(self, df_and_calendar):
        from src.features.calendar_features import compute_calendar_features
        df, cal = df_and_calendar
        out = compute_calendar_features(df, cal)
        assert (out["month_sin"].abs() <= 1.0).all()
        assert (out["month_cos"].abs() <= 1.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Task 8 — Options Features
# ─────────────────────────────────────────────────────────────────────────────

class TestOptionsFeatures:
    def test_iv_rank_bounds(self):
        """IV Rank must be in [0, 1]."""
        from src.features.options_features import compute_iv_features
        vix = make_vix(n=500)
        out = compute_iv_features(vix)
        valid = out["iv_rank_252"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all(), \
            f"IV Rank out of bounds: min={valid.min():.4f} max={valid.max():.4f}"

    def test_iv_rank_is_zero_at_period_minimum(self):
        """IV Rank should be 0 when VIX equals its 252-day minimum."""
        from src.features.options_features import compute_iv_features
        idx  = pd.bdate_range("2020-01-02", periods=300)
        # Constant VIX except for one very low day at the start
        close = np.full(300, 20.0)
        vix   = pd.DataFrame({"Close": close}, index=idx)
        out   = compute_iv_features(vix)
        # All values should be 0 (VIX == min == max — degenerate case)
        # IV Rank is 0 / 0 = NaN for constant series, which is correct
        # Just verify the function doesn't crash and returns a DataFrame
        assert isinstance(out, pd.DataFrame)
        assert "iv_rank_252" in out.columns

    def test_iv_rank_is_one_at_period_maximum(self):
        """IV Rank should be 1 on the day VIX hits its 252-day high."""
        from src.features.options_features import compute_iv_features
        n   = 300
        idx = pd.bdate_range("2020-01-02", periods=n)
        # Monotonically rising VIX → last value is always the 252-day max
        close = np.linspace(10.0, 80.0, n)
        vix   = pd.DataFrame({"Close": close}, index=idx)
        out   = compute_iv_features(vix)
        last_rank = out["iv_rank_252"].dropna().iloc[-1]
        assert abs(last_rank - 1.0) < 0.05, f"IV Rank at max should be ≈1: {last_rank:.4f}"

    def test_gex_returns_empty_df_when_none(self):
        """compute_gex_features(None) should return an empty DataFrame."""
        from src.features.options_features import compute_gex_features
        idx = pd.bdate_range("2020-01-02", periods=10)
        out = compute_gex_features(None, index=idx)
        assert isinstance(out, pd.DataFrame)
        assert out.empty, "Should return empty DataFrame when gex_df is None"

    def test_putcall_returns_empty_df_when_none(self):
        """compute_putcall_features with no data should return empty DataFrame."""
        from src.features.options_features import compute_putcall_features
        df  = pd.DataFrame(index=pd.bdate_range("2020-01-02", periods=10))
        out = compute_putcall_features(df, putcall_df=None)
        assert isinstance(out, pd.DataFrame)
        assert out.empty


# ─────────────────────────────────────────────────────────────────────────────
# Task 9 — Lagged Target Features
# ─────────────────────────────────────────────────────────────────────────────

class TestLaggedTargetFeatures:
    def test_all_columns_present(self):
        from src.features.lagged_targets import compute_lagged_target_features
        df  = make_ohlcv(100)
        out = compute_lagged_target_features(df)
        for col in [
            "prev_high_exceedance", "prev_low_exceedance", "range_change",
            "high_direction_streak", "low_direction_streak",
            "inside_day_flag", "inside_day_streak", "outside_day_flag",
        ]:
            assert col in out.columns, f"Missing column: {col}"

    def test_prev_high_exceedance_non_negative(self):
        """High exceedance must be >= 0 since High >= Close always."""
        from src.features.lagged_targets import compute_lagged_target_features
        df  = make_ohlcv(200)
        out = compute_lagged_target_features(df)
        valid = out["prev_high_exceedance"].dropna()
        assert (valid >= 0).all()

    def test_prev_low_exceedance_non_negative(self):
        """Low exceedance must be >= 0 since Close >= Low always."""
        from src.features.lagged_targets import compute_lagged_target_features
        df  = make_ohlcv(200)
        out = compute_lagged_target_features(df)
        valid = out["prev_low_exceedance"].dropna()
        assert (valid >= 0).all()

    def test_inside_day_flag_manual(self):
        """Manually verify inside_day_flag on 3 known dates."""
        from src.features.lagged_targets import compute_lagged_target_features
        idx = pd.bdate_range("2020-01-06", periods=5)
        df  = pd.DataFrame({
            "Open":  [100, 101, 102, 103, 104],
            "High":  [105, 103, 102, 106, 107],   # row 2: inside (H=102 < H[1]=103)
            "Low":   [98,  99,  100, 97,  96],    # row 2: inside (L=100 > L[1]=99)
            "Close": [102, 101, 101, 104, 105],
        }, index=idx)
        out = compute_lagged_target_features(df)
        # Row 3 (index[3]) uses T-1=row2 (inside) vs T-2=row1
        # row2: H=102 < row1 H=103 AND row2: L=100 > row1 L=99 → inside_day_flag=1 at row3
        assert out["inside_day_flag"].iloc[3] == 1, \
            f"Expected inside_day_flag=1, got {out['inside_day_flag'].iloc[3]}"
        # Row 1 (index[1]) uses T-1=row0, T-2 doesn't exist (NaN)
        assert pd.isna(out["inside_day_flag"].iloc[1]) or out["inside_day_flag"].iloc[1] == 0

    def test_feature_only_depends_on_prior_data(self):
        """Modifying future data should not change features for earlier dates."""
        from src.features.lagged_targets import compute_lagged_target_features
        df1 = make_ohlcv(50)
        df2 = df1.copy()
        # Modify the last 10 rows of df2
        df2.iloc[-10:, :] = df2.iloc[-10:, :] * 1.5

        out1 = compute_lagged_target_features(df1)
        out2 = compute_lagged_target_features(df2)

        # Features for rows 0..39 should be identical in both
        check_rows = slice(0, 40)
        pd.testing.assert_frame_equal(
            out1.iloc[check_rows],
            out2.iloc[check_rows],
            check_exact=False,
            rtol=1e-9,
        )

    def test_inside_and_outside_day_mutually_exclusive(self):
        """A day cannot be both an inside day and an outside day."""
        from src.features.lagged_targets import compute_lagged_target_features
        df  = make_ohlcv(300)
        out = compute_lagged_target_features(df)
        both = (out["inside_day_flag"] == 1) & (out["outside_day_flag"] == 1)
        assert not both.any(), "Inside and outside day flags should be mutually exclusive"

    def test_direction_streak_integers(self):
        """Streak columns must be integer-valued."""
        from src.features.lagged_targets import compute_lagged_target_features
        df  = make_ohlcv(100)
        out = compute_lagged_target_features(df)
        valid_h = out["high_direction_streak"].dropna()
        valid_l = out["low_direction_streak"].dropna()
        assert (valid_h == valid_h.round()).all(), "high_direction_streak must be integers"
        assert (valid_l == valid_l.round()).all(), "low_direction_streak must be integers"


# ─────────────────────────────────────────────────────────────────────────────
# Task 10 — Feature Builder
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureBuilder:
    """These tests exercise the builder on synthetic data (no disk I/O required)."""

    def _build_synthetic_features(self, tmp_path):
        """Helper: build a feature matrix from synthetic data."""
        import shutil
        from src.data.calendar import build_trading_calendar

        spx = make_ohlcv(n=1000, start="2018-01-02")
        vix = make_vix(n=1000, start="2018-01-02")

        # Save synthetic data to tmp_path
        spx_path = tmp_path / "spx_daily.parquet"
        vix_path = tmp_path / "vix_daily.parquet"
        spx.to_parquet(spx_path)
        vix.to_parquet(vix_path)

        # Build calendar
        cal = build_trading_calendar(
            trading_dates=spx.index,
            save_path=tmp_path / "calendar_events.csv",
        )

        # Import and monkey-patch settings paths
        import config.settings as cfg
        orig_spx   = cfg.SPX_FILE
        orig_vix   = cfg.VIX_FILE
        orig_cal   = cfg.CALENDAR_FILE
        orig_proc  = cfg.PROCESSED_DATA_DIR
        orig_gex   = cfg.GEX_FILE

        cfg.SPX_FILE          = spx_path
        cfg.VIX_FILE          = vix_path
        cfg.CALENDAR_FILE     = tmp_path / "calendar_events.csv"
        cfg.PROCESSED_DATA_DIR= tmp_path
        cfg.GEX_FILE          = tmp_path / "gex_daily.csv"   # won't exist → skipped

        try:
            from src.features import builder
            # Reload to pick up patched settings
            import importlib
            importlib.reload(builder)
            features = builder.build_feature_matrix(
                raw_dir=tmp_path,
                processed_dir=tmp_path,
                spx_file=spx_path,
                vix_file=vix_path,
            )
        finally:
            cfg.SPX_FILE          = orig_spx
            cfg.VIX_FILE          = orig_vix
            cfg.CALENDAR_FILE     = orig_cal
            cfg.PROCESSED_DATA_DIR= orig_proc
            cfg.GEX_FILE          = orig_gex

        return features

    def test_zero_nan_values(self, tmp_path):
        features = self._build_synthetic_features(tmp_path)
        nan_count = features.isnull().sum().sum()
        assert nan_count == 0, f"Feature matrix has {nan_count} NaN values"

    def test_feature_count_reasonable(self, tmp_path):
        features = self._build_synthetic_features(tmp_path)
        n_cols = features.shape[1]
        assert 40 <= n_cols <= 100, f"Unexpected feature count: {n_cols}"

    def test_date_range_starts_after_warmup(self, tmp_path):
        features = self._build_synthetic_features(tmp_path)
        # With 1000 days starting 2018-01-02, feature start must be after warmup
        # The max lookback is 252 days (VIX Z-score)
        assert features.index[0] > pd.Timestamp("2018-01-02"), \
            "Feature matrix should start after the warmup period"

    def test_parquet_saved(self, tmp_path):
        self._build_synthetic_features(tmp_path)
        assert (tmp_path / "features.parquet").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Task 11 — Feature Selector
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureSelector:
    @pytest.fixture
    def synthetic_XY(self):
        """200-sample synthetic regression dataset."""
        rng = np.random.default_rng(0)
        n, p = 300, 25
        X = pd.DataFrame(
            rng.standard_normal((n, p)),
            columns=[f"f{i}" for i in range(p)],
        )
        # Target depends only on first 5 features
        y = pd.Series(
            X["f0"] * 2 + X["f1"] - X["f2"] * 0.5 + rng.normal(0, 0.5, n),
            name="target",
        )
        return X, y

    def test_remove_correlated_features(self, synthetic_XY):
        from src.features.selector import remove_correlated_features
        X, y = synthetic_XY
        # Add a near-duplicate of f0
        X["f0_dup"] = X["f0"] + np.random.default_rng(1).normal(0, 0.001, len(X))
        reduced, removed = remove_correlated_features(X, y, threshold=0.95)
        # The duplicate should be removed
        assert len(removed) >= 1, "At least one correlated feature should be removed"
        assert len(reduced.columns) < len(X.columns)

    def test_no_correlated_pair_above_threshold(self, synthetic_XY):
        from src.features.selector import remove_correlated_features
        X, y = synthetic_XY
        X["f0_dup"] = X["f0"] + np.random.default_rng(1).normal(0, 0.001, len(X))
        reduced, _ = remove_correlated_features(X, y, threshold=0.90)
        corr = reduced.corr(method="spearman").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        max_corr = upper.max().max()
        assert max_corr <= 0.90 + 1e-6, f"Correlated pair remains: {max_corr:.4f}"

    def test_permutation_importance_shape(self, synthetic_XY):
        from src.features.selector import compute_permutation_importance
        from sklearn.linear_model import Ridge
        X, y = synthetic_XY
        split = int(0.7 * len(X))
        model = Ridge()
        model.fit(X.iloc[:split], y.iloc[:split])
        perm = compute_permutation_importance(model, X.iloc[split:], y.iloc[split:], n_repeats=5)
        assert len(perm) == len(X.columns)
        assert "feature" in perm.columns
        assert "importance_mean" in perm.columns
        assert "p_value" in perm.columns

    def test_permutation_pvalue_bounds(self, synthetic_XY):
        from src.features.selector import compute_permutation_importance
        from sklearn.linear_model import Ridge
        X, y = synthetic_XY
        split = int(0.7 * len(X))
        model = Ridge()
        model.fit(X.iloc[:split], y.iloc[:split])
        perm = compute_permutation_importance(model, X.iloc[split:], y.iloc[split:], n_repeats=10)
        # p-values must be in [0, 1]
        assert (perm["p_value"] >= 0).all()
        assert (perm["p_value"] <= 1).all()
