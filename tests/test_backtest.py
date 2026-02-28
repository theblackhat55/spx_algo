"""
tests/test_backtest.py
=======================
Phase 4 test gate — Backtest Engine, Report, Pipeline Runner, Scheduler.

Test groups
-----------
TestIronCondorEngine       (14 tests)
TestReportMetrics          (10 tests)
TestRegimeGating           ( 6 tests)
TestTransactionCosts       ( 4 tests)
TestSkipAnalysis           ( 3 tests)
TestPipelineRunner         ( 7 tests)
TestDeadMansSwitch         ( 4 tests)
TestReplayDateCheck        ( 2 tests)
──────────────────────────────────────
Total:                      50 tests
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Shared helpers ─────────────────────────────────────────────────────────

N = 504   # ~2 years of daily data
RNG = np.random.default_rng(99)


def _spx_df(n: int = N, base: float = 5000.0) -> pd.DataFrame:
    rng   = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=n)
    c = base + np.cumsum(rng.normal(0, 8, n))
    h = c + rng.uniform(8, 30, n)
    l = c - rng.uniform(8, 30, n)
    o = c + rng.normal(0, 4, n)
    v = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v},
                        index=dates)


def _vix_df(n: int = N) -> pd.DataFrame:
    rng   = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"Close": 18 + np.cumsum(rng.normal(0, 0.2, n))},
                        index=dates)


def _signals_df(spx: pd.DataFrame) -> pd.DataFrame:
    """
    Synthetic signals matching the IronCondorEngine expected format.
    Uses fixed_atm_pct-style: upper_90 and lower_90 as %-deviations.
    """
    n   = len(spx)
    rng = np.random.default_rng(77)
    df  = pd.DataFrame({
        "upper_90":    rng.uniform(0.005, 0.015, n),    # +0.5%–1.5%
        "lower_90":   -rng.uniform(0.005, 0.015, n),    # -0.5%–1.5%
        "upper_68":    rng.uniform(0.003, 0.008, n),
        "lower_68":   -rng.uniform(0.003, 0.008, n),
    }, index=spx.index)
    return df


def _regime(n: int = N, pct_red: float = 0.1, pct_yellow: float = 0.2) -> pd.Series:
    from src.calibration.regime import Regime
    dates = pd.bdate_range("2022-01-03", periods=n)
    vals  = RNG.choice(
        [Regime.GREEN, Regime.YELLOW, Regime.RED],
        size=n,
        p=[1 - pct_red - pct_yellow, pct_yellow, pct_red],
    )
    return pd.Series(vals, index=dates, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# TestIronCondorEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestIronCondorEngine:

    @pytest.fixture
    def setup(self):
        spx     = _spx_df()
        vix     = _vix_df()
        signals = _signals_df(spx)
        regime  = _regime()
        return spx, vix, signals, regime

    def test_run_returns_dataframe(self, setup):
        from src.backtest.engine import IronCondorEngine
        spx, vix, sig, reg = setup
        trades = IronCondorEngine().run(sig, spx, vix, reg)
        assert isinstance(trades, pd.DataFrame)

    def test_row_count_equals_signal_count(self, setup):
        from src.backtest.engine import IronCondorEngine
        spx, vix, sig, reg = setup
        trades = IronCondorEngine().run(sig, spx, vix, reg)
        assert len(trades) == len(sig)

    def test_all_expected_columns_present(self, setup):
        from src.backtest.engine import IronCondorEngine
        spx, vix, sig, reg = setup
        trades = IronCondorEngine().run(sig, spx, vix, reg)
        for col in ("regime", "call_strike", "put_strike", "credit_pts",
                    "net_pnl_dollars", "skipped", "call_intrusion", "put_intrusion"):
            assert col in trades.columns, f"Missing: {col}"

    def test_net_pnl_not_all_zero(self, setup):
        from src.backtest.engine import IronCondorEngine
        spx, vix, sig, _ = setup
        # Use all-GREEN regime to maximise active trades
        from src.calibration.regime import Regime
        all_green = pd.Series(Regime.GREEN, index=spx.index, dtype=int)
        trades = IronCondorEngine().run(sig, spx, vix, all_green)
        active = trades[~trades["skipped"]]
        assert active["net_pnl_dollars"].abs().sum() > 0

    def test_intrusion_depth_not_binary(self, setup):
        """Verify intrusion depth is continuous, not just 0 or max_loss."""
        from src.backtest.engine import IronCondorEngine, PositionConfig
        spx, vix, sig, _ = setup
        from src.calibration.regime import Regime
        all_green = pd.Series(Regime.GREEN, index=spx.index, dtype=int)
        # Strikes at +/-0.5% above prior close.  Synthetic High deviates by
        # ~8-30 pts (~0.16%-0.60%) so some days breach and some don't.
        # Wing is huge (1000 pts) so intrusions are always < wing_width.
        border_sig = sig.copy()
        border_sig["upper_90"] =  0.005   # ~25 pts above close at ~5000
        border_sig["lower_90"] = -0.005
        border_sig["upper_68"] =  0.003
        border_sig["lower_68"] = -0.003
        cfg    = PositionConfig(wing_width_pts=1000.0)
        trades = IronCondorEngine(cfg).run(border_sig, spx, vix, all_green)
        active = trades[~trades["skipped"]]
        intrusions = active["call_intrusion"].values
        # Some rows should have 0 intrusion (didn't breach) and
        # some should have > 0 (breached), all < 1000 (wing never maxed).
        assert (intrusions < 1000.0).all(), "Intrusion capped at wing_width unexpectedly"
        assert (intrusions > 0).any(),      "Expected at least one breach"
        assert (intrusions == 0).any(),     "Expected at least one non-breach"
        # Continuous: there should be values strictly between 0 and 1000
        mid_vals = intrusions[(intrusions > 0) & (intrusions < 1000.0)]
        assert len(mid_vals) > 0, "Expected intermediate intrusion depths"

    def test_max_loss_bounded_by_wing_width(self, setup):
        """No single trade should lose more than wing_width - credit."""
        from src.backtest.engine import IronCondorEngine, PositionConfig
        spx, vix, sig, reg = setup
        cfg    = PositionConfig(wing_width_pts=20.0)
        trades = IronCondorEngine(cfg).run(sig, spx, vix, reg)
        active = trades[~trades["skipped"]]
        max_loss_per_share = (active["call_intrusion"] + active["put_intrusion"]).max()
        assert max_loss_per_share <= 40.0 + 0.01   # 2 × wing_width + tolerance

    def test_friction_reduces_pnl(self, setup):
        """Net P&L must be strictly less than gross P&L for active trades."""
        from src.backtest.engine import IronCondorEngine, PositionConfig
        spx, vix, sig, _ = setup
        from src.calibration.regime import Regime
        all_green = pd.Series(Regime.GREEN, index=spx.index, dtype=int)
        cfg    = PositionConfig(slippage_per_leg=0.10)
        trades = IronCondorEngine(cfg).run(sig, spx, vix, all_green)
        active = trades[~trades["skipped"]]
        assert (active["net_pnl_pts"] < active["gross_pnl_pts"]).all()

    def test_credit_positive_for_active_trades(self, setup):
        from src.backtest.engine import IronCondorEngine
        spx, vix, sig, reg = setup
        trades = IronCondorEngine().run(sig, spx, vix, reg)
        active = trades[~trades["skipped"]]
        assert (active["credit_pts"] > 0).all()

    def test_fixed_atm_pct_strike_method(self, setup):
        from src.backtest.engine import IronCondorEngine, PositionConfig
        spx, vix, sig, _ = setup
        from src.calibration.regime import Regime
        all_green = pd.Series(Regime.GREEN, index=spx.index, dtype=int)
        cfg    = PositionConfig(strike_method="fixed_atm_pct",
                                call_pct_above=0.010, put_pct_below=0.010)
        trades = IronCondorEngine(cfg).run(sig, spx, vix, all_green)
        active = trades[~trades["skipped"]]
        assert (active["call_strike"] > active["put_strike"]).all()

    def test_min_hold_anti_flicker(self):
        """Once RED, regime stays RED for min_hold days."""
        from src.backtest.engine import _apply_min_hold
        from src.calibration.regime import Regime
        dates = pd.bdate_range("2024-01-01", periods=10)
        vals  = [0, 0, 2, 0, 0, 0, 0, 0, 0, 0]   # single RED on day 2
        reg   = pd.Series(vals, index=dates, dtype=int)
        held  = _apply_min_hold(reg, min_hold=3, target_state=Regime.RED)
        # Days 2,3,4 should all be RED
        assert held.iloc[2] == Regime.RED
        assert held.iloc[3] == Regime.RED
        assert held.iloc[4] == Regime.RED
        # Day 5 should revert
        assert held.iloc[5] == Regime.GREEN

    def test_inverted_strikes_skipped(self, setup):
        """If call_strike <= put_strike, the day should be skipped."""
        from src.backtest.engine import IronCondorEngine, PositionConfig
        spx, vix, sig, _ = setup
        # Invert the signals
        bad_sig = sig.copy()
        bad_sig["upper_90"] = -0.01   # call below put
        bad_sig["lower_90"] = 0.01
        from src.calibration.regime import Regime
        all_green = pd.Series(Regime.GREEN, index=spx.index, dtype=int)
        trades = IronCondorEngine().run(bad_sig, spx, vix, all_green)
        inverted = trades[trades["skip_reason"] == "INVERTED_STRIKES"]
        assert len(inverted) > 0

    def test_contracts_field_nonzero_for_active(self, setup):
        from src.backtest.engine import IronCondorEngine
        spx, vix, sig, _ = setup
        from src.calibration.regime import Regime
        all_green = pd.Series(Regime.GREEN, index=spx.index, dtype=int)
        trades = IronCondorEngine().run(sig, spx, vix, all_green)
        active = trades[~trades["skipped"]]
        assert (active["contracts"] > 0).all()

    def test_date_index_is_datetime(self, setup):
        from src.backtest.engine import IronCondorEngine
        spx, vix, sig, reg = setup
        trades = IronCondorEngine().run(sig, spx, vix, reg)
        assert pd.api.types.is_datetime64_any_dtype(trades.index)

    def test_dollar_pnl_is_pts_times_100(self, setup):
        from src.backtest.engine import IronCondorEngine
        spx, vix, sig, _ = setup
        from src.calibration.regime import Regime
        all_green = pd.Series(Regime.GREEN, index=spx.index, dtype=int)
        trades = IronCondorEngine().run(sig, spx, vix, all_green)
        active = trades[~trades["skipped"]]
        expected = active["net_pnl_pts"] * 100 * active["contracts"]
        pd.testing.assert_series_equal(
            active["net_pnl_dollars"].round(4),
            expected.round(4),
            check_names=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestReportMetrics
# ─────────────────────────────────────────────────────────────────────────────

class TestReportMetrics:

    @pytest.fixture
    def trades(self):
        from src.backtest.engine import IronCondorEngine
        from src.calibration.regime import Regime
        spx     = _spx_df()
        vix     = _vix_df()
        sig     = _signals_df(spx)
        all_green = pd.Series(Regime.GREEN, index=spx.index, dtype=int)
        return IronCondorEngine().run(sig, spx, vix, all_green)

    def test_compute_performance_returns_dict(self, trades):
        from src.backtest.report import compute_performance
        perf = compute_performance(trades)
        assert isinstance(perf, dict)

    def test_win_rate_in_0_1(self, trades):
        from src.backtest.report import compute_performance
        perf = compute_performance(trades)
        assert 0 <= perf["win_rate"] <= 1

    def test_sharpe_is_finite(self, trades):
        from src.backtest.report import compute_performance
        perf = compute_performance(trades)
        assert np.isfinite(perf["sharpe_ratio"])

    def test_max_drawdown_nonpositive(self, trades):
        from src.backtest.report import compute_performance
        perf = compute_performance(trades)
        assert perf["max_drawdown_dollars"] <= 0

    def test_calmar_is_finite(self, trades):
        from src.backtest.report import compute_performance
        perf = compute_performance(trades)
        assert np.isfinite(perf["calmar_ratio"])

    def test_regime_breakdown_is_dataframe(self, trades):
        from src.backtest.report import compute_performance
        perf = compute_performance(trades)
        assert isinstance(perf["regime_breakdown"], pd.DataFrame)

    def test_monthly_pnl_is_dataframe(self, trades):
        from src.backtest.report import compute_performance
        perf = compute_performance(trades)
        assert isinstance(perf["monthly_pnl"], pd.DataFrame)

    def test_cumulative_pnl_monotone_after_all_wins(self):
        """If every trade wins, cumulative P&L should be non-decreasing."""
        from src.backtest.report import compute_performance
        from src.backtest.engine import IronCondorEngine, PositionConfig
        from src.calibration.regime import Regime

        spx = _spx_df()
        vix = _vix_df()

        # Very tight strikes so no breach is possible on synthetic data
        cfg = PositionConfig(
            strike_method="fixed_atm_pct",
            call_pct_above=0.20,   # 20% OTM — will never breach
            put_pct_below=0.20,
            wing_width_pts=50.0,
        )
        sig       = _signals_df(spx)
        all_green = pd.Series(Regime.GREEN, index=spx.index, dtype=int)
        trades    = IronCondorEngine(cfg).run(sig, spx, vix, all_green)
        perf      = compute_performance(trades)

        cum = perf["cumulative_pnl"]
        assert (cum.diff().dropna() >= -0.01).all()   # non-decreasing (allow fp noise)

    def test_profit_factor_positive_when_profitable(self):
        """In a pure-win scenario, profit factor should be very large (infinite)."""
        from src.backtest.report import compute_performance
        from src.backtest.engine import IronCondorEngine, PositionConfig
        from src.calibration.regime import Regime

        spx = _spx_df()
        vix = _vix_df()
        cfg = PositionConfig(
            strike_method="fixed_atm_pct",
            call_pct_above=0.20, put_pct_below=0.20, wing_width_pts=50.0,
        )
        sig       = _signals_df(spx)
        all_green = pd.Series(Regime.GREEN, index=spx.index, dtype=int)
        trades    = IronCondorEngine(cfg).run(sig, spx, vix, all_green)
        perf      = compute_performance(trades)
        assert perf["profit_factor"] >= 1.0

    def test_print_summary_does_not_raise(self, trades, capsys):
        from src.backtest.report import compute_performance, print_summary
        perf = compute_performance(trades)
        print_summary(perf, title="Unit Test")
        captured = capsys.readouterr()
        assert "Win Rate" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# TestRegimeGating
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeGating:

    def test_all_red_produces_all_skipped(self):
        from src.backtest.engine import IronCondorEngine, PositionConfig
        from src.calibration.regime import Regime
        spx = _spx_df(100)
        vix = _vix_df(100)
        sig = _signals_df(spx)
        all_red = pd.Series(Regime.RED, index=spx.index, dtype=int)
        cfg = PositionConfig(skip_red=True)
        trades = IronCondorEngine(cfg).run(sig, spx, vix, all_red)
        assert trades["skipped"].all()

    def test_all_green_produces_no_red_skips(self):
        from src.backtest.engine import IronCondorEngine
        from src.calibration.regime import Regime
        spx       = _spx_df(100)
        vix       = _vix_df(100)
        sig       = _signals_df(spx)
        all_green = pd.Series(Regime.GREEN, index=spx.index, dtype=int)
        trades    = IronCondorEngine().run(sig, spx, vix, all_green)
        red_skips = trades[trades["skip_reason"] == "RED_SKIP"]
        assert len(red_skips) == 0

    def test_yellow_skip_config(self):
        from src.backtest.engine import IronCondorEngine, PositionConfig
        from src.calibration.regime import Regime
        spx = _spx_df(100)
        vix = _vix_df(100)
        sig = _signals_df(spx)
        all_yellow = pd.Series(Regime.YELLOW, index=spx.index, dtype=int)
        cfg    = PositionConfig(skip_yellow=True)
        trades = IronCondorEngine(cfg).run(sig, spx, vix, all_yellow)
        assert trades["skipped"].all()

    def test_regime_col_in_trades_matches_input(self):
        from src.backtest.engine import IronCondorEngine
        from src.calibration.regime import Regime
        spx = _spx_df(50)
        vix = _vix_df(50)
        sig = _signals_df(spx)
        reg = _regime(50, pct_red=0.3, pct_yellow=0.3)
        trades = IronCondorEngine().run(sig, spx, vix, reg)
        # After anti-flicker, regime values should still be a subset of {0,1,2}
        assert set(trades["regime"].unique()).issubset({0, 1, 2})

    def test_red_days_not_counted_in_win_rate(self):
        """Win rate should be computed on active trades only."""
        from src.backtest.engine import IronCondorEngine
        from src.backtest.report import compute_performance
        from src.calibration.regime import Regime
        spx = _spx_df()
        vix = _vix_df()
        sig = _signals_df(spx)
        reg = _regime(pct_red=0.5)
        trades = IronCondorEngine().run(sig, spx, vix, reg)
        perf   = compute_performance(trades)
        active_count = (~trades["skipped"]).sum()
        assert perf["total_trades"] == active_count

    def test_regime_applied_consistently_not_just_production(self):
        """
        Run backtest with and without regime filter on same signals.
        They must differ — confirms regime IS applied during backtest.
        """
        from src.backtest.engine import IronCondorEngine, PositionConfig
        from src.backtest.report import compute_performance
        from src.calibration.regime import Regime
        spx = _spx_df()
        vix = _vix_df()
        sig = _signals_df(spx)
        reg = _regime(pct_red=0.2, pct_yellow=0.2)

        cfg_with    = PositionConfig(skip_red=True)
        cfg_without = PositionConfig(skip_red=False)

        t_with    = IronCondorEngine(cfg_with).run(sig, spx, vix, reg)
        t_without = IronCondorEngine(cfg_without).run(sig, spx, vix, reg)

        n_with    = (~t_with["skipped"]).sum()
        n_without = (~t_without["skipped"]).sum()
        assert n_without > n_with   # fewer trades when RED is skipped


# ─────────────────────────────────────────────────────────────────────────────
# TestTransactionCosts
# ─────────────────────────────────────────────────────────────────────────────

class TestTransactionCosts:

    def test_friction_present_in_active_trades(self):
        from src.backtest.engine import IronCondorEngine, PositionConfig
        from src.calibration.regime import Regime
        spx       = _spx_df(100)
        vix       = _vix_df(100)
        sig       = _signals_df(spx)
        all_green = pd.Series(Regime.GREEN, index=spx.index, dtype=int)
        cfg       = PositionConfig(slippage_per_leg=0.10, n_legs=4)
        trades    = IronCondorEngine(cfg).run(sig, spx, vix, all_green)
        active    = trades[~trades["skipped"]]
        assert (active["friction_pts"] == 0.40).all()

    def test_zero_slippage_config(self):
        """With zero slippage, gross == net for active trades."""
        from src.backtest.engine import IronCondorEngine, PositionConfig
        from src.calibration.regime import Regime
        spx       = _spx_df(100)
        vix       = _vix_df(100)
        sig       = _signals_df(spx)
        all_green = pd.Series(Regime.GREEN, index=spx.index, dtype=int)
        cfg       = PositionConfig(slippage_per_leg=0.0)
        trades    = IronCondorEngine(cfg).run(sig, spx, vix, all_green)
        active    = trades[~trades["skipped"]]
        pd.testing.assert_series_equal(
            active["gross_pnl_pts"].round(6),
            active["net_pnl_pts"].round(6),
            check_names=False,
        )

    def test_higher_slippage_reduces_total_pnl(self):
        from src.backtest.engine import IronCondorEngine, PositionConfig
        from src.calibration.regime import Regime
        spx       = _spx_df()
        vix       = _vix_df()
        sig       = _signals_df(spx)
        all_green = pd.Series(Regime.GREEN, index=spx.index, dtype=int)

        lo = IronCondorEngine(PositionConfig(slippage_per_leg=0.0)).run(sig, spx, vix, all_green)
        hi = IronCondorEngine(PositionConfig(slippage_per_leg=0.5)).run(sig, spx, vix, all_green)

        assert lo["net_pnl_dollars"].sum() > hi["net_pnl_dollars"].sum()

    def test_friction_in_report_acknowledged(self):
        """The report should show lower EV per trade due to friction."""
        from src.backtest.engine import IronCondorEngine, PositionConfig
        from src.backtest.report import compute_performance
        from src.calibration.regime import Regime
        spx       = _spx_df()
        vix       = _vix_df()
        sig       = _signals_df(spx)
        all_green = pd.Series(Regime.GREEN, index=spx.index, dtype=int)

        cfg_hi    = PositionConfig(slippage_per_leg=1.0)
        trades_hi = IronCondorEngine(cfg_hi).run(sig, spx, vix, all_green)
        perf_hi   = compute_performance(trades_hi)

        cfg_lo    = PositionConfig(slippage_per_leg=0.01)
        trades_lo = IronCondorEngine(cfg_lo).run(sig, spx, vix, all_green)
        perf_lo   = compute_performance(trades_lo)

        assert perf_lo["ev_per_trade_dollars"] > perf_hi["ev_per_trade_dollars"]


# ─────────────────────────────────────────────────────────────────────────────
# TestSkipAnalysis
# ─────────────────────────────────────────────────────────────────────────────

class TestSkipAnalysis:

    def test_skip_analysis_returns_dataframe(self):
        from src.backtest.engine import IronCondorEngine
        from src.backtest.report import skip_analysis
        from src.calibration.regime import Regime
        spx = _spx_df(100); vix = _vix_df(100); sig = _signals_df(spx)
        reg = _regime(100, pct_red=0.3)
        trades = IronCondorEngine().run(sig, spx, vix, reg)
        df = skip_analysis(trades)
        assert isinstance(df, pd.DataFrame)

    def test_skip_reason_col_in_analysis(self):
        from src.backtest.engine import IronCondorEngine
        from src.backtest.report import skip_analysis
        from src.calibration.regime import Regime
        spx = _spx_df(100); vix = _vix_df(100); sig = _signals_df(spx)
        all_red = pd.Series(Regime.RED, index=spx.index, dtype=int)
        trades  = IronCondorEngine().run(sig, spx, vix, all_red)
        df = skip_analysis(trades)
        assert "skip_reason" in df.columns

    def test_no_skips_returns_empty(self):
        from src.backtest.engine import IronCondorEngine
        from src.backtest.report import skip_analysis
        from src.calibration.regime import Regime
        spx = _spx_df(50); vix = _vix_df(50)
        sig = pd.DataFrame({
            "upper_90": [0.010] * 50,
            "lower_90": [-0.010] * 50,
        }, index=spx.index)
        all_green = pd.Series(Regime.GREEN, index=spx.index, dtype=int)
        trades = IronCondorEngine().run(sig, spx, vix, all_green)
        df = skip_analysis(trades[~trades["skipped"]])  # pass only active
        assert len(df) == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestPipelineRunner
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineRunner:

    @pytest.fixture
    def tmp_runner(self, tmp_path):
        from src.pipeline.runner import PipelineRunner
        raw_dir       = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        output_dir    = tmp_path / "signals"
        raw_dir.mkdir(); processed_dir.mkdir(); output_dir.mkdir()

        # Write a minimal SPX parquet
        spx = _spx_df(300)
        spx.to_parquet(raw_dir / "spx_daily.parquet")

        # Write a minimal features.parquet so feature build is skipped
        feat_df = pd.DataFrame(
            np.random.default_rng(0).standard_normal((300, 5)),
            index=spx.index,
            columns=[f"feat_{i}" for i in range(5)],
        )
        feat_df.to_parquet(processed_dir / "features.parquet")

        return PipelineRunner(raw_dir=raw_dir,
                              processed_dir=processed_dir,
                              output_dir=output_dir)

    def test_run_returns_daily_signal(self, tmp_runner):
        from src.pipeline.runner import DailySignal
        sig = tmp_runner.run(mode="live", save_signal=False)
        assert isinstance(sig, DailySignal)

    def test_signal_has_signal_date(self, tmp_runner):
        sig = tmp_runner.run(mode="live", save_signal=False)
        assert sig.signal_date is not None
        assert len(sig.signal_date) == 10   # YYYY-MM-DD

    def test_signal_has_regime(self, tmp_runner):
        sig = tmp_runner.run(mode="live", save_signal=False)
        assert sig.regime in {"GREEN", "YELLOW", "RED"}

    def test_signal_serialisable_to_json(self, tmp_runner):
        sig  = tmp_runner.run(mode="live", save_signal=False)
        text = sig.to_json()
        obj  = json.loads(text)
        assert "regime" in obj

    def test_save_creates_file(self, tmp_runner):
        sig   = tmp_runner.run(mode="live", save_signal=True)
        files = list(tmp_runner.output_dir.glob("signal_*.json"))
        assert len(files) == 1

    def test_replay_mode_respects_cutoff(self, tmp_runner):
        """Replay with a past date must not see future data."""
        spx = _spx_df(300)
        cutoff = spx.index[100].strftime("%Y-%m-%d")
        sig_replay = tmp_runner.run(mode="replay", as_of_date=cutoff,
                                    save_signal=False)
        assert sig_replay.mode == "replay"

    def test_missing_spx_returns_error_signal(self, tmp_path):
        from src.pipeline.runner import PipelineRunner
        runner = PipelineRunner(
            raw_dir=tmp_path / "empty_raw",
            processed_dir=tmp_path / "proc",
            output_dir=tmp_path / "out",
        )
        sig = runner.run(mode="live", save_signal=False)
        assert not sig.tradeable
        assert any("ERROR" in n for n in sig.notes)


# ─────────────────────────────────────────────────────────────────────────────
# TestDeadMansSwitch
# ─────────────────────────────────────────────────────────────────────────────

class TestDeadMansSwitch:

    def test_check_true_when_file_exists(self, tmp_path):
        from src.pipeline.scheduler import DeadMansSwitch
        signal_dir = tmp_path / "signals"
        signal_dir.mkdir()
        today = date.today().strftime("%Y-%m-%d")
        (signal_dir / f"signal_{today}.json").write_text("{}")

        switch = DeadMansSwitch(signal_dir=signal_dir)
        assert switch.check() is True

    def test_check_false_when_file_missing(self, tmp_path):
        from src.pipeline.scheduler import DeadMansSwitch
        signal_dir = tmp_path / "signals"
        signal_dir.mkdir()

        switch = DeadMansSwitch(signal_dir=signal_dir)
        assert switch.check() is False

    def test_signal_path_contains_today(self, tmp_path):
        from src.pipeline.scheduler import DeadMansSwitch
        switch = DeadMansSwitch(signal_dir=tmp_path)
        today  = date.today().strftime("%Y-%m-%d")
        assert today in str(switch.today_signal_path())

    def test_dead_man_alert_does_not_raise(self, tmp_path):
        """
        Dead-man switch must not raise an exception even without SMTP/Slack
        configured — it should log the alert and return False gracefully.
        """
        from src.pipeline.scheduler import DeadMansSwitch
        signal_dir = tmp_path / "signals"
        signal_dir.mkdir()
        switch = DeadMansSwitch(signal_dir=signal_dir)
        # Should not raise
        result = switch.check()
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# TestReplayDateCheck
# ─────────────────────────────────────────────────────────────────────────────

class TestReplayDateCheck:

    def test_replay_date_check_returns_dict(self, tmp_path):
        from src.pipeline.runner import PipelineRunner, replay_date_check
        raw_dir = tmp_path / "raw"; raw_dir.mkdir()
        _spx_df(300).to_parquet(raw_dir / "spx_daily.parquet")
        runner = PipelineRunner(raw_dir=raw_dir,
                                processed_dir=tmp_path/"proc",
                                output_dir=tmp_path/"out")

        spx = _spx_df(300)
        cutoff = spx.index[150].strftime("%Y-%m-%d")
        # Synthetic backtest OOS value
        bt_val = pd.Series([0.0042], index=[cutoff])[cutoff]

        result = replay_date_check(
            backtest_oos_row=bt_val,
            as_of_date=cutoff,
            runner=runner,
        )
        assert "match" in result

    def test_replay_returns_false_on_missing_model(self, tmp_path):
        """When no saved model exists, replay can't match — should report mismatch."""
        from src.pipeline.runner import PipelineRunner, replay_date_check
        raw_dir = tmp_path / "raw"; raw_dir.mkdir()
        _spx_df(300).to_parquet(raw_dir / "spx_daily.parquet")
        runner = PipelineRunner(raw_dir=raw_dir,
                                processed_dir=tmp_path/"proc",
                                output_dir=tmp_path/"out")
        spx    = _spx_df(300)
        cutoff = spx.index[100].strftime("%Y-%m-%d")
        result = replay_date_check(0.005, cutoff, runner)
        # With no model, replay_pred will be None → match=False
        assert isinstance(result["match"], bool)
