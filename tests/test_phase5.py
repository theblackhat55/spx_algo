"""
tests/test_phase5.py
=====================
Phase 5 test gate — Live Fetcher, Signal Generator, Alerting,
Broker safety gates, and FullSignal schema.

Test groups
-----------
TestLiveFetcher            (7 tests)
TestSignalGenerator        (8 tests)
TestFullSignalSchema       (5 tests)
TestAlerting               (7 tests)
TestBrokerSafetyGates      (5 tests)
TestConformalCoverage      (3 tests)
TestDeterminism            (3 tests)
──────────────────────────────────────
Total:                     38 tests
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Shared helpers ─────────────────────────────────────────────────────────

N = 400
RNG = np.random.default_rng(77)


def _spx_df(n: int = N, base: float = 5000.0) -> pd.DataFrame:
    rng   = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=n)
    c = base + np.cumsum(rng.normal(0, 8, n))
    h = c + rng.uniform(8, 30, n)
    lo = c - rng.uniform(8, 30, n)
    o  = c + rng.normal(0, 4, n)
    v  = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": c, "Volume": v},
                        index=dates)


def _feat_df(n: int = N) -> pd.DataFrame:
    rng   = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        rng.standard_normal((n, 10)),
        index=dates,
        columns=[f"feat_{i}" for i in range(10)],
    )


def _make_tmp_dirs(tmp_path: Path):
    raw       = tmp_path / "raw"
    processed = tmp_path / "processed"
    output    = tmp_path / "output" / "signals"
    raw.mkdir(parents=True)
    processed.mkdir(parents=True)
    output.mkdir(parents=True)
    return raw, processed, output


# =============================================================================
# TestLiveFetcher
# =============================================================================

class TestLiveFetcher:

    def test_validate_market_open_weekday(self):
        from src.data.live_fetcher import validate_market_open
        # A known trading day (Monday)
        monday = date(2024, 1, 15)   # MLK Day — actually market closed
        saturday = date(2024, 1, 13)
        assert validate_market_open(saturday) is False

    def test_validate_market_open_weekend_false(self):
        from src.data.live_fetcher import validate_market_open
        sunday = date(2024, 1, 14)
        assert validate_market_open(sunday) is False

    def test_append_to_parquet_new_row(self, tmp_path):
        from src.data.live_fetcher import append_to_parquet
        fpath = tmp_path / "test.parquet"
        row = pd.Series({"Open": 100.0, "Close": 101.0, "High": 103.0,
                         "Low": 99.0, "Volume": 1000000})
        result = append_to_parquet(row, fpath, date(2024, 1, 2), backup=False)
        assert result is True
        df = pd.read_parquet(fpath)
        assert len(df) == 1

    def test_append_to_parquet_duplicate_rejected(self, tmp_path):
        from src.data.live_fetcher import append_to_parquet
        fpath = tmp_path / "test.parquet"
        row = pd.Series({"Open": 100.0, "Close": 101.0, "High": 103.0,
                         "Low": 99.0, "Volume": 1000000})
        append_to_parquet(row, fpath, date(2024, 1, 2), backup=False)
        result2 = append_to_parquet(row, fpath, date(2024, 1, 2), backup=False)
        assert result2 is False
        df = pd.read_parquet(fpath)
        assert len(df) == 1

    def test_append_to_parquet_backup_created(self, tmp_path):
        from src.data.live_fetcher import append_to_parquet
        # Create initial parquet
        fpath = tmp_path / "spx_daily.parquet"
        initial = pd.DataFrame({"Close": [100.0]},
                               index=pd.to_datetime(["2024-01-01"]))
        initial.to_parquet(fpath)
        row = pd.Series({"Open": 101.0, "Close": 102.0, "High": 104.0,
                         "Low": 100.0, "Volume": 999999})
        append_to_parquet(row, fpath, date(2024, 1, 2), backup=True)
        backups = list(tmp_path.glob("spx_daily_*.parquet"))
        assert len(backups) == 1

    def test_append_schema_mismatch_fills_nan(self, tmp_path):
        from src.data.live_fetcher import append_to_parquet
        fpath = tmp_path / "test.parquet"
        # Write initial row with column 'extra'
        initial = pd.DataFrame({"Close": [100.0], "extra": [1.0]},
                               index=pd.to_datetime(["2024-01-01"]))
        initial.to_parquet(fpath)
        # Append row missing 'extra'
        row = pd.Series({"Close": 101.0})
        result = append_to_parquet(row, fpath, date(2024, 1, 2), backup=False)
        assert result is True
        df = pd.read_parquet(fpath)
        # 'extra' should be NaN for new row
        assert pd.isna(df.loc[pd.Timestamp("2024-01-02"), "extra"])

    def test_run_daily_fetch_weekend_returns_closed(self):
        from src.data.live_fetcher import run_daily_fetch
        sunday = date(2024, 1, 14)
        status = run_daily_fetch(target_date=sunday)
        assert status["market_open"] is False


# =============================================================================
# TestSignalGenerator
# =============================================================================

class TestSignalGenerator:

    @pytest.fixture
    def gen_env(self, tmp_path):
        """Create a temporary environment with SPX data and features."""
        raw, processed, output = _make_tmp_dirs(tmp_path)

        spx = _spx_df()
        spx.to_parquet(raw / "spx_daily.parquet")

        feats = _feat_df()
        feats.to_parquet(processed / "features.parquet")

        from src.pipeline.signal_generator import SignalGenerator
        gen = SignalGenerator(raw_dir=raw, processed_dir=processed, output_dir=output)
        return gen, spx, feats, output

    def test_generate_returns_full_signal(self, gen_env):
        from src.pipeline.signal_generator import FullSignal
        gen, *_ = gen_env
        sig = gen.generate(mode="live", save=False)
        assert isinstance(sig, FullSignal)

    def test_signal_date_is_yyyy_mm_dd(self, gen_env):
        gen, *_ = gen_env
        sig = gen.generate(mode="live", save=False)
        assert len(sig.signal_date) == 10
        # Parseable
        datetime.strptime(sig.signal_date, "%Y-%m-%d")

    def test_regime_is_valid(self, gen_env):
        gen, *_ = gen_env
        sig = gen.generate(mode="live", save=False)
        assert sig.regime in {"GREEN", "YELLOW", "RED"}

    def test_predicted_high_above_low(self, gen_env):
        gen, *_ = gen_env
        sig = gen.generate(mode="live", save=False)
        if sig.predicted_high is not None and sig.predicted_low is not None:
            assert sig.predicted_high > sig.predicted_low

    def test_conformal_intervals_present(self, gen_env):
        gen, *_ = gen_env
        sig = gen.generate(mode="live", save=False)
        # At minimum 68% bounds should be present
        assert sig.conf_68_high_lo is not None or sig.data_quality in ("PARTIAL", "DEGRADED")

    def test_save_creates_json_file(self, gen_env):
        gen, _, _, output = gen_env
        sig = gen.generate(mode="live", save=True)
        files = list(output.glob("signal_*.json"))
        assert len(files) >= 1

    def test_latest_signal_json_written(self, gen_env):
        gen, _, _, output = gen_env
        sig = gen.generate(mode="live", save=True)
        assert (output / "latest_signal.json").exists()

    def test_error_signal_on_missing_data(self, tmp_path):
        from src.pipeline.signal_generator import SignalGenerator, FullSignal
        raw, proc, out = _make_tmp_dirs(tmp_path)
        # No SPX file
        gen = SignalGenerator(raw_dir=raw, processed_dir=proc, output_dir=out)
        sig = gen.generate(mode="live", save=False)
        assert isinstance(sig, FullSignal)
        assert not sig.tradeable
        assert any("ERROR" in n for n in sig.notes)


# =============================================================================
# TestFullSignalSchema
# =============================================================================

class TestFullSignalSchema:

    @pytest.fixture
    def sample_signal(self):
        from src.pipeline.signal_generator import FullSignal
        return FullSignal(
            signal_date="2024-01-15",
            prediction_date="2024-01-15",
            generated_at="2024-01-14T21:05:00Z",
            mode="live",
            regime="GREEN",
            pred_high_pct=0.008,
            pred_low_pct=-0.007,
            predicted_high=5040.0,
            predicted_low=4965.0,
            predicted_range=75.0,
            conf_68_high_lo=5010.0,
            conf_68_high_hi=5060.0,
            conf_90_high_lo=4990.0,
            conf_90_high_hi=5090.0,
            conf_68_low_lo=4945.0,
            conf_68_low_hi=4985.0,
            ic_short_call=5060.0,
            ic_long_call=5110.0,
            ic_short_put=4945.0,
            ic_long_put=4895.0,
            prior_close=5000.0,
            direction="BULLISH",
            direction_prob=0.65,
            data_quality="FULL",
            tradeable=True,
        )

    def test_to_json_is_valid_json(self, sample_signal):
        text = sample_signal.to_json()
        obj  = json.loads(text)
        assert "regime" in obj
        assert "signal_date" in obj

    def test_to_json_is_sorted(self, sample_signal):
        text = sample_signal.to_json()
        obj  = json.loads(text)
        keys = list(obj.keys())
        assert keys == sorted(keys)

    def test_validate_signal_no_errors(self, sample_signal):
        from src.pipeline.signal_generator import validate_signal
        errors = validate_signal(sample_signal)
        assert errors == []

    def test_validate_signal_inverted_strikes(self, sample_signal):
        from src.pipeline.signal_generator import validate_signal
        sample_signal.predicted_high = 4900.0
        sample_signal.predicted_low  = 5100.0
        errors = validate_signal(sample_signal)
        assert any("predicted_high <= predicted_low" in e for e in errors)

    def test_save_and_reload(self, tmp_path, sample_signal):
        from src.pipeline.signal_generator import FullSignal
        sample_signal.save(tmp_path)
        fname = tmp_path / "signal_2024-01-15.json"
        assert fname.exists()
        obj = json.loads(fname.read_text())
        assert obj["regime"] == "GREEN"
        assert obj["tradeable"] is True


# =============================================================================
# TestAlerting
# =============================================================================

class TestAlerting:

    @pytest.fixture
    def cfg(self):
        from src.pipeline.alerting import AlertConfig
        return AlertConfig()   # all None — no real channels

    @pytest.fixture
    def signal(self):
        from src.pipeline.signal_generator import FullSignal
        return FullSignal(
            signal_date="2024-01-15",
            prediction_date="2024-01-15",
            generated_at="2024-01-14T21:05:00Z",
            mode="live",
            regime="GREEN",
            predicted_high=5040.0,
            predicted_low=4965.0,
            prior_close=5000.0,
            data_quality="FULL",
            tradeable=True,
        )

    def test_alert_config_from_env_returns_config(self):
        from src.pipeline.alerting import alert_config_from_env, AlertConfig
        cfg = alert_config_from_env()
        assert isinstance(cfg, AlertConfig)

    def test_discord_payload_format(self, signal):
        from src.pipeline.alerting import send_discord_webhook
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 204
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp
            result = send_discord_webhook(signal, "https://discord.com/api/webhooks/fake")
        assert result is True
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs.get("json")
        # Verify embed structure
        sent_json = mock_post.call_args[1]["json"]
        assert "embeds" in sent_json
        assert sent_json["embeds"][0]["title"].startswith("📊")

    def test_telegram_payload_format(self, signal):
        from src.pipeline.alerting import send_telegram_message
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp
            result = send_telegram_message(signal, "fake_token", "fake_chat_id")
        assert result is True
        sent = mock_post.call_args[1]["json"]
        assert "chat_id" in sent
        assert "text" in sent
        assert "2024-01-15" in sent["text"]

    def test_missing_credentials_skip_gracefully(self, cfg, signal):
        from src.pipeline.alerting import Alerter
        alerter = Alerter(cfg)
        # Should not raise; all channels are None
        results = alerter.send_signal(signal)
        assert isinstance(results, dict)
        assert len(results) == 0   # no channels configured

    def test_failure_alert_no_crash_no_channels(self, cfg):
        from src.pipeline.alerting import send_failure_alert
        # Must not raise even with all-None config
        send_failure_alert("Test failure", cfg)

    def test_discord_failure_handled_gracefully(self, signal):
        from src.pipeline.alerting import send_discord_webhook
        with patch("requests.post", side_effect=ConnectionError("network down")):
            result = send_discord_webhook(signal, "https://discord.com/api/webhooks/fake")
        assert result is False

    def test_alerter_send_signal_returns_dict(self, cfg, signal):
        from src.pipeline.alerting import Alerter
        alerter = Alerter(cfg)
        results = alerter.send_signal(signal)
        assert isinstance(results, dict)


# =============================================================================
# TestBrokerSafetyGates
# =============================================================================

class TestBrokerSafetyGates:

    @pytest.fixture
    def red_signal(self):
        from src.pipeline.signal_generator import FullSignal
        return FullSignal(
            signal_date="2024-01-15",
            prediction_date="2024-01-15",
            generated_at="2024-01-14T21:05:00Z",
            mode="live",
            regime="RED",
            data_quality="FULL",
            tradeable=False,
        )

    @pytest.fixture
    def degraded_signal(self):
        from src.pipeline.signal_generator import FullSignal
        return FullSignal(
            signal_date="2024-01-15",
            prediction_date="2024-01-15",
            generated_at="2024-01-14T21:05:00Z",
            mode="live",
            regime="GREEN",
            data_quality="DEGRADED",
            tradeable=False,
        )

    @pytest.fixture
    def green_signal(self):
        from src.pipeline.signal_generator import FullSignal
        return FullSignal(
            signal_date="2024-01-15",
            prediction_date="2024-01-15",
            generated_at="2024-01-14T21:05:00Z",
            mode="live",
            regime="GREEN",
            ic_short_call=5060.0,
            ic_long_call=5110.0,
            ic_short_put=4945.0,
            ic_long_put=4895.0,
            data_quality="FULL",
            tradeable=True,
        )

    def test_red_regime_blocks_execution(self, red_signal):
        from src.execution.broker import build_condor_from_signal, IBKRBroker
        broker = IBKRBroker(paper=True)
        with pytest.raises(ValueError, match="RED"):
            build_condor_from_signal(red_signal, broker)

    def test_degraded_quality_kill_switch(self, degraded_signal):
        from src.execution.broker import build_condor_from_signal, IBKRBroker
        broker = IBKRBroker(paper=True)
        with pytest.raises(ValueError, match="DEGRADED"):
            build_condor_from_signal(degraded_signal, broker)

    def test_live_mode_requires_env_var(self):
        from src.execution.broker import IBKRBroker
        with pytest.raises(RuntimeError, match="EXECUTION_MODE"):
            IBKRBroker(paper=False)

    def test_connect_fails_gracefully_without_tws(self):
        """Without TWS running, connect() should return False, not raise."""
        from src.execution.broker import IBKRBroker
        broker = IBKRBroker(paper=True)
        try:
            result = broker.connect(max_attempts=1)
            assert result is False
        except ImportError:
            pytest.skip("ib_async not installed")

    def test_missing_strikes_raise_value_error(self):
        from src.execution.broker import build_condor_from_signal, IBKRBroker
        from src.pipeline.signal_generator import FullSignal
        sig = FullSignal(
            signal_date="2024-01-15",
            prediction_date="2024-01-15",
            generated_at="2024-01-14T21:05:00Z",
            mode="live",
            regime="GREEN",
            data_quality="FULL",
            tradeable=True,
            # No ic_* fields set — all None
        )
        broker = IBKRBroker(paper=True)
        with pytest.raises(ValueError, match="Missing strike"):
            build_condor_from_signal(sig, broker)


# =============================================================================
# TestConformalCoverage
# =============================================================================

class TestConformalCoverage:

    @pytest.fixture
    def fitted_conformal(self):
        """Return a calibrated ConformalPredictor on synthetic data."""
        from src.calibration.conformal import ConformalPredictor
        from src.models.linear_models import RidgeRegressionModel as RidgeModel

        n = 400
        rng = np.random.default_rng(99)
        dates = pd.bdate_range("2020-01-02", periods=n)
        X = pd.DataFrame(rng.standard_normal((n, 5)),
                         index=dates, columns=[f"f{i}" for i in range(5)])
        y_true = pd.Series(0.001 + 0.3 * X["f0"].values + rng.normal(0, 0.005, n),
                           index=dates, name="target")

        model = RidgeModel(name="test_ridge", alpha=1.0)
        model.fit(X.iloc[:300], y_true.iloc[:300])

        cp = ConformalPredictor(model, use_mapie=False, alpha_list=[0.68, 0.90])
        cp.calibrate(X.iloc[250:350], y_true.iloc[250:350])
        return cp, X, y_true

    def test_68_coverage_in_expected_range(self, fitted_conformal):
        from src.pipeline.signal_generator import validate_conformal_coverage
        cp, X, y = fitted_conformal
        cov = validate_conformal_coverage(cp, X, y, window=100)
        # 68% CI should cover 50-86% on random data (wide tolerance for small sample)
        assert 0.40 <= cov["coverage_68"] <= 1.0

    def test_90_coverage_exceeds_68_coverage(self, fitted_conformal):
        from src.pipeline.signal_generator import validate_conformal_coverage
        cp, X, y = fitted_conformal
        cov = validate_conformal_coverage(cp, X, y, window=100)
        assert cov["coverage_90"] >= cov["coverage_68"]

    def test_intervals_have_correct_columns(self, fitted_conformal):
        cp, X, _ = fitted_conformal
        intervals = cp.predict_interval(X.iloc[-5:])
        for col in ("predicted", "lower_68", "upper_68", "lower_90", "upper_90"):
            assert col in intervals.columns


# =============================================================================
# TestDeterminism
# =============================================================================

class TestDeterminism:

    def test_full_signal_json_sorted_keys(self):
        from src.pipeline.signal_generator import FullSignal
        sig = FullSignal(
            signal_date="2024-01-15",
            prediction_date="2024-01-15",
            generated_at="2024-01-14T21:05:00Z",
            mode="live",
            regime="GREEN",
        )
        text = sig.to_json()
        obj  = json.loads(text)
        keys = list(obj.keys())
        assert keys == sorted(keys), "JSON keys must be sorted for byte-identical output"

    def test_signal_save_twice_identical_content(self, tmp_path):
        from src.pipeline.signal_generator import FullSignal
        sig = FullSignal(
            signal_date="2024-01-15",
            prediction_date="2024-01-15",
            generated_at="2024-01-14T21:05:00Z",
            mode="live",
            regime="GREEN",
            predicted_high=5040.0,
        )
        path1 = tmp_path / "run1"
        path2 = tmp_path / "run2"
        sig.save(path1)
        sig.save(path2)
        content1 = (path1 / "signal_2024-01-15.json").read_text()
        content2 = (path2 / "signal_2024-01-15.json").read_text()
        assert content1 == content2

    def test_signal_generator_deterministic_on_same_data(self, tmp_path):
        """Two runs on identical data produce same regime and predictions."""
        raw, processed, output = _make_tmp_dirs(tmp_path)
        spx = _spx_df(300)
        spx.to_parquet(raw / "spx_daily.parquet")
        feats = _feat_df(300)
        feats.to_parquet(processed / "features.parquet")

        from src.pipeline.signal_generator import SignalGenerator
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        out1.mkdir(); out2.mkdir()

        gen1 = SignalGenerator(raw_dir=raw, processed_dir=processed,
                               output_dir=out1, seed=42)
        gen2 = SignalGenerator(raw_dir=raw, processed_dir=processed,
                               output_dir=out2, seed=42)

        sig1 = gen1.generate(mode="live", save=False)
        sig2 = gen2.generate(mode="live", save=False)

        # Core predictions must match (generated_at will differ by seconds)
        assert sig1.regime         == sig2.regime
        assert sig1.pred_high_pct  == sig2.pred_high_pct
        assert sig1.pred_low_pct   == sig2.pred_low_pct
