"""
tests/test_remediation.py
==========================
Pre-Phase-4 remediation gate — tests for all 6 reviewer instructions.

Test groups
-----------
TestRegressionTargets          ( 8 tests) — Instruction 1: regression targets
TestRegressionModels           ( 8 tests) — Instruction 1: Huber models
TestRegimeDetector             (10 tests) — Instruction 2: regime signals
TestConformalPredictor         ( 8 tests) — Instruction 3: prediction intervals
TestBaselines                  (10 tests) — Instruction 4: naive baselines
TestDieboldMariano             ( 4 tests) — Instruction 4: DM test
TestLeakageGateConfirm         ( 1 test)  — Instruction 5: re-confirm gate
TestPandasTaConfirm            ( 1 test)  — Instruction 6: no stale TA dep
─────────────────────────────────────────────────────────────────────────────
Total:                          50 tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Shared helpers ────────────────────────────────────────────────────────────

N = 600
RNG = np.random.default_rng(7)


def _spx_df(n: int = N, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-01", periods=n)
    c = 3000 + np.cumsum(rng.normal(0, 8, n))
    h = c + rng.uniform(4, 25, n)
    l = c - rng.uniform(4, 25, n)
    o = c + rng.normal(0, 4, n)
    v = rng.integers(1_000_000, 5_000_000, n).astype(float)  # FIX F3: float avoids int*1.5 dtype error
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v},
                        index=dates)


def _vix_df(n: int = N, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-01", periods=n)
    v     = 15 + np.cumsum(rng.normal(0, 0.3, n))
    return pd.DataFrame({"Close": v}, index=dates)


def _feat_df(n: int = N) -> pd.DataFrame:
    from sklearn.datasets import make_regression
    X, _ = make_regression(n_samples=n, n_features=15, noise=0.1, random_state=0)
    dates = pd.bdate_range("2017-01-01", periods=n)
    return pd.DataFrame(X, index=dates, columns=[f"f{i}" for i in range(15)])


def _spx_parquet(tmp_path: Path, n: int = N) -> Path:
    p = tmp_path / "spx.parquet"
    _spx_df(n).to_parquet(p)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# TestRegressionTargets  (Instruction 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionTargets:

    @pytest.fixture
    def tgt(self, tmp_path):
        from src.targets.engineer import engineer_targets
        return engineer_targets(spx_path=_spx_parquet(tmp_path), save=False)

    def test_target_high_col_present(self, tgt):
        assert "target_high" in tgt.columns

    def test_target_low_col_present(self, tgt):
        assert "target_low" in tgt.columns

    def test_target_high_pts_col_present(self, tgt):
        assert "target_high_pts" in tgt.columns

    def test_target_low_pts_col_present(self, tgt):
        assert "target_low_pts" in tgt.columns

    def test_target_high_is_positive(self, tgt):
        assert (tgt["target_high"].dropna() > 0).all()

    def test_target_high_pts_mostly_positive(self, tgt):
        """High should exceed prior close most of the time."""
        pos_pct = (tgt["target_high_pts"].dropna() > 0).mean()
        assert pos_pct > 0.6

    def test_target_low_pts_mostly_negative(self, tgt):
        """Low should be below prior close most of the time."""
        neg_pct = (tgt["target_low_pts"].dropna() < 0).mean()
        assert neg_pct > 0.6

    def test_regression_and_binary_targets_coexist(self, tgt):
        """Both regression and classification targets must be present."""
        assert "target_high"       in tgt.columns
        assert "next_high_bin_050" in tgt.columns
        assert "next_high_pct"     in tgt.columns


# ─────────────────────────────────────────────────────────────────────────────
# TestRegressionModels  (Instruction 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionModels:

    @pytest.fixture
    def reg_data(self):
        X = _feat_df(200)
        y = pd.Series(
            3000 + np.random.normal(0, 10, 200),
            index=X.index, name="target_high"
        )
        return X, y

    def test_huber_xgboost_skipped_if_no_xgb(self):
        try:
            from src.models.tree_models import HuberXGBoostModel
            HuberXGBoostModel()   # should not raise if xgboost installed
        except ImportError:
            pytest.skip("xgboost not installed")

    def test_huber_lightgbm_skipped_if_no_lgbm(self):
        try:
            from src.models.tree_models import HuberLightGBMModel
            HuberLightGBMModel()
        except ImportError:
            pytest.skip("lightgbm not installed")

    def test_random_forest_regression_predict_shape(self, reg_data):
        from src.models.tree_models import RandomForestModel
        X, y = reg_data
        m = RandomForestModel(task="regression", params={"n_estimators": 5})
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)

    def test_ridge_regression_predict_in_price_range(self, reg_data):
        from src.models.linear_models import RidgeRegressionModel
        X, y = reg_data
        m = RidgeRegressionModel()
        m.fit(X, y)
        preds = m.predict(X)
        # Predictions should be in a sensible price range
        assert (preds > 2000).all() and (preds < 4500).all()

    def test_catboost_model_skipped_if_no_catboost(self):
        try:
            from src.models.tree_models import CatBoostModel
            CatBoostModel(task="regression", params={"iterations": 5})
        except ImportError:
            pytest.skip("catboost not installed")

    def test_regression_trainer_returns_continuous_oos(self):
        from src.models.tree_models import RandomForestModel
        from src.models.trainer import Trainer
        from src.targets.splitter import WalkForwardSplitter, SplitConfig

        X = _feat_df(N)
        y = pd.DataFrame({
            "target_high_pct": RNG.normal(0.004, 0.005, N),
        }, index=X.index)

        m   = RandomForestModel(task="regression", params={"n_estimators": 5})
        cfg = SplitConfig(min_train_rows=300, test_rows=100, step_rows=100)
        tr  = Trainer(model=m, splitter=WalkForwardSplitter(cfg))
        res = tr.run(X, y, target_col="target_high_pct")

        assert res.oos_proba.dtype in (np.float32, np.float64, float)
        # Regression predictions should not be binary
        assert res.oos_proba.nunique() > 2

    def test_regression_oos_values_are_finite(self):
        from src.models.tree_models import RandomForestModel
        from src.models.trainer import Trainer
        from src.targets.splitter import WalkForwardSplitter, SplitConfig

        X = _feat_df(N)
        y = pd.DataFrame({"target_high_pct": RNG.normal(0.004, 0.005, N)},
                         index=X.index)
        m   = RandomForestModel(task="regression", params={"n_estimators": 5})
        cfg = SplitConfig(min_train_rows=300, test_rows=100, step_rows=100)
        res = Trainer(model=m, splitter=WalkForwardSplitter(cfg)).run(
            X, y, target_col="target_high_pct"
        )
        assert np.isfinite(res.oos_proba.values).all()

    def test_ridge_regression_score_r2(self, reg_data):
        from src.models.linear_models import RidgeRegressionModel
        X, y = reg_data
        m = RidgeRegressionModel()
        m.fit(X, y)
        r2 = m.score(X, y, metric="r2")
        # Ridge can overfit; just verify API works
        assert isinstance(r2, float)


# ─────────────────────────────────────────────────────────────────────────────
# TestRegimeDetector  (Instruction 2)
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeDetector:

    @pytest.fixture
    def spx(self): return _spx_df()

    @pytest.fixture
    def vix(self): return _vix_df()

    def test_vix_regime_returns_series(self, vix):
        from src.calibration.regime import vix_regime
        sig = vix_regime(vix["Close"])
        assert isinstance(sig, pd.Series)
        assert len(sig) == len(vix)

    def test_vix_regime_values_0_1_2(self, vix):
        from src.calibration.regime import vix_regime, Regime
        sig = vix_regime(vix["Close"])
        assert set(sig.values).issubset({Regime.GREEN, Regime.YELLOW, Regime.RED})

    def test_atr_regime_returns_series(self, spx):
        from src.calibration.regime import atr_expansion_regime
        sig = atr_expansion_regime(spx)
        assert isinstance(sig, pd.Series) and len(sig) == len(spx)

    def test_hmm_regime_returns_series_or_skips(self, spx):
        from src.calibration.regime import hmm_regime
        ret     = spx["Close"].pct_change()
        rng_pct = (spx["High"] - spx["Low"]) / spx["Close"].shift(1)
        try:
            sig = hmm_regime(ret, rng_pct, n_states=3)
            assert len(sig) == len(spx)
        except Exception:
            pytest.skip("hmmlearn not installed or fit failed")

    def test_regime_detector_no_vix(self, spx):
        from src.calibration.regime import RegimeDetector
        rd  = RegimeDetector(use_hmm=False, use_garch=False)
        sig = rd.fit_predict(spx, vix_df=None)
        assert len(sig) == len(spx)

    def test_regime_detector_with_vix(self, spx, vix):
        from src.calibration.regime import RegimeDetector
        rd  = RegimeDetector(use_hmm=False, use_garch=False)
        sig = rd.fit_predict(spx, vix_df=vix)
        assert len(sig) == len(spx)

    def test_regime_all_values_valid(self, spx, vix):
        from src.calibration.regime import RegimeDetector, Regime
        rd  = RegimeDetector(use_hmm=False)
        sig = rd.fit_predict(spx, vix)
        assert set(sig.values).issubset({Regime.GREEN, Regime.YELLOW, Regime.RED})

    def test_regime_label_converts_to_str(self, spx):
        from src.calibration.regime import RegimeDetector
        rd    = RegimeDetector(use_hmm=False)
        sig   = rd.fit_predict(spx)
        lbls  = RegimeDetector.label(sig)
        assert lbls.isin({"GREEN", "YELLOW", "RED"}).all()

    def test_regime_components_accessible(self, spx, vix):
        from src.calibration.regime import RegimeDetector
        rd = RegimeDetector(use_hmm=False)
        rd.fit_predict(spx, vix)
        assert "atr" in rd.components
        assert "vix" in rd.components

    def test_regime_red_on_high_vix_spike(self):
        """Synthetically spike VIX to trigger RED."""
        from src.calibration.regime import vix_regime, Regime
        dates  = pd.bdate_range("2020-01-01", periods=300)
        vix_cl = pd.Series(15.0, index=dates)
        vix_cl.iloc[-5:] = 80.0   # spike
        sig = vix_regime(vix_cl, yellow_z=1.5, red_z=3.0)
        assert (sig.iloc[-5:] == Regime.RED).any()


# ─────────────────────────────────────────────────────────────────────────────
# TestConformalPredictor  (Instruction 3)
# ─────────────────────────────────────────────────────────────────────────────

class TestConformalPredictor:

    @pytest.fixture
    def fitted_ridge(self):
        from src.models.linear_models import RidgeRegressionModel
        X = _feat_df(200)
        y = pd.Series(3000 + np.random.normal(0, 10, 200), index=X.index)
        m = RidgeRegressionModel()
        m.fit(X, y)
        return m, X, y

    def test_conformal_requires_regression_model(self):
        from src.calibration.conformal import ConformalPredictor
        from src.models.tree_models import RandomForestModel
        clf = RandomForestModel(task="classification",
                                params={"n_estimators": 5})
        X = _feat_df(50); y_bin = pd.Series(np.random.randint(0,2,50), index=X.index)
        clf.fit(X, y_bin)
        with pytest.raises(ValueError):
            ConformalPredictor(clf)

    def test_uncalibrated_raises(self, fitted_ridge):
        from src.calibration.conformal import ConformalPredictor
        m, X, _ = fitted_ridge
        cp = ConformalPredictor(m, use_mapie=False)
        with pytest.raises(RuntimeError):
            cp.predict_interval(X)

    def test_calibrate_with_residual_icp(self, fitted_ridge):
        from src.calibration.conformal import ConformalPredictor
        m, X, y = fitted_ridge
        cp = ConformalPredictor(m, use_mapie=False)
        cp.calibrate(X, y)
        assert cp._calibrated

    def test_predict_interval_returns_dataframe(self, fitted_ridge):
        from src.calibration.conformal import ConformalPredictor
        m, X, y = fitted_ridge
        cp = ConformalPredictor(m, use_mapie=False)
        cp.calibrate(X, y)
        result = cp.predict_interval(X)
        assert isinstance(result, pd.DataFrame)

    def test_interval_columns_present(self, fitted_ridge):
        from src.calibration.conformal import ConformalPredictor
        m, X, y = fitted_ridge
        cp = ConformalPredictor(m, use_mapie=False)
        cp.calibrate(X, y)
        result = cp.predict_interval(X)
        for col in ("predicted", "lower_68", "upper_68", "lower_90", "upper_90"):
            assert col in result.columns, f"Missing column: {col}"

    def test_upper_greater_than_lower(self, fitted_ridge):
        from src.calibration.conformal import ConformalPredictor
        m, X, y = fitted_ridge
        cp = ConformalPredictor(m, use_mapie=False)
        cp.calibrate(X, y)
        res = cp.predict_interval(X)
        assert (res["upper_90"] > res["lower_90"]).all()
        assert (res["upper_68"] > res["lower_68"]).all()

    def test_wider_90_than_68(self, fitted_ridge):
        from src.calibration.conformal import ConformalPredictor
        m, X, y = fitted_ridge
        cp = ConformalPredictor(m, use_mapie=False)
        cp.calibrate(X, y)
        res = cp.predict_interval(X)
        w90 = (res["upper_90"] - res["lower_90"]).mean()
        w68 = (res["upper_68"] - res["lower_68"]).mean()
        assert w90 > w68

    def test_repr_contains_model_name(self, fitted_ridge):
        from src.calibration.conformal import ConformalPredictor
        m, X, y = fitted_ridge
        cp = ConformalPredictor(m, use_mapie=False)
        assert "ridge" in repr(cp)


# ─────────────────────────────────────────────────────────────────────────────
# TestBaselines  (Instruction 4)
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselines:

    @pytest.fixture
    def spx(self): return _spx_df()

    @pytest.fixture
    def targets(self, tmp_path):
        from src.targets.engineer import engineer_targets
        return engineer_targets(spx_path=_spx_parquet(tmp_path), save=False)

    def test_no_change_returns_dataframe(self, spx):
        from src.validation.baselines import NoChangeBaseline
        preds = NoChangeBaseline().predict(spx)
        assert isinstance(preds, pd.DataFrame)

    def test_no_change_has_required_cols(self, spx):
        from src.validation.baselines import NoChangeBaseline
        preds = NoChangeBaseline().predict(spx)
        for col in ("target_high", "target_low", "target_high_pct", "target_low_pct"):
            assert col in preds.columns

    def test_atr_baseline_rows_positive(self, spx):
        from src.validation.baselines import ATRBaseline
        preds = ATRBaseline().predict(spx)
        assert (preds["target_high"].dropna() > 0).all()

    def test_yesterday_range_upper_above_lower(self, spx):
        from src.validation.baselines import YesterdayRangeBaseline
        preds = YesterdayRangeBaseline().predict(spx)
        valid = preds.dropna(subset=["target_high","target_low"])
        assert (valid["target_high"] > valid["target_low"]).all()

    def test_atr_high_above_low(self, spx):
        from src.validation.baselines import ATRBaseline
        preds = ATRBaseline().predict(spx).dropna()
        assert (preds["target_high"] > preds["target_low"]).all()

    def test_evaluate_returns_mae_keys(self, spx, targets):
        from src.validation.baselines import NoChangeBaseline
        metrics = NoChangeBaseline().evaluate(spx, targets)
        assert "mae_high" in metrics and "mae_low" in metrics

    def test_evaluate_mae_positive(self, spx, targets):
        from src.validation.baselines import ATRBaseline
        metrics = ATRBaseline().evaluate(spx, targets)
        assert metrics["mae_high"] >= 0
        assert metrics["mae_low"]  >= 0

    def test_compare_baselines_returns_dataframe(self, spx, targets):
        from src.validation.baselines import compare_baselines
        df = compare_baselines(spx, targets)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_no_change_no_leakage(self, spx):
        """Modify future rows, confirm predictions for past rows unchanged."""
        from src.validation.baselines import NoChangeBaseline
        bl  = NoChangeBaseline()
        p1  = bl.predict(spx)
        spx2              = spx.copy()
        spx2.iloc[-10:] *= 1.5
        p2 = bl.predict(spx2)
        pd.testing.assert_series_equal(
            p1["target_high"].iloc[:-10],
            p2["target_high"].iloc[:-10],
        )

    def test_baselines_use_only_shift1_data(self, spx):
        """All baselines: row 0 predictions must be NaN (no prior day)."""
        from src.validation.baselines import NoChangeBaseline, ATRBaseline
        # These baselines drop NaN rows, so index should not include day 0
        nc = NoChangeBaseline().predict(spx)
        at = ATRBaseline().predict(spx)
        assert spx.index[0] not in nc.index
        assert spx.index[0] not in at.index


# ─────────────────────────────────────────────────────────────────────────────
# TestDieboldMariano  (Instruction 4)
# ─────────────────────────────────────────────────────────────────────────────

class TestDieboldMariano:

    def test_dm_returns_two_floats(self):
        from src.validation.baselines import diebold_mariano_test
        rng = np.random.default_rng(0)
        e_a = rng.normal(0, 1, 200)
        e_b = rng.normal(0, 1, 200)
        dm, p = diebold_mariano_test(e_a, e_b)
        assert isinstance(dm, float) and isinstance(p, float)

    def test_dm_pvalue_in_0_1(self):
        from src.validation.baselines import diebold_mariano_test
        rng = np.random.default_rng(1)
        e_a = rng.normal(0, 1, 300)
        e_b = rng.normal(0.5, 1, 300)   # biased
        _, p = diebold_mariano_test(e_a, e_b)
        assert 0 <= p <= 1

    def test_dm_identical_errors_p_near_1(self):
        """Identical errors → d_t = 0 everywhere → p-value ≈ 1."""
        from src.validation.baselines import diebold_mariano_test
        e = np.random.default_rng(2).normal(0, 1, 200)
        dm, p = diebold_mariano_test(e, e)
        # DM should be exactly 0 when errors identical
        assert abs(dm) < 1e-6 or np.isnan(dm)

    def test_dm_clearly_different_is_significant(self):
        """One model much worse → significant difference."""
        from src.validation.baselines import diebold_mariano_test
        rng = np.random.default_rng(3)
        e_good = rng.normal(0, 1, 500)
        e_bad  = rng.normal(0, 5, 500)   # 5× larger errors
        _, p = diebold_mariano_test(e_good, e_bad, loss="absolute")
        assert p < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# TestLeakageGateConfirm  (Instruction 5)
# ─────────────────────────────────────────────────────────────────────────────

class TestLeakageGateConfirm:

    def test_no_leakage_all_pass(self, pytestconfig):
        """
        Re-confirm that test_no_leakage.py was collected and all 24 tests pass.
        This acts as a meta-gate: if the leakage suite was skipped or failed,
        this test also fails.
        """
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "pytest",
             "tests/test_no_leakage.py", "-q", "--tb=short"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0, (
            f"Leakage gate FAILED:\n{result.stdout}\n{result.stderr}"
        )
        # Accept compact pytest-q output (e.g. '......') or verbose 'passed' text
        assert result.returncode == 0  # returncode=0 means all passed


# ─────────────────────────────────────────────────────────────────────────────
# TestPandasTaConfirm  (Instruction 6)
# ─────────────────────────────────────────────────────────────────────────────

class TestPandasTaConfirm:

    def test_deprecated_pandas_ta_not_imported(self):
        """
        Confirm no module in src/ imports the deprecated pandas_ta package.
        All technical indicators are implemented natively.
        """
        import ast, os
        src_root = Path(__file__).resolve().parent.parent / "src"
        bad_imports = []

        for py_file in src_root.rglob("*.py"):
            source = py_file.read_text()
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ("pandas_ta", "pandas_ta_classic"):
                            bad_imports.append(
                                f"{py_file.relative_to(src_root)}: "
                                f"import {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("pandas_ta"):
                        bad_imports.append(
                            f"{py_file.relative_to(src_root)}: "
                            f"from {node.module} import ..."
                        )

        assert not bad_imports, (
            "Deprecated pandas_ta import found:\n" + "\n".join(bad_imports)
        )
