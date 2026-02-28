"""
tests/test_targets_models.py
============================
Phase 3 test gate — Target Engineering, Splitter, Models, Trainer,
Metrics, and Calibration.

Test groups
-----------
TestTargetEngineer      (12 tests)
TestWalkForwardSplitter (10 tests)
TestBaseModel           ( 6 tests)
TestRandomForestModel   ( 8 tests)
TestLinearModels        ( 6 tests)
TestTrainer             ( 8 tests)
TestMetrics             (10 tests)
TestCalibration         ( 6 tests)
────────────────────────────────
Total:                   66 tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── lazy imports so missing optional deps skip gracefully ─────────────────────
try:
    import xgboost as _xgb
    _xgb_available = True
except ImportError:
    _xgb_available = False

xgb_skip = pytest.mark.skipif(not _xgb_available, reason="xgboost not installed")

# ── module-level dummy for pickle/joblib tests ───────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from src.models.base_model import BaseModel as _BaseModel

class _DummyModelForSave(_BaseModel):
    task = "classification"
    def fit(self, X, y): self._fitted = True; return self
    def predict(self, X): return np.zeros(len(X))
    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X))*0.6, np.ones(len(X))*0.4])


# ── shared fixtures ───────────────────────────────────────────────────────────

N_ROWS    = 800
N_FEATS   = 20
RNG       = np.random.default_rng(42)

def _make_spx_df(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily OHLCV data."""
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = 3000 + np.cumsum(rng.normal(0, 10, n))
    high  = close + rng.uniform(5, 30, n)
    low   = close - rng.uniform(5, 30, n)
    open_ = close + rng.normal(0, 5, n)
    vol   = rng.integers(1_000_000, 5_000_000, n)
    df    = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )
    return df


def _make_feature_df(n: int = N_ROWS) -> pd.DataFrame:
    X, _ = make_classification(
        n_samples=n, n_features=N_FEATS, n_informative=10,
        n_redundant=5, random_state=42
    )
    dates = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame(X, index=dates,
                        columns=[f"feat_{i}" for i in range(N_FEATS)])


def _make_target_series(n: int = N_ROWS) -> pd.Series:
    dates = pd.bdate_range("2018-01-01", periods=n)
    y = pd.Series(
        RNG.integers(0, 2, n).astype(int), index=dates, name="next_high_bin_050"
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# TestTargetEngineer
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetEngineer:

    @pytest.fixture
    def spx_parquet(self, tmp_path: Path) -> Path:
        df = _make_spx_df(500)
        p  = tmp_path / "spx_daily.parquet"
        df.to_parquet(p)
        return p

    def test_returns_dataframe(self, spx_parquet):
        from src.targets.engineer import engineer_targets
        tgt = engineer_targets(spx_path=spx_parquet, save=False)
        assert isinstance(tgt, pd.DataFrame)

    def test_row_count_minus_one(self, spx_parquet):
        """Last row dropped (NaN from shift(-1))."""
        from src.targets.engineer import engineer_targets
        df  = pd.read_parquet(spx_parquet)
        tgt = engineer_targets(spx_path=spx_parquet, save=False)
        assert len(tgt) == len(df) - 1

    def test_expected_columns_present(self, spx_parquet):
        from src.targets.engineer import engineer_targets
        tgt = engineer_targets(spx_path=spx_parquet, save=False)
        required = {
            "next_high_pct", "next_low_pct", "next_range_pct",
            "next_high_bin_050", "next_high_bin_100", "next_high_bin_150",
            "next_low_bin_050",  "next_low_bin_100",  "next_low_bin_150",
            "next_range_bin_med",
        }
        assert required.issubset(set(tgt.columns))

    def test_no_target_nan_in_first_half(self, spx_parquet):
        from src.targets.engineer import engineer_targets
        tgt = engineer_targets(spx_path=spx_parquet, save=False)
        mid = len(tgt) // 2
        assert tgt.iloc[mid:][["next_high_pct", "next_low_pct"]].isna().sum().sum() == 0

    def test_binary_targets_are_0_or_1(self, spx_parquet):
        from src.targets.engineer import engineer_targets
        tgt = engineer_targets(spx_path=spx_parquet, save=False)
        bin_cols = [c for c in tgt.columns if "_bin_" in c]
        for col in bin_cols:
            vals = tgt[col].dropna().values
            assert set(vals).issubset({0, 1}), f"{col} has non-binary values"

    def test_next_high_pct_positive_mostly(self, spx_parquet):
        """High of next day should exceed prior close most of the time."""
        from src.targets.engineer import engineer_targets
        tgt = engineer_targets(spx_path=spx_parquet, save=False)
        pct_positive = (tgt["next_high_pct"] > 0).mean()
        assert pct_positive > 0.7

    def test_next_low_pct_negative_mostly(self, spx_parquet):
        from src.targets.engineer import engineer_targets
        tgt = engineer_targets(spx_path=spx_parquet, save=False)
        pct_negative = (tgt["next_low_pct"] < 0).mean()
        assert pct_negative > 0.7

    def test_range_target_non_negative(self, spx_parquet):
        from src.targets.engineer import engineer_targets
        tgt = engineer_targets(spx_path=spx_parquet, save=False)
        assert (tgt["next_range_pct"].dropna() >= 0).all()

    def test_save_creates_file(self, spx_parquet, tmp_path):
        from src.targets.engineer import engineer_targets
        out = tmp_path / "out"
        engineer_targets(spx_path=spx_parquet, save=True, output_path=out)
        assert (out / "targets.parquet").exists()

    def test_align_features_targets_inner_join(self, spx_parquet):
        from src.targets.engineer import engineer_targets, align_features_targets
        tgt = engineer_targets(spx_path=spx_parquet, save=False)
        feats = _make_feature_df(len(tgt))
        feats.index = tgt.index   # same dates
        X, y = align_features_targets(feats, tgt, target_cols=["next_high_bin_050"])
        assert len(X) == len(y)

    def test_align_drops_nan_target_rows(self, spx_parquet):
        from src.targets.engineer import engineer_targets, align_features_targets
        tgt = engineer_targets(spx_path=spx_parquet, save=False)
        # Inject a NaN
        tgt.iloc[5, tgt.columns.get_loc("next_high_bin_050")] = np.nan
        feats = _make_feature_df(len(tgt))
        feats.index = tgt.index
        X, y = align_features_targets(feats, tgt, target_cols=["next_high_bin_050"])
        assert y["next_high_bin_050"].isna().sum() == 0

    def test_file_not_found_raises(self, tmp_path):
        from src.targets.engineer import engineer_targets
        with pytest.raises(FileNotFoundError):
            engineer_targets(spx_path=tmp_path / "missing.parquet", save=False)


# ─────────────────────────────────────────────────────────────────────────────
# TestWalkForwardSplitter
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForwardSplitter:

    @pytest.fixture
    def Xdf(self) -> pd.DataFrame:
        return _make_feature_df(700)

    def test_returns_list_of_tuples(self, Xdf):
        from src.targets.splitter import WalkForwardSplitter
        wf = WalkForwardSplitter()
        splits = wf.split(Xdf)
        assert isinstance(splits, list)
        assert all(isinstance(s, tuple) and len(s) == 2 for s in splits)

    def test_no_test_index_in_train(self, Xdf):
        from src.targets.splitter import WalkForwardSplitter
        wf = WalkForwardSplitter()
        for tr, te in wf.split(Xdf):
            assert len(np.intersect1d(tr, te)) == 0

    def test_train_always_before_test(self, Xdf):
        from src.targets.splitter import WalkForwardSplitter
        wf = WalkForwardSplitter()
        for tr, te in wf.split(Xdf):
            assert tr.max() < te.min()

    def test_expanding_train_grows(self, Xdf):
        from src.targets.splitter import WalkForwardSplitter
        wf = WalkForwardSplitter()
        splits = wf.split(Xdf)
        train_sizes = [len(tr) for tr, _ in splits]
        assert all(a <= b for a, b in zip(train_sizes, train_sizes[1:]))

    def test_test_folds_non_overlapping(self, Xdf):
        from src.targets.splitter import WalkForwardSplitter
        wf = WalkForwardSplitter()
        splits = wf.split(Xdf)
        te_indices = [set(te) for _, te in splits]
        for i, s in enumerate(te_indices):
            for j, t in enumerate(te_indices):
                if i != j:
                    assert len(s & t) == 0

    def test_gap_respected(self, Xdf):
        from src.targets.splitter import WalkForwardSplitter, SplitConfig
        cfg = SplitConfig(gap_rows=5)
        wf  = WalkForwardSplitter(config=cfg)
        for tr, te in wf.split(Xdf):
            assert te.min() - tr.max() >= 5

    def test_too_small_raises(self):
        from src.targets.splitter import WalkForwardSplitter
        tiny = _make_feature_df(10)
        with pytest.raises(ValueError):
            WalkForwardSplitter().split(tiny)

    def test_rolling_window_mode(self, Xdf):
        from src.targets.splitter import WalkForwardSplitter, SplitConfig
        cfg = SplitConfig(expanding=False, min_train_rows=200, max_train_rows=200)
        wf  = WalkForwardSplitter(config=cfg)
        for tr, _ in wf.split(Xdf):
            assert len(tr) <= 200

    def test_describe_returns_dataframe(self, Xdf):
        from src.targets.splitter import WalkForwardSplitter
        wf = WalkForwardSplitter()
        desc = wf.describe(Xdf)
        assert isinstance(desc, pd.DataFrame)
        assert "fold" in desc.columns

    def test_final_split_helper(self, Xdf):
        from src.targets.splitter import train_test_final_split
        y = _make_target_series(len(Xdf))
        X_tr, X_te, y_tr, y_te = train_test_final_split(Xdf, y.to_frame())
        # test comes after train
        assert X_tr.index[-1] < X_te.index[0]
        assert len(X_tr) + len(X_te) < len(Xdf)   # gap removed rows


# ─────────────────────────────────────────────────────────────────────────────
# TestBaseModel
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseModel:

    def _concrete_model(self):
        """Minimal concrete subclass for testing."""
        from src.models.base_model import BaseModel

        class _DummyModel(BaseModel):
            task = "classification"

            def fit(self, X, y):
                self._fitted = True
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

            def predict_proba(self, X):
                return np.column_stack([
                    np.ones(len(X)) * 0.6,
                    np.ones(len(X)) * 0.4,
                ])

        return _DummyModel(name="dummy")

    def test_repr_contains_name(self):
        m = self._concrete_model()
        assert "dummy" in repr(m)

    def test_not_fitted_before_fit(self):
        m = self._concrete_model()
        assert not m._fitted

    def test_fitted_after_fit(self):
        m = self._concrete_model()
        X = _make_feature_df(50)
        y = _make_target_series(50)
        m.fit(X, y)
        assert m._fitted

    def test_set_threshold(self):
        m = self._concrete_model()
        m.set_threshold(0.7)
        assert m._threshold == 0.7

    def test_invalid_threshold_raises(self):
        m = self._concrete_model()
        with pytest.raises(ValueError):
            m.set_threshold(1.5)

    def test_save_load_roundtrip(self, tmp_path):
        m = _DummyModelForSave(name="dummy_save")
        m._fitted = True
        p = tmp_path / "dummy.pkl"
        m.save(p)
        m2 = _DummyModelForSave.load(p)
        assert m2.name == "dummy_save"
        assert m2._fitted


# ─────────────────────────────────────────────────────────────────────────────
# TestRandomForestModel
# ─────────────────────────────────────────────────────────────────────────────

class TestRandomForestModel:

    @pytest.fixture
    def small_dataset(self):
        X = _make_feature_df(200)
        y = _make_target_series(200)
        return X, y

    def test_fit_sets_fitted(self, small_dataset):
        from src.models.tree_models import RandomForestModel
        X, y = small_dataset
        m = RandomForestModel(params={"n_estimators": 10})
        m.fit(X, y)
        assert m._fitted

    def test_predict_correct_length(self, small_dataset):
        from src.models.tree_models import RandomForestModel
        X, y = small_dataset
        m = RandomForestModel(params={"n_estimators": 10})
        m.fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(X)

    def test_predict_proba_shape(self, small_dataset):
        from src.models.tree_models import RandomForestModel
        X, y = small_dataset
        m = RandomForestModel(params={"n_estimators": 10})
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_proba_sums_to_one(self, small_dataset):
        from src.models.tree_models import RandomForestModel
        X, y = small_dataset
        m = RandomForestModel(params={"n_estimators": 10})
        m.fit(X, y)
        proba = m.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_feature_importances_length(self, small_dataset):
        from src.models.tree_models import RandomForestModel
        X, y = small_dataset
        m = RandomForestModel(params={"n_estimators": 10})
        m.fit(X, y)
        fi = m.feature_importances_
        assert fi is not None
        assert len(fi) == X.shape[1]

    def test_regression_mode(self, small_dataset):
        from src.models.tree_models import RandomForestModel
        X, y = small_dataset
        y_cont = y.astype(float) + np.random.normal(0, 0.1, len(y))
        m = RandomForestModel(task="regression", params={"n_estimators": 10})
        m.fit(X, y_cont)
        preds = m.predict(X)
        assert preds.shape == (len(X),)

    def test_extra_trees_fit_predict(self, small_dataset):
        from src.models.tree_models import ExtraTreesModel
        X, y = small_dataset
        m = ExtraTreesModel(params={"n_estimators": 10})
        m.fit(X, y)
        assert m._fitted
        assert len(m.predict(X)) == len(X)

    def test_unfitted_predict_raises(self):
        from src.models.tree_models import RandomForestModel
        m = RandomForestModel()
        X = _make_feature_df(10)
        with pytest.raises(RuntimeError):
            m.predict(X)


# ─────────────────────────────────────────────────────────────────────────────
# TestLinearModels
# ─────────────────────────────────────────────────────────────────────────────

class TestLinearModels:

    @pytest.fixture
    def small_dataset(self):
        X = _make_feature_df(200)
        y = _make_target_series(200)
        return X, y

    def test_logistic_regression_fits(self, small_dataset):
        from src.models.linear_models import LogisticRegressionModel
        X, y = small_dataset
        m = LogisticRegressionModel()
        m.fit(X, y)
        assert m._fitted

    def test_logistic_proba_valid(self, small_dataset):
        from src.models.linear_models import LogisticRegressionModel
        X, y = small_dataset
        m = LogisticRegressionModel()
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape[1] == 2
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_ridge_regression_fits(self, small_dataset):
        from src.models.linear_models import RidgeRegressionModel
        X, y = small_dataset
        y_cont = y.astype(float)
        m = RidgeRegressionModel()
        m.fit(X, y_cont)
        assert m._fitted

    def test_lasso_regression_fits(self, small_dataset):
        from src.models.linear_models import LassoRegressionModel
        X, y = small_dataset
        y_cont = y.astype(float)
        m = LassoRegressionModel()
        m.fit(X, y_cont)
        preds = m.predict(X)
        assert len(preds) == len(X)

    def test_elasticnet_fits(self, small_dataset):
        from src.models.linear_models import ElasticNetModel
        X, y = small_dataset
        m = ElasticNetModel()
        m.fit(X, y.astype(float))
        assert m._fitted

    def test_logistic_feature_importances_length(self, small_dataset):
        from src.models.linear_models import LogisticRegressionModel
        X, y = small_dataset
        m = LogisticRegressionModel()
        m.fit(X, y)
        fi = m.feature_importances_
        assert fi is not None and len(fi) == X.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# TestTrainer
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainer:

    @pytest.fixture
    def dataset(self):
        X = _make_feature_df(N_ROWS)
        y = _make_target_series(N_ROWS).to_frame()
        return X, y

    @pytest.fixture
    def fast_rf(self):
        from src.models.tree_models import RandomForestModel
        return RandomForestModel(params={"n_estimators": 5, "max_depth": 3})

    def test_run_returns_train_result(self, dataset, fast_rf):
        from src.targets.splitter import WalkForwardSplitter, SplitConfig
        from src.models.trainer import Trainer
        X, y = dataset
        cfg = SplitConfig(min_train_rows=400, test_rows=100, step_rows=100)
        trainer = Trainer(model=fast_rf, splitter=WalkForwardSplitter(cfg))
        result = trainer.run(X, y, target_col="next_high_bin_050")
        assert result is not None

    def test_oos_pred_index_within_X_index(self, dataset, fast_rf):
        from src.targets.splitter import WalkForwardSplitter, SplitConfig
        from src.models.trainer import Trainer
        X, y = dataset
        cfg = SplitConfig(min_train_rows=400, test_rows=100, step_rows=100)
        trainer = Trainer(model=fast_rf, splitter=WalkForwardSplitter(cfg))
        result = trainer.run(X, y, target_col="next_high_bin_050")
        assert result.oos_pred.index.isin(X.index).all()

    def test_oos_pred_is_binary(self, dataset, fast_rf):
        from src.targets.splitter import WalkForwardSplitter, SplitConfig
        from src.models.trainer import Trainer
        X, y = dataset
        cfg = SplitConfig(min_train_rows=400, test_rows=100, step_rows=100)
        trainer = Trainer(model=fast_rf, splitter=WalkForwardSplitter(cfg))
        result = trainer.run(X, y, target_col="next_high_bin_050")
        assert set(result.oos_pred.values).issubset({0, 1})

    def test_oos_proba_in_0_1(self, dataset, fast_rf):
        from src.targets.splitter import WalkForwardSplitter, SplitConfig
        from src.models.trainer import Trainer
        X, y = dataset
        cfg = SplitConfig(min_train_rows=400, test_rows=100, step_rows=100)
        trainer = Trainer(model=fast_rf, splitter=WalkForwardSplitter(cfg))
        result = trainer.run(X, y, target_col="next_high_bin_050")
        assert (result.oos_proba >= 0).all() and (result.oos_proba <= 1).all()

    def test_metrics_df_has_expected_cols(self, dataset, fast_rf):
        from src.targets.splitter import WalkForwardSplitter, SplitConfig
        from src.models.trainer import Trainer
        X, y = dataset
        cfg = SplitConfig(min_train_rows=400, test_rows=100, step_rows=100)
        trainer = Trainer(model=fast_rf, splitter=WalkForwardSplitter(cfg))
        result = trainer.run(X, y, target_col="next_high_bin_050")
        for col in ("fold", "accuracy", "f1", "roc_auc"):
            assert col in result.metrics_df.columns

    def test_overall_dict_has_roc_auc(self, dataset, fast_rf):
        from src.targets.splitter import WalkForwardSplitter, SplitConfig
        from src.models.trainer import Trainer
        X, y = dataset
        cfg = SplitConfig(min_train_rows=400, test_rows=100, step_rows=100)
        trainer = Trainer(model=fast_rf, splitter=WalkForwardSplitter(cfg))
        result = trainer.run(X, y, target_col="next_high_bin_050")
        assert "roc_auc" in result.overall

    def test_feature_importance_dataframe(self, dataset, fast_rf):
        from src.targets.splitter import WalkForwardSplitter, SplitConfig
        from src.models.trainer import Trainer
        X, y = dataset
        cfg = SplitConfig(min_train_rows=400, test_rows=100, step_rows=100)
        trainer = Trainer(model=fast_rf, splitter=WalkForwardSplitter(cfg))
        result = trainer.run(X, y, target_col="next_high_bin_050")
        assert result.feature_importance is not None
        assert len(result.feature_importance) == N_FEATS

    def test_no_data_leakage_oos_dates_increasing(self, dataset, fast_rf):
        """OOS predictions must be strictly chronological (no overlap)."""
        from src.targets.splitter import WalkForwardSplitter, SplitConfig
        from src.models.trainer import Trainer
        X, y = dataset
        cfg = SplitConfig(min_train_rows=400, test_rows=100, step_rows=100)
        trainer = Trainer(model=fast_rf, splitter=WalkForwardSplitter(cfg))
        result = trainer.run(X, y, target_col="next_high_bin_050")
        assert result.oos_pred.index.is_monotonic_increasing


# ─────────────────────────────────────────────────────────────────────────────
# TestMetrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:

    @pytest.fixture
    def pred_data(self):
        rng   = np.random.default_rng(0)
        y_t   = rng.integers(0, 2, 200)
        y_p   = (rng.random(200) > 0.45).astype(int)
        y_pr  = rng.random(200)
        return y_t, y_p, y_pr

    def test_classification_report_is_dataframe(self, pred_data):
        from src.validation.metrics import classification_report_df
        y_t, y_p, y_pr = pred_data
        df = classification_report_df(y_t, y_p, y_pr)
        assert isinstance(df, pd.DataFrame)

    def test_classification_report_has_precision(self, pred_data):
        from src.validation.metrics import classification_report_df
        y_t, y_p, _ = pred_data
        df = classification_report_df(y_t, y_p)
        assert "precision" in df.columns

    def test_brier_score_between_0_1(self, pred_data):
        from src.validation.metrics import calibration_metrics
        y_t, _, y_pr = pred_data
        cal = calibration_metrics(y_t, y_pr)
        assert 0 <= cal["brier_score"] <= 1

    def test_ece_between_0_1(self, pred_data):
        from src.validation.metrics import calibration_metrics
        y_t, _, y_pr = pred_data
        cal = calibration_metrics(y_t, y_pr)
        assert 0 <= cal["ece"] <= 1

    def test_reliability_df_is_dataframe(self, pred_data):
        from src.validation.metrics import calibration_metrics
        y_t, _, y_pr = pred_data
        cal = calibration_metrics(y_t, y_pr)
        assert isinstance(cal["reliability_df"], pd.DataFrame)

    def test_trading_metrics_hit_rate_in_0_1(self, pred_data):
        from src.validation.metrics import trading_metrics
        y_t, y_p, _ = pred_data
        tm = trading_metrics(y_t, y_p)
        assert 0 <= tm["hit_rate"] <= 1

    def test_trading_metrics_threshold_sweep(self, pred_data):
        from src.validation.metrics import trading_metrics
        y_t, y_p, y_pr = pred_data
        tm = trading_metrics(y_t, y_p, y_pr)
        assert "threshold_sweep" in tm
        assert len(tm["threshold_sweep"]) > 0

    def test_generate_full_report_keys(self, pred_data):
        from src.validation.metrics import generate_full_report
        y_t, y_p, y_pr = pred_data
        r = generate_full_report(y_t, y_p, y_pr, "test_target")
        for key in ("classification", "confusion", "trading", "calibration"):
            assert key in r

    def test_optimal_threshold_returns_float_tuple(self, pred_data):
        from src.validation.metrics import optimal_threshold
        y_t, _, y_pr = pred_data
        thr, score = optimal_threshold(y_t, y_pr)
        assert isinstance(thr, float)
        assert isinstance(score, float)
        assert 0 <= thr <= 1

    def test_optimal_threshold_score_better_than_default(self, pred_data):
        """Optimised threshold should match or beat the 0.5 default."""
        from src.validation.metrics import optimal_threshold
        from sklearn.metrics import f1_score
        y_t, _, y_pr = pred_data
        thr, _ = optimal_threshold(y_t, y_pr, metric="f1")
        f1_opt = f1_score(y_t, (y_pr >= thr).astype(int), zero_division=0)
        f1_def = f1_score(y_t, (y_pr >= 0.5).astype(int), zero_division=0)
        assert f1_opt >= f1_def - 0.01   # allow tiny float tolerance


# ─────────────────────────────────────────────────────────────────────────────
# TestCalibration
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibration:

    @pytest.fixture
    def fitted_rf(self):
        from src.models.tree_models import RandomForestModel
        X = _make_feature_df(200)
        y = _make_target_series(200)
        m = RandomForestModel(params={"n_estimators": 10})
        m.fit(X, y)
        return m, X, y

    def test_isotonic_calibrator_fit_transform(self, fitted_rf):
        from src.calibration.isotonic import IsotonicCalibrator
        m, X, y = fitted_rf
        raw  = m.predict_proba(X)[:, 1]
        cal  = IsotonicCalibrator()
        cal.fit(raw, y.values)
        cal_p = cal.transform(raw)
        assert len(cal_p) == len(raw)
        assert ((cal_p >= 0) & (cal_p <= 1)).all()

    def test_platt_calibrator_fit_transform(self, fitted_rf):
        from src.calibration.isotonic import PlattCalibrator
        m, X, y = fitted_rf
        raw  = m.predict_proba(X)[:, 1]
        cal  = PlattCalibrator()
        cal.fit(raw, y.values)
        cal_p = cal.transform(raw)
        assert len(cal_p) == len(raw)
        np.testing.assert_array_less(-0.01, cal_p)
        np.testing.assert_array_less(cal_p, 1.01)

    def test_calibrated_predictor_predict_proba_shape(self, fitted_rf):
        from src.calibration.isotonic import CalibratedPredictor
        m, X, y = fitted_rf
        cp = CalibratedPredictor(m)
        cp.calibrate(X, y.values)
        proba = cp.predict_proba(X)
        assert len(proba) == len(X)

    def test_calibrated_predictor_predict_binary(self, fitted_rf):
        from src.calibration.isotonic import CalibratedPredictor
        m, X, y = fitted_rf
        cp = CalibratedPredictor(m)
        cp.calibrate(X, y.values)
        preds = cp.predict(X)
        assert set(preds).issubset({0, 1})

    def test_calibrated_predictor_unfitted_raises(self, fitted_rf):
        """Calibrate must be called before using calibrated proba."""
        from src.calibration.isotonic import (
            CalibratedPredictor, IsotonicCalibrator
        )
        m, X, y = fitted_rf
        iso = IsotonicCalibrator()   # not yet fitted
        cp  = CalibratedPredictor(m, calibrator=iso)
        with pytest.raises(RuntimeError):
            cp.predict_proba(X)

    def test_calibrated_predictor_repr(self, fitted_rf):
        from src.calibration.isotonic import CalibratedPredictor
        m, X, _ = fitted_rf
        cp = CalibratedPredictor(m, threshold=0.6)
        assert "0.6" in repr(cp)
