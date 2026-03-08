import numpy as np
import pandas as pd

from src.models.ohlc_forecaster import (
    OHLC_TARGET_COLUMNS,
    align_features_and_ohlc_targets,
    evaluate_component_predictions,
    predict_ohlc_components,
    reconstruct_ohlc,
    train_ohlc_models,
)


def _sample_features(n=40):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "feat_1": np.linspace(0, 1, n),
            "feat_2": np.linspace(1, 2, n),
            "feat_3": np.sin(np.linspace(0, 3, n)),
        },
        index=idx,
    )


def _sample_targets(n=40):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "target_gap_ret": np.full(n, 0.001),
            "target_high_from_open": np.full(n, 0.01),
            "target_low_from_open": np.full(n, 0.008),
            "target_close_from_open": np.full(n, 0.002),
        },
        index=idx,
    )


def test_align_features_and_ohlc_targets():
    X = _sample_features()
    y = _sample_targets()
    X2, y2 = align_features_and_ohlc_targets(X, y)

    assert list(y2.columns) == OHLC_TARGET_COLUMNS
    assert len(X2) == len(y2) == 40


def test_train_and_predict_ohlc_models():
    X = _sample_features()
    y = _sample_targets()

    models = train_ohlc_models(X, y, model_params={"n_estimators": 20})
    preds = predict_ohlc_components(models, X)

    assert list(preds.columns) == OHLC_TARGET_COLUMNS
    assert preds.shape == y.shape


def test_reconstruct_ohlc_enforces_bounds():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    prev_close = pd.Series([100.0, 101.0, 102.0], index=idx)

    components = pd.DataFrame(
        {
            "target_gap_ret": [0.01, 0.0, -0.01],
            "target_high_from_open": [0.02, 0.01, 0.005],
            "target_low_from_open": [0.01, 0.02, 0.01],
            "target_close_from_open": [0.03, -0.01, -0.02],
        },
        index=idx,
    )

    out = reconstruct_ohlc(prev_close, components)

    assert set(out.columns) == {"pred_open", "pred_high", "pred_low", "pred_close"}
    assert (out["pred_high"] >= out["pred_open"]).all()
    assert (out["pred_high"] >= out["pred_close"]).all()
    assert (out["pred_low"] <= out["pred_open"]).all()
    assert (out["pred_low"] <= out["pred_close"]).all()


def test_evaluate_component_predictions_returns_metrics():
    y_true = _sample_targets(10)
    y_pred = y_true.copy()
    y_pred["target_gap_ret"] += 0.001

    metrics = evaluate_component_predictions(y_true, y_pred)
    assert "target_gap_ret" in metrics
    assert "mae" in metrics["target_gap_ret"]
    assert metrics["target_gap_ret"]["mae"] > 0
