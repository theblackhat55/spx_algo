import pandas as pd

from src.evaluation.ohlc_benchmark import (
    build_naive_component_baseline,
    build_rolling_component_baseline,
    compare_metric_dicts,
)
from src.models.ohlc_forecaster import OHLC_TARGET_COLUMNS


def test_build_naive_component_baseline():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    y_train = pd.DataFrame(
        {
            "target_gap_ret": [0.001, 0.002, 0.003, 0.004, 0.005],
            "target_high_from_open": [0.010, 0.011, 0.012, 0.013, 0.014],
            "target_low_from_open": [0.008, 0.009, 0.010, 0.011, 0.012],
            "target_close_from_open": [0.002, 0.001, 0.000, -0.001, -0.002],
        },
        index=idx,
    )

    test_idx = pd.date_range("2024-02-01", periods=3, freq="B")
    baseline = build_naive_component_baseline(y_train, test_idx)

    assert list(baseline.columns) == OHLC_TARGET_COLUMNS
    assert len(baseline) == 3
    assert baseline.loc[test_idx[0], "target_gap_ret"] == y_train["target_gap_ret"].mean()


def test_build_rolling_component_baseline():
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    y_all = pd.DataFrame(
        {
            "target_gap_ret": [1, 2, 3, 4, 5, 6],
            "target_high_from_open": [10, 11, 12, 13, 14, 15],
            "target_low_from_open": [20, 21, 22, 23, 24, 25],
            "target_close_from_open": [30, 31, 32, 33, 34, 35],
        },
        index=idx,
    )

    test_idx = pd.DatetimeIndex([idx[4], idx[5]])
    baseline = build_rolling_component_baseline(y_all, test_idx, window=3)

    assert baseline.loc[idx[4], "target_gap_ret"] == (2 + 3 + 4) / 3
    assert baseline.loc[idx[5], "target_gap_ret"] == (3 + 4 + 5) / 3


def test_compare_metric_dicts():
    model = {
        "open": {"mae": 10.0, "rmse": 15.0},
        "close_direction_accuracy": 0.54,
        "range": {"actual_mean": 70.0, "pred_mean": 72.0, "mae": 20.0},
    }
    baseline = {
        "open": {"mae": 12.0, "rmse": 18.0},
        "close_direction_accuracy": 0.50,
        "range": {"actual_mean": 70.0, "pred_mean": 80.0, "mae": 25.0},
    }

    out = compare_metric_dicts(model, baseline)

    assert out["open"]["mae"] == 2.0
    assert out["open"]["rmse"] == 3.0
    assert abs(out["close_direction_accuracy"] - 0.04) < 1e-12
    assert out["range"]["mae"] == 5.0
    assert "actual_mean" not in out["range"]
    assert "pred_mean" not in out["range"]
