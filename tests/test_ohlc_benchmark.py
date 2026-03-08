import pandas as pd

from src.evaluation.ohlc_benchmark import (
    build_naive_component_baseline,
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


def test_compare_metric_dicts():
    model = {
        "open": {"mae": 10.0, "rmse": 15.0},
        "close_direction_accuracy": 0.54,
    }
    baseline = {
        "open": {"mae": 12.0, "rmse": 18.0},
        "close_direction_accuracy": 0.50,
    }

    out = compare_metric_dicts(model, baseline)

    assert out["open"]["mae"] == 2.0
    assert out["open"]["rmse"] == 3.0
    assert abs(out["close_direction_accuracy"] - 0.04) < 1e-12
