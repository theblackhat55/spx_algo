import pandas as pd

from src.evaluation.ohlc_hybrid import (
    build_hybrid_components,
    choose_best_component_source,
)
from src.models.ohlc_forecaster import OHLC_TARGET_COLUMNS


def test_choose_best_component_source():
    model_metrics = {
        "target_gap_ret": {"mae": 0.50},
        "target_high_from_open": {"mae": 0.20},
        "target_low_from_open": {"mae": 0.30},
        "target_close_from_open": {"mae": 0.40},
    }
    rolling_metrics = {
        "target_gap_ret": {"mae": 0.40},
        "target_high_from_open": {"mae": 0.25},
        "target_low_from_open": {"mae": 0.35},
        "target_close_from_open": {"mae": 0.10},
    }

    out = choose_best_component_source(model_metrics, rolling_metrics)

    assert out["target_gap_ret"] == "rolling_baseline"
    assert out["target_high_from_open"] == "model"
    assert out["target_low_from_open"] == "model"
    assert out["target_close_from_open"] == "rolling_baseline"


def test_build_hybrid_components():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    model_df = pd.DataFrame(
        {
            "target_gap_ret": [1, 1, 1],
            "target_high_from_open": [2, 2, 2],
            "target_low_from_open": [3, 3, 3],
            "target_close_from_open": [4, 4, 4],
        },
        index=idx,
    )
    rolling_df = pd.DataFrame(
        {
            "target_gap_ret": [10, 10, 10],
            "target_high_from_open": [20, 20, 20],
            "target_low_from_open": [30, 30, 30],
            "target_close_from_open": [40, 40, 40],
        },
        index=idx,
    )
    selection = {
        "target_gap_ret": "rolling_baseline",
        "target_high_from_open": "model",
        "target_low_from_open": "rolling_baseline",
        "target_close_from_open": "model",
    }

    out = build_hybrid_components(model_df, rolling_df, selection)

    assert list(out.columns) == OHLC_TARGET_COLUMNS
    assert (out["target_gap_ret"] == 10).all()
    assert (out["target_high_from_open"] == 2).all()
    assert (out["target_low_from_open"] == 30).all()
    assert (out["target_close_from_open"] == 4).all()
