import pandas as pd

from src.pipeline.hybrid_forecast_generator import (
    DEFAULT_COMPONENT_SELECTION,
    _assemble_hybrid_components,
    _infer_next_trading_day,
)


def test_infer_next_trading_day_skips_weekend():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-03-05"),
        pd.Timestamp("2026-03-06"),
    ])
    assert _infer_next_trading_day(idx) == "2026-03-09"


def test_assemble_hybrid_components():
    model_row = pd.Series(
        {
            "target_gap_ret": 0.001,
            "target_high_from_open": 0.010,
            "target_low_from_open": 0.008,
            "target_close_from_open": 0.002,
        },
        name=pd.Timestamp("2026-03-06"),
    )
    baseline_row = pd.Series(
        {
            "target_gap_ret": -0.001,
            "target_high_from_open": 0.020,
            "target_low_from_open": 0.018,
            "target_close_from_open": -0.002,
        },
        name=pd.Timestamp("2026-03-06"),
    )

    out = _assemble_hybrid_components(model_row, baseline_row, DEFAULT_COMPONENT_SELECTION)
    assert out.loc[pd.Timestamp("2026-03-06"), "target_gap_ret"] == -0.001
    assert out.loc[pd.Timestamp("2026-03-06"), "target_high_from_open"] == 0.010
    assert out.loc[pd.Timestamp("2026-03-06"), "target_low_from_open"] == 0.008
    assert out.loc[pd.Timestamp("2026-03-06"), "target_close_from_open"] == 0.002
