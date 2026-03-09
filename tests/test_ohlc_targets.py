import pandas as pd
import numpy as np

from src.targets.ohlc_targets import build_ohlc_component_targets


def test_build_ohlc_component_targets_basic_values():
    idx = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    df = pd.DataFrame(
        {
            "Open":  [100.0, 102.0, 101.0],
            "High":  [103.0, 105.0, 104.0],
            "Low":   [ 99.0, 100.0,  98.0],
            "Close": [101.0, 103.0,  99.0],
        },
        index=idx,
    )

    out = build_ohlc_component_targets(df, dropna=False)

    assert np.isclose(out.loc[idx[0], "target_gap_ret"], 102.0 / 101.0 - 1.0)
    assert np.isclose(out.loc[idx[0], "target_high_from_open"], 105.0 / 102.0 - 1.0)
    assert np.isclose(out.loc[idx[0], "target_low_from_open"], 1.0 - (100.0 / 102.0))
    assert np.isclose(out.loc[idx[0], "target_close_from_open"], 103.0 / 102.0 - 1.0)
    assert np.isclose(out.loc[idx[0], "target_range_from_open"], (105.0 - 100.0) / 102.0)

    assert pd.isna(out.loc[idx[-1], "target_gap_ret"])


def test_build_ohlc_component_targets_dropna():
    idx = pd.to_datetime(["2026-01-05", "2026-01-06"])
    df = pd.DataFrame(
        {
            "Open":  [100.0, 102.0],
            "High":  [103.0, 105.0],
            "Low":   [ 99.0, 100.0],
            "Close": [101.0, 103.0],
        },
        index=idx,
    )

    out = build_ohlc_component_targets(df, dropna=True)
    assert len(out) == 1
    assert out.index[0] == idx[0]


def test_build_ohlc_component_targets_requires_columns():
    idx = pd.to_datetime(["2026-01-05", "2026-01-06"])
    df = pd.DataFrame(
        {
            "Open":  [100.0, 102.0],
            "High":  [103.0, 105.0],
            "Close": [101.0, 103.0],
        },
        index=idx,
    )

    try:
        build_ohlc_component_targets(df)
        assert False, "Expected ValueError for missing Low column"
    except ValueError as e:
        assert "Missing required OHLC columns" in str(e)


def test_close_location_in_range_is_nan_when_zero_range():
    idx = pd.to_datetime(["2026-01-05", "2026-01-06"])
    df = pd.DataFrame(
        {
            "Open":  [100.0, 100.0],
            "High":  [100.0, 100.0],
            "Low":   [100.0, 100.0],
            "Close": [100.0, 100.0],
        },
        index=idx,
    )

    out = build_ohlc_component_targets(df, dropna=False)
    assert pd.isna(out.loc[idx[0], "target_close_loc_in_range"])
