import pandas as pd

from src.features.es_features import compute_es_features


def test_compute_es_features_basic():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    es = pd.DataFrame(
        {
            "Open": [100 + i for i in range(30)],
            "High": [101 + i for i in range(30)],
            "Low": [99 + i for i in range(30)],
            "Close": [100.5 + i for i in range(30)],
        },
        index=idx,
    )
    spx = pd.DataFrame(
        {
            "Close": [4000 + i for i in range(30)],
        },
        index=idx,
    )

    feat = compute_es_features(es, spx_df=spx)

    expected_cols = {
        "es_ret_1d",
        "es_ret_3d",
        "es_ret_5d",
        "es_open_gap_pct",
        "es_intraday_ret",
        "es_range_pct",
        "es_high_from_open",
        "es_low_from_open",
        "es_realized_vol_5",
        "es_realized_vol_20",
        "es_avg_range_5",
        "es_avg_range_20",
        "es_close_vs_ma5",
        "es_close_vs_ma20",
        "es_minus_spx_ret_1d",
        "es_spx_ratio_vs_ma20",
    }

    assert expected_cols.issubset(set(feat.columns))
    assert feat.index.equals(idx)


def test_compute_es_features_missing_cols():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    es = pd.DataFrame({"Open": [1, 2, 3, 4, 5], "Close": [1, 2, 3, 4, 5]}, index=idx)

    try:
        compute_es_features(es)
        assert False, "Expected ValueError for missing columns"
    except ValueError as e:
        assert "missing required columns" in str(e).lower()
