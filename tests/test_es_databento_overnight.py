from __future__ import annotations

import pandas as pd

from src.features.es_databento_overnight import build_es_databento_overnight_features


def test_build_es_databento_overnight_features_basic():
    idx = pd.to_datetime(
        [
            "2026-03-05 23:00:00+00:00",
            "2026-03-05 23:01:00+00:00",
            "2026-03-06 13:30:00+00:00",
            "2026-03-06 14:00:00+00:00",
            "2026-03-06 14:29:00+00:00",
        ],
        utc=True,
    )
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [101, 102, 103, 104, 105],
            "volume": [10, 20, 30, 40, 50],
        },
        index=idx,
    )

    out = build_es_databento_overnight_features(df)
    assert len(out) == 1
    row = out.iloc[0]

    assert row["es_overnight_open"] == 100
    assert row["es_overnight_close"] == 105
    assert row["es_overnight_high"] == 105
    assert row["es_overnight_low"] == 99
    assert row["es_overnight_volume"] == 150
    assert row["es_overnight_ret"] > 0
    assert row["es_overnight_range_pct"] > 0
    assert "es_preopen_ret_last_60m" in out.columns
    assert "es_preopen_ret_last_30m" in out.columns


def test_build_es_databento_overnight_features_two_sessions():
    idx = pd.to_datetime(
        [
            "2026-03-04 23:00:00+00:00",
            "2026-03-05 14:29:00+00:00",
            "2026-03-05 23:00:00+00:00",
            "2026-03-06 14:29:00+00:00",
        ],
        utc=True,
    )
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103],
            "high": [101, 102, 103, 104],
            "low": [99, 100, 101, 102],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1, 2, 3, 4],
        },
        index=idx,
    )

    out = build_es_databento_overnight_features(df)
    assert len(out) == 2
    assert out.index[0] == pd.Timestamp("2026-03-05")
    assert out.index[1] == pd.Timestamp("2026-03-06")
