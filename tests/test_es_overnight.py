import pandas as pd

from src.features.es_overnight import build_es_overnight_features_from_5m


def test_build_es_overnight_features_from_5m():
    idx = pd.to_datetime([
        "2026-03-05 15:55:00",
        "2026-03-05 16:05:00",
        "2026-03-05 18:00:00",
        "2026-03-06 08:00:00",
        "2026-03-06 09:25:00",
    ])
    df = pd.DataFrame(
        {
            "Open": [99.5, 100.0, 101.0, 102.0, 103.0],
            "High": [100.0, 101.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 99.0, 100.0, 101.0, 102.0],
            "Close": [99.8, 100.5, 102.0, 103.0, 104.0],
        },
        index=idx,
    )

    out = build_es_overnight_features_from_5m(df)
    assert len(out) >= 1

    expected = {
        "es_overnight_gap_pct",
        "es_overnight_ret",
        "es_overnight_range_pct",
        "es_overnight_high_from_prev_close",
        "es_overnight_low_from_prev_close",
        "es_overnight_close_vs_open",
    }
    assert expected.issubset(set(out.columns))
