from __future__ import annotations

import pandas as pd

from scripts.query_databento_ohlc_for_date import build_summary


def test_build_summary_basic():
    idx = pd.to_datetime(
        [
            "2026-03-04 00:00:00+00:00",
            "2026-03-04 14:00:00+00:00",
            "2026-03-04 14:30:00+00:00",
            "2026-03-04 20:59:00+00:00",
        ],
        utc=True,
    )
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103],
            "high": [101, 102, 103, 104],
            "low": [99, 100, 101, 102],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [10, 20, 30, 40],
        },
        index=idx,
    )

    out = build_summary(df, pd.Timestamp("2026-03-04"))
    assert "full_day" in out
    assert "overnight_preopen" in out
    assert "cash_session_like" in out
    assert out["full_day"]["rows"] == 4
