from __future__ import annotations

import pandas as pd

from src.models.gap_forecaster_databento import (
    DB_GAP_FEATURES,
    GAP_TARGET,
    build_gap_feature_matrix,
    align_gap_data,
)


def test_build_gap_feature_matrix():
    idx = pd.to_datetime(["2026-03-05", "2026-03-06"])
    base = pd.DataFrame({"a": [1.0, 2.0]}, index=idx)
    db = pd.DataFrame(
        {
            "es_overnight_ret": [0.1, 0.2],
            "es_preopen_ret_last_60m": [0.01, 0.02],
            "es_preopen_ret_last_30m": [0.03, 0.04],
            "es_overnight_range_pct": [0.05, 0.06],
        },
        index=idx,
    )
    out = build_gap_feature_matrix(base, db)
    assert list(out.columns) == ["a"] + DB_GAP_FEATURES
    assert len(out) == 2


def test_align_gap_data():
    idx = pd.to_datetime(["2026-03-05", "2026-03-06"])
    X = pd.DataFrame({"x": [1.0, 2.0]}, index=idx)
    y = pd.Series([0.01, 0.02], index=idx, name=GAP_TARGET)
    X2, y2 = align_gap_data(X, y)
    assert len(X2) == 2
    assert len(y2) == 2
    assert y2.name == GAP_TARGET
