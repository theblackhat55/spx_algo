from __future__ import annotations

import pandas as pd

from scripts.generate_backdated_gap_augmented_forecast import _next_business_day, _date_range


def test_next_business_day_weekday():
    d = pd.Timestamp("2026-03-03")
    assert _next_business_day(d) == pd.Timestamp("2026-03-04")


def test_next_business_day_friday():
    d = pd.Timestamp("2026-03-06")
    assert _next_business_day(d) == pd.Timestamp("2026-03-09")


def test_date_range():
    out = _date_range("2026-03-03", "2026-03-05")
    assert out == [pd.Timestamp("2026-03-03"), pd.Timestamp("2026-03-04"), pd.Timestamp("2026-03-05")]
