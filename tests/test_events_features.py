from pathlib import Path
import tempfile

import pandas as pd

from src.features.events import load_event_calendar, compute_event_features


def test_load_event_calendar_empty_missing_file():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "missing.csv"
        df = load_event_calendar(path)
        assert "is_cpi_day" in df.columns
        assert len(df) == 0


def test_compute_event_features_with_manual_flags():
    idx = pd.to_datetime(["2026-01-14", "2026-01-15", "2026-01-16"])
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "event_calendar.csv"
        pd.DataFrame(
            {
                "date": ["2026-01-15"],
                "is_cpi_day": [1],
                "is_fomc_day": [0],
                "is_nfp_day": [0],
                "is_opex_day": [0],
            }
        ).to_csv(path, index=False)

        out = compute_event_features(idx, path)
        assert out.loc[pd.Timestamp("2026-01-15"), "is_cpi_day"] == 1
        assert out.loc[pd.Timestamp("2026-01-14"), "is_cpi_day"] == 0


def test_holiday_adjacent_detection():
    idx = pd.to_datetime(["2026-01-16", "2026-01-20", "2026-01-21"])
    out = compute_event_features(idx)
    assert out.loc[pd.Timestamp("2026-01-16"), "is_holiday_adjacent"] == 1
    assert out.loc[pd.Timestamp("2026-01-20"), "is_holiday_adjacent"] == 1




def test_holiday_adjacent_detection():
    idx = pd.to_datetime(["2026-01-16", "2026-01-20", "2026-01-21"])
    out = compute_event_features(idx)
    assert out.loc[pd.Timestamp("2026-01-16"), "is_holiday_adjacent"] == 1
    assert out.loc[pd.Timestamp("2026-01-20"), "is_holiday_adjacent"] == 1
