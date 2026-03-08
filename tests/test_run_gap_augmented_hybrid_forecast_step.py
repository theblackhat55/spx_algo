from __future__ import annotations

from pathlib import Path


def test_archive_filename_format():
    forecast_date = "2026-03-09"
    p = Path(f"output/forecasts/archive/{forecast_date}_gap_augmented_hybrid_ohlc_forecast.json")
    assert p.name == "2026-03-09_gap_augmented_hybrid_ohlc_forecast.json"
