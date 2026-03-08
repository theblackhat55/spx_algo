from pathlib import Path
import json


def test_hybrid_archive_filename_format(tmp_path: Path):
    forecast = {
        "forecast_for_date": "2026-03-09",
        "predicted_ohlc": {"open": 1, "high": 2, "low": 0, "close": 1.5},
    }

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    archive_path = archive_dir / f"{forecast['forecast_for_date']}_hybrid_ohlc_forecast.json"
    archive_path.write_text(json.dumps(forecast), encoding="utf-8")

    assert archive_path.exists()
    assert archive_path.name == "2026-03-09_hybrid_ohlc_forecast.json"
