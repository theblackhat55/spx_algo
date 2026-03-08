#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import shutil

from src.pipeline.hybrid_forecast_generator import (
    FORECAST_FILE,
    save_latest_hybrid_ohlc_forecast,
)


ARCHIVE_DIR = Path("output/forecasts/archive")


def main() -> None:
    forecast = save_latest_hybrid_ohlc_forecast()
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    forecast_date = forecast["forecast_for_date"]
    archive_path = ARCHIVE_DIR / f"{forecast_date}_hybrid_ohlc_forecast.json"

    if FORECAST_FILE.exists():
        shutil.copy2(FORECAST_FILE, archive_path)

    print("Saved latest hybrid forecast to:", FORECAST_FILE)
    print("Archived hybrid forecast to:", archive_path)
    print(json.dumps(forecast, indent=2))


if __name__ == "__main__":
    main()
