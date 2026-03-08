#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from shutil import copy2

from src.pipeline.gap_augmented_hybrid_forecast import save_gap_augmented_hybrid_forecast


LATEST_PATH = Path("output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json")
ARCHIVE_DIR = Path("output/forecasts/archive")


def main() -> None:
    forecast = save_gap_augmented_hybrid_forecast(out_path=LATEST_PATH)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    forecast_date = forecast["forecast_for_date"]
    archive_path = ARCHIVE_DIR / f"{forecast_date}_gap_augmented_hybrid_ohlc_forecast.json"
    copy2(LATEST_PATH, archive_path)

    print(f"Saved latest forecast to: {LATEST_PATH}")
    print(f"Archived forecast to   : {archive_path}")
    print("Forecast date         :", forecast["forecast_for_date"])
    print("Feature date          :", forecast["generated_from_feature_date"])
    print("Source selection      :", json.dumps(forecast["component_source_selection"], indent=2))
    print("Predicted OHLC        :", json.dumps(forecast["predicted_ohlc"], indent=2))


if __name__ == "__main__":
    main()
