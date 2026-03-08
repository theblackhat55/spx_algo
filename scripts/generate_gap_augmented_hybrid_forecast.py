#!/usr/bin/env python3
from __future__ import annotations

from src.pipeline.gap_augmented_hybrid_forecast import save_gap_augmented_hybrid_forecast


def main() -> None:
    forecast = save_gap_augmented_hybrid_forecast()
    print("Saved forecast to: output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json")
    print("Forecast for date:", forecast["forecast_for_date"])
    print("Generated from feature date:", forecast["generated_from_feature_date"])
    print("Component source selection:", forecast["component_source_selection"])
    print("Predicted OHLC:", forecast["predicted_ohlc"])


if __name__ == "__main__":
    main()
