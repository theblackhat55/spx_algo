#!/usr/bin/env python3
import json

from src.pipeline.hybrid_forecast_generator import save_latest_hybrid_ohlc_forecast

if __name__ == "__main__":
    forecast = save_latest_hybrid_ohlc_forecast()
    print("Saved hybrid forecast.")
    print(json.dumps(forecast, indent=2))
