import json
from pathlib import Path

import pandas as pd

from src.pipeline.forecast_generator import _infer_next_trading_day


def test_infer_next_trading_day_weekday_progression():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-03-05"),  # Thu
        pd.Timestamp("2026-03-06"),  # Fri
    ])
    nxt = _infer_next_trading_day(idx)
    assert nxt == "2026-03-09"


def test_forecast_json_schema_smoke(tmp_path: Path):
    sample = {
        "forecast_for_date": "2026-03-09",
        "generated_from_feature_date": "2026-03-06",
        "prev_close": 5750.25,
        "predicted_components": {
            "target_gap_ret": 0.001,
            "target_high_from_open": 0.004,
            "target_low_from_open": 0.003,
            "target_close_from_open": 0.002,
        },
        "predicted_ohlc": {
            "open": 5756.0,
            "high": 5779.0,
            "low": 5738.0,
            "close": 5768.0,
        },
        "model_artifacts": {
            "model_dir": "output/models/ohlc",
            "forecast_file": "output/forecasts/latest_ohlc_forecast.json",
        },
    }

    out = tmp_path / "forecast.json"
    out.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert "forecast_for_date" in loaded
    assert "predicted_components" in loaded
    assert "predicted_ohlc" in loaded
    assert set(loaded["predicted_ohlc"].keys()) == {"open", "high", "low", "close"}
