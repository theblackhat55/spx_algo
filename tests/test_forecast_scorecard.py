import pandas as pd

from src.evaluation.forecast_scorecard import build_scorecard


def test_build_scorecard_basic():
    df = pd.DataFrame(
        {
            "forecast_for_date": pd.to_datetime(["2026-03-09", "2026-03-10"]),
            "open_abs_error": [10.0, 20.0],
            "high_abs_error": [15.0, 25.0],
            "low_abs_error": [12.0, 22.0],
            "close_abs_error": [18.0, 28.0],
            "range_abs_error": [8.0, 12.0],
            "direction_correct": [True, False],
            "gap_source": ["rolling_baseline", "rolling_baseline"],
            "high_source": ["model", "model"],
            "low_source": ["model", "model"],
            "close_source": ["model", "model"],
        }
    )

    out = build_scorecard(df)

    assert out["all_history"]["rows"] == 2
    assert out["all_history"]["open_mae"] == 15.0
    assert out["all_history"]["direction_accuracy"] == 0.5
    assert out["source_counts"]["gap_source"]["rolling_baseline"] == 2
    assert out["latest_forecast_date"] == "2026-03-10"
