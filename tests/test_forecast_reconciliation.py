import pandas as pd

from src.evaluation.forecast_reconciliation import (
    reconcile_forecast_to_actual,
    append_reconciliation_history,
)


def test_reconcile_forecast_to_actual_basic():
    forecast = {
        "forecast_for_date": "2026-03-09",
        "generated_from_feature_date": "2026-03-06",
        "prev_close": 100.0,
        "component_source_selection": {
            "target_gap_ret": "rolling_baseline",
            "target_high_from_open": "model",
            "target_low_from_open": "model",
            "target_close_from_open": "model",
        },
        "predicted_ohlc": {
            "open": 101.0,
            "high": 105.0,
            "low": 99.0,
            "close": 102.0,
        },
    }
    actual = pd.Series({"Open": 100.5, "High": 104.0, "Low": 98.0, "Close": 101.0})

    out = reconcile_forecast_to_actual(forecast, actual)

    assert out["errors"]["open_abs_error"] == 0.5
    assert out["errors"]["high_abs_error"] == 1.0
    assert out["errors"]["low_abs_error"] == 1.0
    assert out["errors"]["close_abs_error"] == 1.0
    assert out["direction"]["direction_correct"] is True


def test_append_reconciliation_history_dedupes(tmp_path):
    reconciliation = {
        "forecast_for_date": "2026-03-09",
        "generated_from_feature_date": "2026-03-06",
        "prev_close": 100.0,
        "predicted_ohlc": {"open": 1, "high": 2, "low": 0, "close": 1.5},
        "actual_ohlc": {"open": 1.1, "high": 2.1, "low": -0.1, "close": 1.4},
        "errors": {
            "open_abs_error": 0.1,
            "high_abs_error": 0.1,
            "low_abs_error": 0.1,
            "close_abs_error": 0.1,
            "range_abs_error": 0.2,
        },
        "direction": {"direction_correct": True},
        "component_source_selection": {
            "target_gap_ret": "rolling_baseline",
            "target_high_from_open": "model",
            "target_low_from_open": "model",
            "target_close_from_open": "model",
        },
    }

    path = tmp_path / "forecast_history.csv"
    hist1 = append_reconciliation_history(reconciliation, path)
    hist2 = append_reconciliation_history(reconciliation, path)

    assert len(hist1) == 1
    assert len(hist2) == 1
    assert hist2.iloc[0]["forecast_for_date"] == "2026-03-09"
