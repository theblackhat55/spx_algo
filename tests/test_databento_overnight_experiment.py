from __future__ import annotations

from scripts.experiments.evaluate_databento_overnight_experiment import _metric_diff


def test_metric_diff_nested():
    base = {
        "open": {"mae": 10.0, "rmse": 12.0},
        "close_direction_accuracy": 0.50,
    }
    db = {
        "open": {"mae": 8.5, "rmse": 11.0},
        "close_direction_accuracy": 0.55,
    }
    out = _metric_diff(base, db)
    assert abs(out["open"]["mae"] - (-1.5)) < 1e-12
    assert abs(out["open"]["rmse"] - (-1.0)) < 1e-12
    assert abs(out["close_direction_accuracy"] - 0.05) < 1e-12
