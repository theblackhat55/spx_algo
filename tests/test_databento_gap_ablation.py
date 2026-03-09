from __future__ import annotations

from scripts.evaluate_databento_gap_ablation import _metric_diff


def test_metric_diff_nested_gap_ablation():
    base = {
        "open": {"mae": 10.0, "rmse": 12.0},
        "close_direction_accuracy": 0.50,
    }
    new = {
        "open": {"mae": 9.0, "rmse": 11.5},
        "close_direction_accuracy": 0.55,
    }
    out = _metric_diff(base, new)
    assert abs(out["open"]["mae"] - (-1.0)) < 1e-12
    assert abs(out["open"]["rmse"] - (-0.5)) < 1e-12
    assert abs(out["close_direction_accuracy"] - 0.05) < 1e-12
