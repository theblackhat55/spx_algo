from pathlib import Path
import json


def test_recent_overnight_report_schema(tmp_path: Path):
    sample = {
        "overlap_rows": 48,
        "train_rows": 38,
        "test_rows": 10,
        "base_feature_count": 96,
        "overnight_feature_count": 102,
        "overnight_added_columns": [
            "es_overnight_gap_pct",
            "es_overnight_ret",
        ],
        "base_component_metrics": {},
        "overnight_component_metrics": {},
        "base_ohlc_metrics": {
            "open": {"mae": 10.0, "rmse": 12.0},
            "close_direction_accuracy": 0.5,
        },
        "overnight_ohlc_metrics": {
            "open": {"mae": 9.0, "rmse": 11.0},
            "close_direction_accuracy": 0.6,
        },
    }

    out = tmp_path / "report.json"
    out.write_text(json.dumps(sample), encoding="utf-8")

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["overlap_rows"] == 48
    assert "overnight_ohlc_metrics" in loaded
    assert "overnight_added_columns" in loaded
