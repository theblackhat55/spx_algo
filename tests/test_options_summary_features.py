import pandas as pd

from src.features.options_features import compute_options_summary_features


def test_compute_options_summary_features_empty():
    idx = pd.to_datetime(["2026-01-05", "2026-01-06"])
    ohlc = pd.DataFrame(
        {"Open": [1, 1], "High": [1, 1], "Low": [1, 1], "Close": [1, 1]},
        index=idx,
    )
    out = compute_options_summary_features(ohlc, None)
    assert out.empty


def test_compute_options_summary_features_alignment():
    idx = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    ohlc = pd.DataFrame(
        {"Open": [1, 1, 1], "High": [1, 1, 1], "Low": [1, 1, 1], "Close": [1, 1, 1]},
        index=idx,
    )

    summary = pd.DataFrame(
        {
            "atm_implied_move_1d": [0.012, 0.013],
            "front_iv_atm": [0.18, 0.19],
            "put_call_skew_25d": [0.04, 0.05],
        },
        index=pd.to_datetime(["2026-01-05", "2026-01-06"]),
    )

    out = compute_options_summary_features(ohlc, summary)
    assert "atm_implied_move_1d" in out.columns
    assert "front_iv_atm" in out.columns
    assert "put_call_skew_25d" in out.columns
    assert out.loc[pd.Timestamp("2026-01-07"), "front_iv_atm"] == 0.19


def test_compute_options_summary_features_ignores_unknown_columns():
    idx = pd.to_datetime(["2026-01-05"])
    ohlc = pd.DataFrame(
        {"Open": [1], "High": [1], "Low": [1], "Close": [1]},
        index=idx,
    )

    summary = pd.DataFrame(
        {"random_col": [123]},
        index=idx,
    )

    out = compute_options_summary_features(ohlc, summary)
    assert out.empty
