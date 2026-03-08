#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from src.evaluation.forecast_reconciliation import (
    append_reconciliation_history,
    load_forecast,
    load_spx_actuals,
    reconcile_forecast_to_actual,
    save_reconciliation_json,
)


LATEST_FORECAST = Path("output/forecasts/latest_hybrid_ohlc_forecast.json")
REPORTS_DIR = Path("output/reports")
HISTORY_CSV = REPORTS_DIR / "forecast_history.csv"


def main() -> None:
    forecast = load_forecast(LATEST_FORECAST)
    forecast_date = forecast["forecast_for_date"]

    spx = load_spx_actuals()
    dt = spx.index.normalize()

    match = spx.loc[dt == dt[dt == dt].normalize()] if False else None  # no-op to avoid lint weirdness

    target_idx = None
    for idx in spx.index:
        if idx.strftime("%Y-%m-%d") == forecast_date:
            target_idx = idx
            break

    if target_idx is None:
        raise RuntimeError(
            f"No actual SPX OHLC found yet for forecast date {forecast_date}. "
            "Run reconciliation after market data is available."
        )

    reconciliation = reconcile_forecast_to_actual(forecast, spx.loc[target_idx])

    report_file = REPORTS_DIR / f"forecast_reconcile_{forecast_date}.json"
    save_reconciliation_json(reconciliation, report_file)
    hist = append_reconciliation_history(reconciliation, HISTORY_CSV)

    print("Saved reconciliation report to:", report_file)
    print("Updated forecast history to:", HISTORY_CSV)
    print("Forecast date:", forecast_date)
    print("Open abs error:", reconciliation["errors"]["open_abs_error"])
    print("High abs error:", reconciliation["errors"]["high_abs_error"])
    print("Low abs error:", reconciliation["errors"]["low_abs_error"])
    print("Close abs error:", reconciliation["errors"]["close_abs_error"])
    print("Range abs error:", reconciliation["errors"]["range_abs_error"])
    print("Direction correct:", reconciliation["direction"]["direction_correct"])
    print("History rows:", len(hist))


if __name__ == "__main__":
    main()
