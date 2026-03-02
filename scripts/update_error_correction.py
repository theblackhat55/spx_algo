#!/usr/bin/env python3
"""
scripts/update_error_correction.py
===================================
Called by morning reconciliation to:
1. Record yesterday's prediction errors
2. Refit the error correction model
3. Print a summary for the WhatsApp report
"""
import sys
import json
import logging
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("error_correction_update")


def main():
    from src.models.error_features import record_error, load_error_history
    from src.models.error_correction import ErrorCorrector

    # Load yesterday's signal and actuals
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Check for weekday
    yd = date.today() - timedelta(days=1)
    if yd.weekday() >= 5:  # Saturday or Sunday
        print(f"SKIP: {yesterday} is a weekend")
        return

    # Load signal
    signal_dir = Path("output/signals")
    signal_path = signal_dir / f"signal_{yesterday}.json"
    if not signal_path.exists():
        # Try latest signal
        latest = signal_dir / "latest_signal.json"
        if latest.exists():
            signal_path = latest
        else:
            print(f"NO SIGNAL found for {yesterday}")
            return

    with open(signal_path) as f:
        signal = json.load(f)

    signal_date = signal.get("signal_date", yesterday)

    # Get actual data
    spx = pd.read_parquet("data/raw/spx_daily.parquet")
    spx.index = pd.to_datetime(spx.index)

    if pd.Timestamp(signal_date) not in spx.index:
        print(f"NO ACTUAL DATA for {signal_date} (market may have been closed)")
        return

    row = spx.loc[pd.Timestamp(signal_date)]
    prior_close = float(signal.get("prior_close", 0))

    if prior_close <= 0:
        print("ERROR: prior_close missing from signal")
        return

    actual_high_pct = (float(row["High"]) - prior_close) / prior_close
    actual_low_pct = (float(row["Low"]) - prior_close) / prior_close

    # Get predicted values
    pred_high_pct = float(signal.get("pred_high_pct", signal.get("predicted_high_pct", 0)))
    pred_low_pct = float(signal.get("pred_low_pct", signal.get("predicted_low_pct", 0)))

    regime = signal.get("regime", "UNKNOWN")
    vix = float(signal.get("vix_spot", 20))
    dow = pd.Timestamp(signal_date).weekday()

    # Record error
    error_df = record_error(
        date=signal_date,
        pred_high_pct=pred_high_pct,
        pred_low_pct=pred_low_pct,
        actual_high_pct=actual_high_pct,
        actual_low_pct=actual_low_pct,
        regime=regime,
        vix=vix,
        day_of_week=dow,
    )

    print(f"\n{'='*50}")
    print(f"ERROR CORRECTION UPDATE — {signal_date}")
    print(f"{'='*50}")
    print(f"Pred High: {pred_high_pct:+.4f}%  Actual: {actual_high_pct:+.4f}%  Error: {actual_high_pct - pred_high_pct:+.4f}%")
    print(f"Pred Low:  {pred_low_pct:+.4f}%  Actual: {actual_low_pct:+.4f}%  Error: {actual_low_pct - pred_low_pct:+.4f}%")
    print(f"Regime: {regime}  VIX: {vix:.1f}  History: {len(error_df)} days")

    # Refit correction model
    corrector = ErrorCorrector()
    fit_result = corrector.fit()
    print(f"\nCorrection model: {fit_result['status']}")

    if fit_result["status"] == "fitted":
        print(f"  Train rows: {fit_result['train_rows']}")
        print(f"  MAE high: {fit_result['mae_correction_high']:.6f}")
        print(f"  MAE low:  {fit_result['mae_correction_low']:.6f}")
        print(f"  Mean bias high: {fit_result['mean_error_high']:+.6f}")
        print(f"  Mean bias low:  {fit_result['mean_error_low']:+.6f}")

        # Show what correction would look like for today
        today_dow = date.today().weekday()
        corr_h, corr_l, meta = corrector.correct(
            pred_high_pct, pred_low_pct, regime, vix, today_dow
        )
        if meta.get("correction_high"):
            print(f"\n  Today's correction would be:")
            print(f"    High: {meta['correction_high']:+.6f}%")
            print(f"    Low:  {meta['correction_low']:+.6f}%")
    else:
        print(f"  {fit_result}")

    print(f"{'='*50}")


if __name__ == "__main__":
    main()
