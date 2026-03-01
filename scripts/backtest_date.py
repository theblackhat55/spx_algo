#!/usr/bin/env python3
"""
Single-date backtest for SPX Iron Condor.
Usage: python3 scripts/backtest_date.py 2026-02-01
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/root/spx_algo')
os.chdir('/root/spx_algo')

import pandas as pd
import numpy as np
import pickle

def run_single_date_backtest(target_date_str: str):
    from src.backtest.engine import IronCondorEngine, PositionConfig
    from src.calibration.regime import RegimeDetector, Regime

    target_date = pd.Timestamp(target_date_str)
    print(f"\n{'='*60}")
    print(f"  SPX IRON CONDOR BACKTEST: {target_date.date()}")
    print(f"{'='*60}\n")

    # Load market data
    spx_df = pd.read_parquet('data/raw/spx_daily.parquet')
    vix_df = pd.read_parquet('data/raw/vix_daily.parquet')

    if target_date not in spx_df.index:
        nearest = spx_df.index[spx_df.index.get_indexer([target_date], method='nearest')[0]]
        print(f"  {target_date.date()} not a trading day. Using nearest: {nearest.date()}")
        target_date = nearest

    idx = spx_df.index.get_loc(target_date)
    if idx < 252:
        print("  ERROR: Not enough history (need 252 days)")
        return

    # Prior day
    prior_date = spx_df.index[idx - 1]
    prior_close = float(spx_df.loc[prior_date, 'Close'])

    # Actuals
    actual_open = float(spx_df.loc[target_date, 'Open'])
    actual_high = float(spx_df.loc[target_date, 'High'])
    actual_low = float(spx_df.loc[target_date, 'Low'])
    actual_close = float(spx_df.loc[target_date, 'Close'])
    vix_close = float(vix_df.loc[target_date, 'Close']) if target_date in vix_df.index else 20.0

    # Regime
    detector = RegimeDetector()
    regime_series = detector.fit_predict(spx_df.loc[:target_date], vix_df.loc[:target_date])
    regime_val = int(regime_series.get(target_date, Regime.YELLOW))
    regime_labels = {0: 'GREEN', 1: 'YELLOW', 2: 'RED'}
    regime_name = regime_labels.get(regime_val, 'UNKNOWN')

    print(f"  MARKET DATA for {target_date.date()}:")
    print(f"  Prior Close: {prior_close:.2f}")
    print(f"  Open:  {actual_open:.2f}  ({(actual_open/prior_close-1)*100:+.2f}%)")
    print(f"  High:  {actual_high:.2f}  ({(actual_high/prior_close-1)*100:+.2f}%)")
    print(f"  Low:   {actual_low:.2f}  ({(actual_low/prior_close-1)*100:+.2f}%)")
    print(f"  Close: {actual_close:.2f}  ({(actual_close/prior_close-1)*100:+.2f}%)")
    print(f"  VIX:   {vix_close:.2f}")
    print(f"  Regime: {regime_name}")

    # Build features and predict
    print("\n  Building features...")
    from src.features.builder import load_feature_matrix
    features_df = load_feature_matrix()

    if target_date not in features_df.index:
        print(f"  ERROR: {target_date.date()} not in features DataFrame")
        return

    # Load models
    try:
        with open('output/models/regressor_target_high_pct.pkl', 'rb') as f:
            high_model = pickle.load(f)
        with open('output/models/regressor_target_low_pct.pkl', 'rb') as f:
            low_model = pickle.load(f)
    except FileNotFoundError as e:
        print(f"  ERROR: Model not found: {e}")
        return

    feature_cols = [c for c in features_df.columns if not c.startswith('target_') and not c.startswith('next_')]
    X = features_df.loc[[target_date], feature_cols]

    pred_high_pct = float(high_model.predict(X)[0])
    pred_low_pct = float(low_model.predict(X)[0])
    predicted_high = prior_close * (1 + pred_high_pct / 100)
    predicted_low = prior_close * (1 + pred_low_pct / 100)

    # Conformal calibration (last 63 days)
    cal_start = max(0, idx - 63)
    cal_dates = features_df.index[cal_start:idx]
    residuals_high = []
    residuals_low = []
    for cd in cal_dates:
        ci = spx_df.index.get_loc(cd)
        if ci > 0:
            pc = float(spx_df.iloc[ci - 1]['Close'])
            X_cal = features_df.loc[[cd], feature_cols]
            try:
                ph = float(high_model.predict(X_cal)[0])
                pl = float(low_model.predict(X_cal)[0])
                residuals_high.append(abs(float(spx_df.loc[cd, 'High']) - pc * (1 + ph / 100)))
                residuals_low.append(abs(float(spx_df.loc[cd, 'Low']) - pc * (1 + pl / 100)))
            except:
                pass

    q90_high = np.percentile(residuals_high, 90) if residuals_high else prior_close * 0.01
    q90_low = np.percentile(residuals_low, 90) if residuals_low else prior_close * 0.01

    # Iron Condor strikes
    wing = 10.0
    call_strike = predicted_high + q90_high
    put_strike = predicted_low - q90_low

    # Credit
    credit = (vix_close / 100) * prior_close * 0.12

    # P&L
    call_intrusion = max(0, min(actual_high - call_strike, wing))
    put_intrusion = max(0, min(put_strike - actual_low, wing))
    gross_pnl = credit - call_intrusion - put_intrusion
    friction = 0.10 * 4
    net_pnl = gross_pnl - friction

    if regime_val == 2:
        result = "SKIPPED (RED regime)"
        net_pnl_display = 0
    else:
        result = "WIN" if net_pnl > 0 else "LOSS"
        net_pnl_display = net_pnl

    print(f"\n  PREDICTIONS:")
    print(f"  Predicted High: {predicted_high:.2f} (actual: {actual_high:.2f}, error: {actual_high - predicted_high:+.2f})")
    print(f"  Predicted Low:  {predicted_low:.2f} (actual: {actual_low:.2f}, error: {actual_low - predicted_low:+.2f})")

    print(f"\n  IRON CONDOR STRIKES:")
    print(f"  Long Call:  {call_strike + wing:.2f}")
    print(f"  Short Call: {call_strike:.2f}")
    print(f"  Short Put:  {put_strike:.2f}")
    print(f"  Long Put:   {put_strike - wing:.2f}")
    print(f"  Credit:     ${credit:.2f} pts")

    print(f"\n  RESULT: {'🟢' if result == 'WIN' else '🔴' if result == 'LOSS' else '⏭️'} {result}")
    print(f"  Call Intrusion: ${call_intrusion:.2f}")
    print(f"  Put Intrusion:  ${put_intrusion:.2f}")
    print(f"  Gross P&L:      ${gross_pnl:.2f} pts")
    print(f"  Friction:       -${friction:.2f} pts")
    print(f"  Net P&L:        ${net_pnl_display:.2f} pts (${net_pnl_display * 100:.2f} per contract)")

    hi_inside = predicted_high - q90_high <= actual_high <= predicted_high + q90_high
    lo_inside = predicted_low - q90_low <= actual_low <= predicted_low + q90_low
    print(f"\n  90% CONFIDENCE:")
    print(f"  High: {predicted_high - q90_high:.2f} — {predicted_high + q90_high:.2f} (actual {actual_high:.2f} {'✅ INSIDE' if hi_inside else '❌ OUTSIDE'})")
    print(f"  Low:  {predicted_low - q90_low:.2f} — {predicted_low + q90_low:.2f} (actual {actual_low:.2f} {'✅ INSIDE' if lo_inside else '❌ OUTSIDE'})")
    print(f"\n{'='*60}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/backtest_date.py YYYY-MM-DD")
        print("Example: python3 scripts/backtest_date.py 2026-02-01")
        sys.exit(1)
    run_single_date_backtest(sys.argv[1])
