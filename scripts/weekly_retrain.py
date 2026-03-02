#!/usr/bin/env python3
"""
Weekly model retraining with auto-promotion.
Retrains LightGBM models on all available data, backtests new vs old,
promotes new model only if it's demonstrably better.

Usage: python3 scripts/weekly_retrain.py
"""
import sys
import os
import json
import shutil
import warnings
import logging
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, '/root/spx_algo')
os.chdir('/root/spx_algo')

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = Path('output/models')
ARCHIVE_DIR = MODEL_DIR / 'archive'
REPORT_DIR = Path('output/reports')
HIGH_MODEL = 'regressor_target_high_pct.pkl'
LOW_MODEL = 'regressor_target_low_pct.pkl'
EVAL_DAYS = 30  # backtest comparison window


def archive_current_models():
    """Save current models to archive with timestamp."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    for name in [HIGH_MODEL, LOW_MODEL]:
        src = MODEL_DIR / name
        if src.exists():
            dst = ARCHIVE_DIR / f'{ts}_{name}'
            shutil.copy2(src, dst)
            logger.info(f'Archived: {src} -> {dst}')
    # Keep only last 4 versions (8 files = 4 pairs)
    archives = sorted(ARCHIVE_DIR.glob('*.pkl'))
    while len(archives) > 8:
        oldest = archives.pop(0)
        oldest.unlink()
        logger.info(f'Pruned old archive: {oldest}')
    return ts


def load_model(path):
    return joblib.load(path)


def backtest_models(high_model, low_model, features_df, spx_df, vix_df, last_n_days):
    """Run backtest on last N trading days, return metrics dict."""
    from src.calibration.regime import RegimeDetector, Regime

    # Get feature columns
    feature_cols = [c for c in features_df.columns
                    if not c.startswith('target_') and not c.startswith('next_')]

    # Get last N trading days that exist in both features and spx
    common = features_df.index.intersection(spx_df.index)
    eval_dates = common[-last_n_days:]

    detector = RegimeDetector()
    regime_series = detector.fit_predict(spx_df, vix_df)

    wins = 0
    losses = 0
    skips = 0
    total_pnl = 0.0
    coverage_90_high = 0
    coverage_90_low = 0
    pnl_list = []

    for i, date in enumerate(eval_dates):
        idx = spx_df.index.get_loc(date)
        if idx < 1:
            continue

        prior_close = float(spx_df.iloc[idx - 1]['Close'])
        actual_high = float(spx_df.loc[date, 'High'])
        actual_low = float(spx_df.loc[date, 'Low'])
        vix_close = float(vix_df.loc[date, 'Close']) if date in vix_df.index else 20.0
        regime_val = int(regime_series.get(date, Regime.YELLOW))

        # Skip RED
        if regime_val == 2:
            skips += 1
            continue

        X = features_df.loc[[date], feature_cols]
        try:
            pred_high_pct = float(high_model.predict(X)[0])
            pred_low_pct = float(low_model.predict(X)[0])
        except Exception:
            skips += 1
            continue

        predicted_high = prior_close * (1 + pred_high_pct / 100)
        predicted_low = prior_close * (1 + pred_low_pct / 100)

        # Simple conformal (use last 63 days before this date)
        cal_start = max(0, spx_df.index.get_loc(date) - 63)
        cal_end = spx_df.index.get_loc(date)
        cal_dates_range = features_df.index[cal_start:cal_end]
        res_high = []
        res_low = []
        for cd in cal_dates_range:
            ci = spx_df.index.get_loc(cd)
            if ci > 0:
                pc = float(spx_df.iloc[ci - 1]['Close'])
                X_cal = features_df.loc[[cd], feature_cols]
                try:
                    ph = float(high_model.predict(X_cal)[0])
                    pl = float(low_model.predict(X_cal)[0])
                    res_high.append(abs(float(spx_df.loc[cd, 'High']) - pc * (1 + ph / 100)))
                    res_low.append(abs(float(spx_df.loc[cd, 'Low']) - pc * (1 + pl / 100)))
                except:
                    pass

        q90_high = np.percentile(res_high, 90) if res_high else prior_close * 0.01
        q90_low = np.percentile(res_low, 90) if res_low else prior_close * 0.01

        call_strike = predicted_high + q90_high
        put_strike = predicted_low - q90_low
        wing = 10.0

        # Coverage check
        if predicted_high - q90_high <= actual_high <= predicted_high + q90_high:
            coverage_90_high += 1
        if predicted_low - q90_low <= actual_low <= predicted_low + q90_low:
            coverage_90_low += 1

        # P&L
        credit = (vix_close / 100) * prior_close * 0.12
        call_intrusion = max(0, min(actual_high - call_strike, wing))
        put_intrusion = max(0, min(put_strike - actual_low, wing))
        net_pnl = credit - call_intrusion - put_intrusion - 0.40

        if net_pnl > 0:
            wins += 1
        else:
            losses += 1
        total_pnl += net_pnl
        pnl_list.append(net_pnl)

    active = wins + losses
    win_rate = (wins / active * 100) if active > 0 else 0
    avg_pnl = (total_pnl / active) if active > 0 else 0
    max_dd = 0
    running = 0
    for p in pnl_list:
        running += p
        if running < max_dd:
            max_dd = running
    sharpe = (np.mean(pnl_list) / np.std(pnl_list) * np.sqrt(252)) if pnl_list and np.std(pnl_list) > 0 else 0
    cov_high = (coverage_90_high / active * 100) if active > 0 else 0
    cov_low = (coverage_90_low / active * 100) if active > 0 else 0

    return {
        'active_trades': active,
        'wins': wins,
        'losses': losses,
        'skips': skips,
        'win_rate': round(win_rate, 1),
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(avg_pnl, 2),
        'max_drawdown': round(max_dd, 2),
        'sharpe': round(sharpe, 2),
        'coverage_90_high': round(cov_high, 1),
        'coverage_90_low': round(cov_low, 1),
    }


def retrain_models():
    """Download fresh data, rebuild features, run walk-forward training."""
    import subprocess

    # Step 1: Download fresh SPX and VIX data
    logger.info('Downloading fresh market data...')
    try:
        from src.data.fetcher import fetch_all_yahoo_data
        fetch_all_yahoo_data()
        logger.info('Fresh data downloaded')
    except Exception as exc:
        logger.warning(f'fetch_all_yahoo_data failed ({exc}), trying yfinance directly...')
        try:
            import yfinance as yf
            import pandas as pd
            for ticker, name in [('^GSPC', 'spx_daily'), ('^VIX', 'vix_daily')]:
                df = yf.download(ticker, start='2000-01-01', auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                df.to_parquet(f'data/raw/{name}.parquet')
                logger.info(f'{name}: {len(df)} rows through {df.index[-1].date()}')
        except Exception as exc2:
            logger.error(f'Data download failed: {exc2}')
            return False

    # Step 2: Rebuild feature matrix with new data
    logger.info('Rebuilding feature matrix...')
    try:
        from src.features.builder import build_feature_matrix
        features = build_feature_matrix()
        logger.info(f'Feature matrix rebuilt: {features.shape[0]} rows x {features.shape[1]} cols')
    except Exception as exc:
        logger.error(f'Feature rebuild failed: {exc}')
        return False

    # Step 3: Run walk-forward training
    logger.info('Running walk-forward training...')
    result = subprocess.run(
        [sys.executable, 'scripts/retrain_full_stack.py'],
        capture_output=True, text=True, cwd='/root/spx_algo'
    )

    if result.stdout:
        for line in result.stdout.strip().split(chr(10))[-10:]:
            logger.info(f'trainer: {line}')

    if result.returncode != 0:
        logger.error(f'Training failed: {result.stderr[-500:]}')
        return False

    logger.info('Retraining complete with fresh data')
    return True

def compare_and_promote(old_metrics, new_metrics):
    """Compare old vs new model metrics. Return (promote: bool, reason: str)."""
    checks = {
        'win_rate': new_metrics['win_rate'] >= old_metrics['win_rate'],
        'total_pnl': new_metrics['total_pnl'] >= old_metrics['total_pnl'],
        'coverage_high': new_metrics['coverage_90_high'] >= old_metrics['coverage_90_high'],
        'sharpe': new_metrics['sharpe'] >= old_metrics['sharpe'],
    }

    wins = sum(checks.values())
    total = len(checks)

    # Safety rails
    if new_metrics['win_rate'] < 60:
        return False, f"BLOCKED: New win rate {new_metrics['win_rate']}% < 60% minimum"

    # Drawdown check (only block if old model had meaningful drawdown)
    old_dd = abs(old_metrics['max_drawdown'])
    new_dd = abs(new_metrics['max_drawdown'])
    if old_dd > 0 and new_dd > old_dd * 1.2:
        return False, f"BLOCKED: New max drawdown ${new_metrics['max_drawdown']} > 120% of old ${old_metrics['max_drawdown']}"

    # Need at least 3 of 4 metrics better
    if wins >= 3:
        details = ', '.join(f'{k}:{"✅" if v else "❌"}' for k, v in checks.items())
        return True, f"PROMOTED ({wins}/{total} better): {details}"

    # Too close to call — within 2%
    close_calls = 0
    if abs(new_metrics['win_rate'] - old_metrics['win_rate']) <= 2:
        close_calls += 1
    if abs(new_metrics['total_pnl'] - old_metrics['total_pnl']) <= old_metrics.get('avg_pnl', 1) * 0.5:
        close_calls += 1

    if close_calls >= 2:
        details = ', '.join(f'{k}:{"✅" if v else "❌"}' for k, v in checks.items())
        return False, f"TOO CLOSE TO CALL ({wins}/{total} better, {close_calls} within margin): {details}. Manual review recommended."

    details = ', '.join(f'{k}:{"✅" if v else "❌"}' for k, v in checks.items())
    return False, f"NOT PROMOTED ({wins}/{total} better): {details}"


def main():
    start_time = datetime.now(timezone.utc)
    report = {
        'timestamp': start_time.isoformat(),
        'status': 'running',
    }

    try:
        # Load data
        logger.info('Loading data...')
        spx_df = pd.read_parquet('data/raw/spx_daily.parquet')
        vix_df = pd.read_parquet('data/raw/vix_daily.parquet')
        from src.features.builder import load_feature_matrix
        features_df = load_feature_matrix()

        # Step 1: Backtest CURRENT models
        logger.info(f'Backtesting current models on last {EVAL_DAYS} days...')
        old_high = load_model(MODEL_DIR / HIGH_MODEL)
        old_low = load_model(MODEL_DIR / LOW_MODEL)
        old_metrics = backtest_models(old_high, old_low, features_df, spx_df, vix_df, EVAL_DAYS)
        logger.info(f'Current model: {old_metrics}')
        report['old_metrics'] = old_metrics

        # Step 2: Archive current models
        archive_ts = archive_current_models()
        report['archive_timestamp'] = archive_ts

        # Step 3: Retrain
        logger.info('Retraining models...')
        success = retrain_models()
        if not success:
            # Restore archived models
            for name in [HIGH_MODEL, LOW_MODEL]:
                archived = ARCHIVE_DIR / f'{archive_ts}_{name}'
                if archived.exists():
                    shutil.copy2(archived, MODEL_DIR / name)
            report['status'] = 'retrain_failed'
            report['decision'] = 'KEPT OLD MODELS (retraining failed)'
            _save_report(report)
            _print_summary(report)
            return

        # Step 4: Rebuild features with new data
        logger.info('Rebuilding features...')
        from src.features.builder import build_feature_matrix
        features_df = build_feature_matrix()

        # Step 5: Backtest NEW models
        logger.info(f'Backtesting new models on last {EVAL_DAYS} days...')
        new_high = load_model(MODEL_DIR / HIGH_MODEL)
        new_low = load_model(MODEL_DIR / LOW_MODEL)
        new_metrics = backtest_models(new_high, new_low, features_df, spx_df, vix_df, EVAL_DAYS)
        logger.info(f'New model: {new_metrics}')
        report['new_metrics'] = new_metrics

        # Step 6: Compare and decide
        promote, reason = compare_and_promote(old_metrics, new_metrics)
        report['promote'] = promote
        report['decision'] = reason

        if not promote:
            # Restore old models
            logger.info('Restoring old models...')
            for name in [HIGH_MODEL, LOW_MODEL]:
                archived = ARCHIVE_DIR / f'{archive_ts}_{name}'
                if archived.exists():
                    shutil.copy2(archived, MODEL_DIR / name)
            logger.info('Old models restored')

        report['status'] = 'complete'
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        report['duration_seconds'] = round(duration, 1)

    except Exception as e:
        report['status'] = 'error'
        report['error'] = str(e)
        logger.exception('Weekly retrain failed')

    _save_report(report)
    _print_summary(report)


def _save_report(report):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / 'weekly_retrain_latest.json'
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    # Also save timestamped copy
    ts = datetime.now().strftime('%Y%m%d')
    ts_path = REPORT_DIR / f'weekly_retrain_{ts}.json'
    with open(ts_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f'Report saved: {path}')


def _print_summary(report):
    """Print WhatsApp-friendly summary."""
    print(f"\n{'='*50}")
    print(f"  WEEKLY MODEL RETRAIN REPORT")
    print(f"  {report.get('timestamp', '')[:19]}")
    print(f"{'='*50}")
    print(f"  Status: {report.get('status', 'unknown').upper()}")

    if 'old_metrics' in report:
        om = report['old_metrics']
        print(f"\n  CURRENT MODEL (last {EVAL_DAYS} days):")
        print(f"  Win Rate: {om['win_rate']}% | Trades: {om['active_trades']}")
        print(f"  Total P&L: ${om['total_pnl']} | Avg: ${om['avg_pnl']}")
        print(f"  Sharpe: {om['sharpe']} | Max DD: ${om['max_drawdown']}")
        print(f"  Coverage 90%: High {om['coverage_90_high']}% | Low {om['coverage_90_low']}%")

    if 'new_metrics' in report:
        nm = report['new_metrics']
        print(f"\n  RETRAINED MODEL (last {EVAL_DAYS} days):")
        print(f"  Win Rate: {nm['win_rate']}% | Trades: {nm['active_trades']}")
        print(f"  Total P&L: ${nm['total_pnl']} | Avg: ${nm['avg_pnl']}")
        print(f"  Sharpe: {nm['sharpe']} | Max DD: ${nm['max_drawdown']}")
        print(f"  Coverage 90%: High {nm['coverage_90_high']}% | Low {nm['coverage_90_low']}%")

    print(f"\n  DECISION: {report.get('decision', 'N/A')}")
    if report.get('duration_seconds'):
        print(f"  Duration: {report['duration_seconds']}s")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
