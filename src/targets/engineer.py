"""
src/targets/engineer.py
========================
Task 12 — Target variable engineering.

For each trading day t the model is trained on features computed from
data known at the *close* of day t-1 and predicts properties of day t.

Targets produced
----------------
High targets
    next_high_pct       : (next_high - today_close) / today_close
    next_high_bin_50    : 1 if next_high_pct > 0.50 %
    next_high_bin_100   : 1 if next_high_pct > 1.00 %
    next_high_bin_150   : 1 if next_high_pct > 1.50 %
    next_high_pct_q33   : 1 if next_high_pct in top 2/3 of its trailing
                           252-day distribution (multi-class: 0/1/2)

Low targets
    next_low_pct        : (next_low  - today_close) / today_close  (negative)
    next_low_bin_50     : 1 if next_low_pct  < -0.50 %
    next_low_bin_100    : 1 if next_low_pct  < -1.00 %
    next_low_bin_150    : 1 if next_low_pct  < -1.50 %

Range target
    next_range_pct      : (next_high - next_low) / today_close
    next_range_bin_med  : 1 if next_range_pct > rolling median(252d)

All targets are shifted so that row t carries *tomorrow's* outcome.
Features must not use any data from day t+1.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import PROCESSED_DATA_DIR, SPX_FILE, RAW_DATA_DIR

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────
HIGH_THRESHOLDS = [0.005, 0.010, 0.015]   # 0.50 %, 1.00 %, 1.50 %
LOW_THRESHOLDS  = [-0.005, -0.010, -0.015]
ROLLING_WINDOW  = 252


# ── helpers ──────────────────────────────────────────────────────────────────

def _pct_move(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safe percentage: (num - denom) / denom, NaN where denom == 0."""
    denom = denominator.replace(0, np.nan)
    return (numerator - denom) / denom


def _rolling_quantile_class(
    series: pd.Series,
    window: int = ROLLING_WINDOW,
    n_classes: int = 3,
) -> pd.Series:
    """
    Assign each value to a quantile class 0..n_classes-1 based on its
    position in the trailing ``window``-day distribution.
    Uses min_periods = window // 2.
    """
    min_p = window // 2

    def _classify(arr: np.ndarray) -> int:
        if len(arr) < 2:
            return np.nan
        val = arr[-1]
        hist = arr[:-1]
        q = np.nanpercentile(hist, np.linspace(0, 100, n_classes + 1)[1:-1])
        return int(np.searchsorted(q, val))

    return series.rolling(window, min_periods=min_p).apply(
        _classify, raw=True
    )


# ── main engineering function ─────────────────────────────────────────────────

def engineer_targets(
    spx_path: Optional[Path] = None,
    save: bool = True,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load raw SPX OHLCV, compute all target columns, optionally save.

    Parameters
    ----------
    spx_path    : Path to raw SPX parquet.  Defaults to settings path.
    save        : Whether to write targets.parquet.
    output_path : Override output path (default: PROCESSED_DATA_DIR).

    Returns
    -------
    DataFrame indexed by date with target columns only (no features).
    """
    # ── load ─────────────────────────────────────────────────────────────────
    spx_path = spx_path or (RAW_DATA_DIR / SPX_FILE)
    if not Path(spx_path).exists():
        raise FileNotFoundError(f"SPX file not found: {spx_path}")

    df = pd.read_parquet(spx_path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    required = {"Open", "High", "Low", "Close"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"SPX data missing columns: {missing}")

    logger.info("Loaded SPX data: %d rows (%s – %s)",
                len(df), df.index[0].date(), df.index[-1].date())

    tgt = pd.DataFrame(index=df.index)

    # ── continuous targets ────────────────────────────────────────────────────
    # shift(-1): row t gets the NEXT day's high/low
    next_high  = df["High"].shift(-1)
    next_low   = df["Low"].shift(-1)
    today_cls  = df["Close"]

    tgt["next_high_pct"]  = _pct_move(next_high, today_cls)
    tgt["next_low_pct"]   = _pct_move(next_low,  today_cls)
    tgt["next_range_pct"] = _pct_move(next_high - next_low + today_cls,
                                      today_cls) - 1.0
    # cleaner range definition
    tgt["next_range_pct"] = (next_high - next_low) / today_cls.replace(0, np.nan)

    # ── REGRESSION targets (absolute price levels) ────────────────────────────
    # These are the primary targets for regression models.
    # target_high / target_low = actual next-day high/low price.
    # Regressor predictions → specific SPX levels for iron condor placement.
    tgt["target_high"] = next_high          # absolute price, e.g. 5942.10
    tgt["target_low"]  = next_low           # absolute price, e.g. 5901.45
    # Convenience: deviation in points (not %) from prior close
    tgt["target_high_pts"] = next_high - today_cls
    tgt["target_low_pts"]  = next_low  - today_cls  # negative value

    # ── binary high targets ───────────────────────────────────────────────────
    for thr in HIGH_THRESHOLDS:
        label = f"next_high_bin_{int(round(thr*10000)):03d}"   # 050, 100, 150
        tgt[label] = (tgt["next_high_pct"] > thr).astype("Int8")

    # ── binary low targets ────────────────────────────────────────────────────
    for thr in LOW_THRESHOLDS:
        label = f"next_low_bin_{int(round(abs(thr)*10000)):03d}"
        tgt[label] = (tgt["next_low_pct"] < thr).astype("Int8")

    # ── range binary ─────────────────────────────────────────────────────────
    rolling_med = tgt["next_range_pct"].rolling(ROLLING_WINDOW,
                                                min_periods=ROLLING_WINDOW // 2).median()
    tgt["next_range_bin_med"] = (tgt["next_range_pct"] > rolling_med).astype("Int8")

    # ── quantile class for high (3-class) ────────────────────────────────────
    tgt["next_high_q3"] = _rolling_quantile_class(
        tgt["next_high_pct"], window=ROLLING_WINDOW, n_classes=3
    ).astype("Int8")

    # ── drop the last row (has NaN targets from shift(-1)) ───────────────────
    tgt = tgt.iloc[:-1]

    logger.info("Engineered %d target rows, %d columns: %s",
                len(tgt), len(tgt.columns), list(tgt.columns))

    # ── save ──────────────────────────────────────────────────────────────────
    if save:
        out_dir  = output_path or PROCESSED_DATA_DIR
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_file = Path(out_dir) / "targets.parquet"
        tgt.to_parquet(out_file)
        logger.info("Saved targets → %s", out_file)

    return tgt


# ── convenience loader ────────────────────────────────────────────────────────

def load_targets(path: Optional[Path] = None) -> pd.DataFrame:
    """Load previously saved targets.parquet."""
    path = path or (PROCESSED_DATA_DIR / "targets.parquet")
    if not Path(path).exists():
        raise FileNotFoundError(f"Targets file not found: {path}")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


# ── alignment helper ──────────────────────────────────────────────────────────

def align_features_targets(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    target_cols: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inner-join features and targets on date index, return (X, y).

    Parameters
    ----------
    features    : Feature DataFrame (output of build_feature_matrix).
    targets     : Target DataFrame (output of engineer_targets).
    target_cols : Subset of target columns to include.  None = all.

    Returns
    -------
    X : features aligned to common dates
    y : targets aligned to common dates
    """
    if target_cols is not None:
        targets = targets[target_cols]

    common = features.index.intersection(targets.index)
    if len(common) == 0:
        raise ValueError("No overlapping dates between features and targets.")

    X = features.loc[common]
    y = targets.loc[common]

    # Sanity: no NaN in targets
    nan_rows = y.isna().any(axis=1).sum()
    if nan_rows > 0:
        logger.warning("%d rows have NaN targets — dropping.", nan_rows)
        mask = ~y.isna().any(axis=1)
        X, y = X[mask], y[mask]

    logger.info("Aligned dataset: %d rows, %d features, %d targets",
                len(X), X.shape[1], y.shape[1])
    return X, y
