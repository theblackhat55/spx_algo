"""
src/validation/baselines.py
============================
Instruction 4 — Naive baseline models for significance testing.

Three baselines (each produces predictions in the same format as the
regression models, enabling direct MAE / Diebold-Mariano comparison):

    NoChangeBaseline       : tomorrow's high = today's high
                             tomorrow's low  = today's low
    ATRBaseline            : high = close + 0.6 × ATR20
                             low  = close - 0.6 × ATR20
    YesterdayRangeBaseline : centre of yesterday's range, scaled to close

All baselines use only data known at the close of day t-1 (no leakage).

Diebold-Mariano test
--------------------
``diebold_mariano_test`` computes the DM statistic and p-value between
two sets of errors.  Significant p < 0.05 on a two-sided test means the
two models have significantly different loss.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _true_range(df: pd.DataFrame) -> pd.Series:
    high_low  = df["High"] - df["Low"]
    high_prev = (df["High"] - df["Close"].shift(1)).abs()
    low_prev  = (df["Low"]  - df["Close"].shift(1)).abs()
    return pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tr = _true_range(df)
    return tr.rolling(window, min_periods=window // 2).mean()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseBaseline:
    """Minimal interface shared by all naive baselines."""

    name: str = "baseline"

    def predict(
        self,
        spx_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate predictions for every row in spx_df.

        Returns
        -------
        DataFrame with columns ``target_high``, ``target_low``,
        ``target_high_pct``, ``target_low_pct``.
        Index = spx_df.index.
        """
        raise NotImplementedError

    def evaluate(
        self,
        spx_df: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Compute MAE for target_high and target_low against actuals.

        Parameters
        ----------
        targets : DataFrame with actual ``target_high`` / ``target_low`` cols.
        """
        preds  = self.predict(spx_df)
        common = preds.index.intersection(targets.index)
        p = preds.loc[common]
        t = targets.loc[common]

        mae_high = float(np.abs(p["target_high"] - t["target_high"]).mean())
        mae_low  = float(np.abs(p["target_low"]  - t["target_low"]).mean())
        mae_range_pts = float(
            np.abs(
                (p["target_high"] - p["target_low"]) -
                (t["target_high"] - t["target_low"])
            ).mean()
        )
        logger.info("%s — MAE high=%.4f low=%.4f range=%.4f",
                    self.name, mae_high, mae_low, mae_range_pts)
        return {
            "mae_high":       mae_high,
            "mae_low":        mae_low,
            "mae_range_pts":  mae_range_pts,
        }


# ---------------------------------------------------------------------------
# 1. No-Change baseline
# ---------------------------------------------------------------------------

class NoChangeBaseline(BaseBaseline):
    """
    Predicts tomorrow's high/low = today's high/low.
    Implements the random-walk hypothesis for intraday extremes.
    """
    name = "no_change"

    def predict(self, spx_df: pd.DataFrame) -> pd.DataFrame:
        close = spx_df["Close"]
        # Shift(1): use today's values to predict tomorrow
        pred_high = spx_df["High"].shift(1)
        pred_low  = spx_df["Low"].shift(1)

        out = pd.DataFrame(index=spx_df.index)
        out["target_high"]     = pred_high
        out["target_low"]      = pred_low
        out["target_high_pct"] = (pred_high - close) / close.replace(0, np.nan)
        out["target_low_pct"]  = (pred_low  - close) / close.replace(0, np.nan)
        return out.iloc[1:]   # drop first row (no prior day)


# ---------------------------------------------------------------------------
# 2. ATR-based baseline
# ---------------------------------------------------------------------------

class ATRBaseline(BaseBaseline):
    """
    Predicts high = close + k × ATR20,  low = close - k × ATR20.

    ``k`` defaults to 0.6 (empirically ~covers 68% of daily ranges on SPX).
    All inputs are lagged by 1 to prevent leakage.
    """
    name = "atr"

    def __init__(self, k: float = 0.6, window: int = 20):
        self.k      = k
        self.window = window

    def predict(self, spx_df: pd.DataFrame) -> pd.DataFrame:
        atr       = _atr(spx_df, self.window).shift(1)   # yesterday's ATR
        close_lag = spx_df["Close"].shift(1)              # yesterday's close
        today_cls = spx_df["Close"]

        pred_high = close_lag + self.k * atr
        pred_low  = close_lag - self.k * atr

        out = pd.DataFrame(index=spx_df.index)
        out["target_high"]     = pred_high
        out["target_low"]      = pred_low
        out["target_high_pct"] = (pred_high - today_cls) / today_cls.replace(0, np.nan)
        out["target_low_pct"]  = (pred_low  - today_cls) / today_cls.replace(0, np.nan)
        return out.dropna(subset=["target_high"])


# ---------------------------------------------------------------------------
# 3. Yesterday-Range baseline
# ---------------------------------------------------------------------------

class YesterdayRangeBaseline(BaseBaseline):
    """
    Centres yesterday's range on today's open, then scales by:
        range_scale = today's rolling-avg-range / yesterday's range.

    Captures mean-reversion of the daily range.
    """
    name = "yesterday_range"

    def __init__(self, range_avg_window: int = 20):
        self.range_avg_window = range_avg_window

    def predict(self, spx_df: pd.DataFrame) -> pd.DataFrame:
        prev_high  = spx_df["High"].shift(1)
        prev_low   = spx_df["Low"].shift(1)
        prev_range = (prev_high - prev_low).clip(lower=0.01)

        roll_range = (spx_df["High"] - spx_df["Low"]).rolling(
            self.range_avg_window, min_periods=self.range_avg_window // 2
        ).mean().shift(1)

        scale       = (roll_range / prev_range).clip(0.5, 2.0)
        today_open  = spx_df["Open"]
        half_range  = 0.5 * prev_range * scale

        pred_high  = today_open + half_range
        pred_low   = today_open - half_range
        today_cls  = spx_df["Close"]

        out = pd.DataFrame(index=spx_df.index)
        out["target_high"]     = pred_high
        out["target_low"]      = pred_low
        out["target_high_pct"] = (pred_high - today_cls) / today_cls.replace(0, np.nan)
        out["target_low_pct"]  = (pred_low  - today_cls) / today_cls.replace(0, np.nan)
        return out.dropna(subset=["target_high"])


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------

def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
    loss: str = "squared",
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: E[d_t] = 0  where d_t = loss(e_a_t) - loss(e_b_t)

    Parameters
    ----------
    errors_a, errors_b : 1-D arrays of forecast errors.
    h    : Forecast horizon (used for HAC correction).
    loss : 'squared' | 'absolute'

    Returns
    -------
    (dm_statistic, two_sided_p_value)
    """
    if loss == "squared":
        d = errors_a ** 2 - errors_b ** 2
    elif loss == "absolute":
        d = np.abs(errors_a) - np.abs(errors_b)
    else:
        raise ValueError(f"Unknown loss: {loss!r}. Use 'squared' or 'absolute'.")

    n       = len(d)
    d_bar   = d.mean()

    # Newey-West HAC variance
    gamma0  = np.var(d, ddof=1)
    var_d   = gamma0
    for lag in range(1, h):
        gamma = np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
        var_d += 2 * (1 - lag / h) * gamma

    if var_d <= 0:
        logger.warning("DM test: non-positive variance. Returning NaN.")
        return float("nan"), float("nan")

    dm_stat = d_bar / np.sqrt(var_d / n)

    try:
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    except ImportError:
        # Manual normal CDF via error function
        import math
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(dm_stat) / math.sqrt(2))))

    return float(dm_stat), float(p_value)


# ---------------------------------------------------------------------------
# Convenience: compare all baselines
# ---------------------------------------------------------------------------

def compare_baselines(
    spx_df: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run all three baselines and return a comparison DataFrame.

    Parameters
    ----------
    targets : DataFrame with actual 'target_high', 'target_low' columns.
    """
    baselines = [
        NoChangeBaseline(),
        ATRBaseline(),
        YesterdayRangeBaseline(),
    ]
    rows = []
    for bl in baselines:
        metrics = bl.evaluate(spx_df, targets)
        metrics["name"] = bl.name
        rows.append(metrics)

    return pd.DataFrame(rows).set_index("name")
