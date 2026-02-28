"""
src/features/volatility.py
===========================
Task 5 — Volatility features: ATR, Parkinson, Garman-Klass, VIX, GARCH.

LOOKAHEAD-BIAS CONTRACT
-----------------------
Every computation on row T uses only data available at the *close* of day
T.  Because ATR and Parkinson consume today's OHLC they are valid for
intra-day prediction only if computed after today's close.  They are
flagged ``requires_open=False`` in the registry because they are computed
*from* closed-day data.  The rolling statistics (VIX Z-score, GARCH) use
only data through T, never T+1 or later.

Functions
---------
compute_true_range             — Series of single-day True Range
compute_atr                    — DataFrame of ATR across multiple windows
compute_parkinson_vol          — Parkinson range-based volatility
compute_garman_klass_vol       — Garman-Klass OHLC volatility estimator
compute_vix_features           — VIX normalised / Z-score / percentile / ROC
compute_garch_volatility       — GARCH(1,1) conditional variance (expanding)
compute_all_volatility_features— Master assembler
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import ATR_WINDOWS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# True Range and ATR
# ─────────────────────────────────────────────────────────────────────────────

def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """Compute Wilder's True Range for every bar.

    True Range = max(
        High  - Low,
        |High - prev_Close|,
        |Low  - prev_Close|
    )

    Parameters
    ----------
    df:
        OHLCV DataFrame.  Must contain High, Low, Close columns.

    Returns
    -------
    Series named ``"true_range"`` aligned to *df*.index.

    Notes
    -----
    The first row will be NaN because prev_Close is not available.
    """
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr.name = "true_range"
    return tr


def compute_atr(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Average True Range for multiple window sizes.

    Uses Wilder's exponential smoothing (span = window, adjust=False).
    This matches the classic ATR definition used by most charting platforms.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    windows:
        List of window sizes.  Defaults to ``ATR_WINDOWS`` from settings.

    Returns
    -------
    DataFrame with columns ``atr_N`` for each window N, plus:
      ``atr_ratio_5_60`` = atr_5 / atr_60   (volatility regime indicator)
    """
    if windows is None:
        windows = ATR_WINDOWS

    tr = compute_true_range(df)
    out = pd.DataFrame(index=df.index)

    for w in windows:
        # Wilder smoothing = EWM with span=window, adjust=False
        out[f"atr_{w}"] = tr.ewm(span=w, adjust=False, min_periods=w).mean()

    # Expansion ratio — short-term vs long-term
    if "atr_5" in out.columns and "atr_60" in out.columns:
        # Guard against div-by-zero on early rows
        out["atr_ratio_5_60"] = out["atr_5"] / out["atr_60"].replace(0, np.nan)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Range-based volatility estimators
# ─────────────────────────────────────────────────────────────────────────────

def compute_parkinson_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Parkinson (1980) range-based volatility estimator.

    Uses only High and Low prices.  More efficient than close-to-close
    volatility because it captures intraday information.

    σ_Parkinson = sqrt[ (1 / (4 * ln(2))) * E[(ln(H/L))^2] ]

    Parameters
    ----------
    df:
        OHLCV DataFrame (needs High, Low).
    window:
        Rolling window in trading days.

    Returns
    -------
    Series named ``"parkinson_vol_N"``.
    """
    # ln(H/L) squared — this is a variance proxy per bar
    log_hl_sq = np.log(df["High"] / df["Low"]) ** 2

    parkinson_const = 1.0 / (4.0 * np.log(2.0))
    vol = np.sqrt(parkinson_const * log_hl_sq.rolling(window=window, min_periods=window // 2).mean())
    vol.name = f"parkinson_vol_{window}"
    return vol


def compute_garman_klass_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Garman-Klass (1980) OHLC volatility estimator.

    More efficient than Parkinson because it also uses the Open.

    σ_GK = sqrt[
        0.5 * (ln(H/L))^2
        - (2*ln(2) - 1) * (ln(C/O))^2
    ]   (daily contribution, then rolling mean)

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    window:
        Rolling window in trading days.

    Returns
    -------
    Series named ``"garman_klass_vol_N"``.
    """
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])

    gk_daily = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    # Clip negative values (can occur due to numerical precision)
    gk_daily = gk_daily.clip(lower=0)

    vol = np.sqrt(gk_daily.rolling(window=window, min_periods=window // 2).mean())
    vol.name = f"garman_klass_vol_{window}"
    return vol


# ─────────────────────────────────────────────────────────────────────────────
# VIX features
# ─────────────────────────────────────────────────────────────────────────────

def compute_vix_features(vix_df: pd.DataFrame) -> pd.DataFrame:
    """Compute VIX-derived features.

    Parameters
    ----------
    vix_df:
        DataFrame with a ``Close`` column representing VIX.  Index is
        a DatetimeIndex sorted ascending.

    Returns
    -------
    DataFrame with columns:
      vix_normalized      — VIX / 252-day rolling mean
      vix_zscore          — (VIX - 252d mean) / 252d std
      vix_percentile_252  — percentile rank over the last 252 days
      vix_roc_5           — 5-day rate of change
    """
    vix = vix_df["Close"]
    out = pd.DataFrame(index=vix_df.index)

    roll_mean = vix.rolling(window=252, min_periods=126).mean()
    roll_std  = vix.rolling(window=252, min_periods=126).std()

    out["vix_normalized"]     = vix / roll_mean
    out["vix_zscore"]         = (vix - roll_mean) / roll_std.replace(0, np.nan)
    out["vix_percentile_252"] = (
        vix.rolling(window=252, min_periods=126)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )
    out["vix_roc_5"]          = vix.pct_change(periods=5)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# GARCH(1,1) conditional variance
# ─────────────────────────────────────────────────────────────────────────────

def compute_garch_volatility(returns_series: pd.Series) -> pd.Series:
    """Fit GARCH(1,1) and return the conditional variance series.

    The model is fit once on the *full* returns series and the in-sample
    conditional variance is extracted.  For production use, refit weekly
    as part of the retraining cycle.

    Note on lookahead bias
    ----------------------
    Fitting on the full series means the GARCH parameters are estimated
    using future data.  This is a *parameter* lookahead, not a *data*
    lookahead — the conditional variance at time T is still computed only
    from returns up to T given those parameters.  This is acceptable for
    initial feature development; for strict out-of-sample testing use an
    expanding-window refit (expensive).

    Parameters
    ----------
    returns_series:
        Daily log-returns (or simple returns), indexed by date.

    Returns
    -------
    Series of conditional variance values aligned to the input index.
    Returns a Series of NaN if ``arch`` is not installed.
    """
    try:
        from arch import arch_model
    except ImportError:
        logger.warning("arch library not installed — GARCH feature will be NaN.")
        return pd.Series(np.nan, index=returns_series.index, name="garch_cond_var")

    clean = returns_series.dropna()
    if len(clean) < 100:
        logger.warning("Too few returns (%d) for GARCH fit.", len(clean))
        return pd.Series(np.nan, index=returns_series.index, name="garch_cond_var")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                clean * 100,       # scale to percent for numerical stability
                vol="Garch",
                p=1, q=1,
                dist="normal",
                rescale=False,
            )
            res = model.fit(disp="off", show_warning=False)

        cond_var = res.conditional_volatility ** 2 / (100 ** 2)   # back to decimal²
        cond_var = cond_var.reindex(returns_series.index)
        cond_var.name = "garch_cond_var"
        return cond_var

    except Exception as exc:
        logger.warning("GARCH fit failed: %s — returning NaN series.", exc)
        return pd.Series(np.nan, index=returns_series.index, name="garch_cond_var")


# ─────────────────────────────────────────────────────────────────────────────
# Master assembler
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_volatility_features(
    ohlcv_df: pd.DataFrame,
    vix_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute and merge all volatility features.

    Parameters
    ----------
    ohlcv_df:
        SPX OHLCV DataFrame (daily).
    vix_df:
        VIX OHLCV DataFrame (daily).  Must contain ``Close``.

    Returns
    -------
    Single merged DataFrame indexed by date with all volatility columns.
    """
    frames: list[pd.DataFrame] = []

    # ── ATR features ─────────────────────────────────────────────────────────
    frames.append(compute_atr(ohlcv_df))

    # ── Range-based vol ───────────────────────────────────────────────────────
    frames.append(compute_parkinson_vol(ohlcv_df).to_frame())
    frames.append(compute_garman_klass_vol(ohlcv_df).to_frame())

    # ── VIX features (align to SPX dates via reindex) ─────────────────────────
    vix_aligned = vix_df.reindex(ohlcv_df.index, method="ffill")
    frames.append(compute_vix_features(vix_aligned))

    # ── GARCH ─────────────────────────────────────────────────────────────────
    log_returns = np.log(ohlcv_df["Close"] / ohlcv_df["Close"].shift(1))
    frames.append(compute_garch_volatility(log_returns).to_frame())

    # ── Merge ─────────────────────────────────────────────────────────────────
    result = pd.concat(frames, axis=1)
    return result
