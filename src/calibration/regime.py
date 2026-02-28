"""
src/calibration/regime.py
==========================
Instruction 2 — Market Regime Detector.

Produces a GREEN / YELLOW / RED signal for each trading day:

    GREEN  — normal, high-quality prediction environment
    YELLOW — elevated uncertainty; reduce position size
    RED    — anomalous regime; skip trading entirely

Signal construction
-------------------
Three independent signals are combined:

1. VIX Z-score (rolling 252-day):
       < +1.5  → GREEN
       +1.5–3  → YELLOW
       > +3    → RED

2. ATR Expansion Ratio (ATR-5 / ATR-60):
       < 1.5   → GREEN
       1.5–2.5 → YELLOW
       > 2.5   → RED

3. Gaussian HMM state (4 states fitted on daily log-returns + range-pct):
       Volatility-ranked: lowest-vol state = GREEN,
                          mid states       = YELLOW,
                          highest-vol      = RED

Final rule: worst component wins.
    any RED    → RED
    any YELLOW → YELLOW
    all GREEN  → GREEN

Dependencies
------------
hmmlearn  : pip install hmmlearn   (optional — falls back gracefully)
arch      : pip install arch       (for GJR-GARCH, optional)
"""
from __future__ import annotations

import logging
import warnings
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

class Regime(IntEnum):
    GREEN  = 0
    YELLOW = 1
    RED    = 2

    @classmethod
    def from_str(cls, s: str) -> "Regime":
        return cls[s.upper()]


REGIME_LABELS = {Regime.GREEN: "GREEN", Regime.YELLOW: "YELLOW", Regime.RED: "RED"}


# ---------------------------------------------------------------------------
# VIX-based signal
# ---------------------------------------------------------------------------

def vix_regime(
    vix_close: pd.Series,
    window: int = 252,
    yellow_z: float = 1.5,
    red_z: float = 3.0,
) -> pd.Series:
    """
    Rolling Z-score of VIX.  Returns Series of Regime ints indexed by date.
    """
    roll_mean = vix_close.rolling(window, min_periods=window // 2).mean()
    roll_std  = vix_close.rolling(window, min_periods=window // 2).std()
    z = (vix_close - roll_mean) / roll_std.replace(0, np.nan)

    regime = pd.Series(Regime.GREEN, index=vix_close.index, dtype=int)
    regime[z >= yellow_z] = Regime.YELLOW
    regime[z >= red_z]    = Regime.RED
    regime = regime.fillna(Regime.YELLOW)
    return regime


# ---------------------------------------------------------------------------
# ATR-expansion signal
# ---------------------------------------------------------------------------

def _true_range(df: pd.DataFrame) -> pd.Series:
    high_low  = df["High"] - df["Low"]
    high_prev = (df["High"] - df["Close"].shift(1)).abs()
    low_prev  = (df["Low"]  - df["Close"].shift(1)).abs()
    return pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)


def atr_expansion_regime(
    df: pd.DataFrame,
    fast_window: int = 5,
    slow_window: int = 60,
    yellow_ratio: float = 1.5,
    red_ratio: float = 2.5,
) -> pd.Series:
    """
    Compares short-term ATR to long-term ATR.
    Elevated ratio → higher uncertainty → worse regime.
    """
    tr      = _true_range(df)
    atr_f   = tr.rolling(fast_window,  min_periods=fast_window // 2).mean()
    atr_s   = tr.rolling(slow_window,  min_periods=slow_window  // 2).mean()
    ratio   = atr_f / atr_s.replace(0, np.nan)

    regime = pd.Series(Regime.GREEN, index=df.index, dtype=int)
    regime[ratio >= yellow_ratio] = Regime.YELLOW
    regime[ratio >= red_ratio]    = Regime.RED
    regime = regime.fillna(Regime.YELLOW)
    return regime


# ---------------------------------------------------------------------------
# HMM-based signal
# ---------------------------------------------------------------------------

def hmm_regime(
    returns: pd.Series,
    range_pct: pd.Series,
    n_states: int = 4,
    random_state: int = 42,
) -> pd.Series:
    """
    Fit a Gaussian HMM on (log-return, range-pct) 2-D observations.
    States are ranked by volatility; highest-vol = RED, lowest = GREEN.

    Falls back to YELLOW if hmmlearn is not installed.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        logger.warning("hmmlearn not installed. HMM regime defaulting to YELLOW.")
        return pd.Series(Regime.YELLOW, index=returns.index, dtype=int)

    log_ret = np.log1p(returns.fillna(0).values)
    rng_pct = range_pct.fillna(range_pct.mean()).values
    obs     = np.column_stack([log_ret, rng_pct])

    # Drop NaN rows for fitting but keep index alignment
    valid   = np.isfinite(obs).all(axis=1)
    obs_fit = obs[valid]

    # Try "full" covariance; fall back to "diag" if not positive-definite
    fitted_model = None
    for cov_type in ("full", "diag", "spherical"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidate = GaussianHMM(
                    n_components=n_states,
                    covariance_type=cov_type,
                    n_iter=200,
                    random_state=random_state,
                )
                candidate.fit(obs_fit)
                # Smoke-test: predict must not raise
                candidate.predict(obs_fit[:min(10, len(obs_fit))])
                fitted_model = candidate
                break
        except Exception:
            continue

    if fitted_model is None:
        logger.warning("HMM failed all covariance types — defaulting to YELLOW")
        return pd.Series(Regime.YELLOW, index=returns.index, dtype=int)

    model = fitted_model

    # Predict states for all rows
    states_arr = np.full(len(obs), -1, dtype=int)
    try:
        states_arr[valid] = model.predict(obs_fit)
    except Exception:
        logger.warning("HMM predict failed — defaulting to YELLOW")
        return pd.Series(Regime.YELLOW, index=returns.index, dtype=int)

    # Rank states by variance of log-return (proxy for vol)
    state_vols = {}
    for s in range(n_states):
        mask = states_arr == s
        if mask.sum() > 1:
            state_vols[s] = log_ret[mask].std()
        else:
            state_vols[s] = 0.0

    sorted_states = sorted(state_vols, key=lambda s: state_vols[s])

    # Map: lowest-vol → GREEN, mid → YELLOW, highest-vol → RED
    n = len(sorted_states)
    state_to_regime: dict[int, int] = {}
    for rank, s in enumerate(sorted_states):
        if rank == 0:
            state_to_regime[s] = Regime.GREEN
        elif rank == n - 1:
            state_to_regime[s] = Regime.RED
        else:
            state_to_regime[s] = Regime.YELLOW

    regime_arr = np.array([
        state_to_regime.get(s, Regime.YELLOW) for s in states_arr
    ])
    # Rows where HMM had no valid prediction → YELLOW
    regime_arr[states_arr == -1] = Regime.YELLOW

    return pd.Series(regime_arr, index=returns.index, dtype=int)


# ---------------------------------------------------------------------------
# GJR-GARCH conditional variance (supplementary signal)
# ---------------------------------------------------------------------------

def gjr_garch_signal(
    returns: pd.Series,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    red_percentile: float = 95.0,
    yellow_percentile: float = 75.0,
) -> pd.Series:
    """
    Fit GJR-GARCH(p,o,q) on the return series.
    Conditional variance percentile → regime signal.

    Falls back to YELLOW if arch is not installed.
    """
    try:
        from arch import arch_model
    except ImportError:
        logger.warning("arch not installed. GARCH regime defaulting to YELLOW.")
        return pd.Series(Regime.YELLOW, index=returns.index, dtype=int)

    r = returns.dropna() * 100   # GARCH works better on % returns

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am  = arch_model(r, p=p, o=o, q=q, vol="GARCH", dist="Normal")
        res = am.fit(disp="off", show_warning=False)

    cond_var = res.conditional_volatility ** 2
    cond_var = cond_var.reindex(returns.index)

    red_thr    = np.nanpercentile(cond_var.values, red_percentile)
    yellow_thr = np.nanpercentile(cond_var.values, yellow_percentile)

    regime = pd.Series(Regime.GREEN, index=returns.index, dtype=int)
    regime[cond_var >= yellow_thr] = Regime.YELLOW
    regime[cond_var >= red_thr]    = Regime.RED
    return regime.fillna(Regime.YELLOW)


# ---------------------------------------------------------------------------
# Master regime combiner
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Combines VIX, ATR-expansion, HMM, and optionally GJR-GARCH into a
    single GREEN / YELLOW / RED series.

    Usage
    -----
    >>> rd = RegimeDetector()
    >>> regime = rd.fit_predict(spx_df, vix_df)
    >>> print(regime.value_counts())
    """

    def __init__(
        self,
        use_hmm: bool = True,
        use_garch: bool = False,      # off by default (slow)
        hmm_states: int = 4,
        vix_yellow_z: float = 1.5,
        vix_red_z: float = 3.0,
        atr_yellow_ratio: float = 1.5,
        atr_red_ratio: float = 2.5,
        random_state: int = 42,       # passed through to HMM; no global seed mutation
    ):
        self.use_hmm          = use_hmm
        self.use_garch        = use_garch
        self.hmm_states       = hmm_states
        self.vix_yellow_z     = vix_yellow_z
        self.vix_red_z        = vix_red_z
        self.atr_yellow_ratio = atr_yellow_ratio
        self.atr_red_ratio    = atr_red_ratio
        self.random_state     = random_state
        self._last_components: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    def fit_predict(
        self,
        spx_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute regime for every row in spx_df.

        Parameters
        ----------
        spx_df : Daily OHLCV DataFrame (must have Open, High, Low, Close).
        vix_df : DataFrame with 'Close' column for VIX.
                 If None, ATR-based and HMM signals are used only.

        Returns
        -------
        pd.Series of int (0=GREEN, 1=YELLOW, 2=RED), date-indexed.
        """
        components: list[pd.Series] = []

        # ── ATR expansion ────────────────────────────────────────────────────
        atr_sig = atr_expansion_regime(
            spx_df,
            yellow_ratio=self.atr_yellow_ratio,
            red_ratio=self.atr_red_ratio,
        )
        self._last_components["atr"] = atr_sig
        components.append(atr_sig)

        # ── VIX Z-score ──────────────────────────────────────────────────────
        if vix_df is not None:
            vix_sig = vix_regime(
                vix_df["Close"].reindex(spx_df.index).ffill(),
                yellow_z=self.vix_yellow_z,
                red_z=self.vix_red_z,
            )
            self._last_components["vix"] = vix_sig
            components.append(vix_sig)

        # ── HMM ──────────────────────────────────────────────────────────────
        if self.use_hmm:
            ret     = spx_df["Close"].pct_change()
            rng_pct = (spx_df["High"] - spx_df["Low"]) / spx_df["Close"].shift(1)
            hmm_sig = hmm_regime(ret, rng_pct, n_states=self.hmm_states,
                                  random_state=self.random_state)
            self._last_components["hmm"] = hmm_sig
            components.append(hmm_sig)

        # ── GJR-GARCH ────────────────────────────────────────────────────────
        if self.use_garch:
            ret      = spx_df["Close"].pct_change()
            garch_sig = gjr_garch_signal(ret)
            self._last_components["garch"] = garch_sig
            components.append(garch_sig)

        # ── Combine: worst-wins ───────────────────────────────────────────────
        combined = pd.concat(components, axis=1).max(axis=1).astype(int)
        combined.index = spx_df.index

        dist = combined.value_counts().sort_index()
        logger.info(
            "Regime distribution: GREEN=%d YELLOW=%d RED=%d",
            dist.get(Regime.GREEN,  0),
            dist.get(Regime.YELLOW, 0),
            dist.get(Regime.RED,    0),
        )
        return combined

    # ------------------------------------------------------------------
    @staticmethod
    def label(regime_series: pd.Series) -> pd.Series:
        """Convert int series to string labels."""
        return regime_series.map(REGIME_LABELS)

    @property
    def components(self) -> dict[str, pd.Series]:
        """Access individual component signals from last fit_predict call."""
        return self._last_components
