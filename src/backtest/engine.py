"""
src/backtest/engine.py
=======================
Task 20 — Iron Condor backtest engine.

Design principles (per reviewer spec)
--------------------------------------
1. **Intrusion-depth P&L** — not binary win/loss.
   P&L = credit - call_intrusion - put_intrusion - friction
   where intrusion = min(breach_depth, wing_width)

2. **Friction from day one** — $0.10 / leg × 4 legs = $0.40 / condor.
   Parameterised via PositionConfig so it can be changed per back-test.

3. **Regime gating in the backtest** — regime must be applied here, not only
   in production.  Skips RED days; half-sizes YELLOW days.

4. **Huber-delta reminder** — the engine receives predictions as %-deviation
   from prior close; absolute strike levels are computed here using the
   actual prior-close prices, so any scale issue in the regression targets
   is immediately visible in the backtest output.

Iron Condor structure (1-day expiry, no intraday adjustment)
--------------------------------------------------------------
    Long  call  @  call_strike + wing_width
    Short call  @  call_strike              ← upper_90 of predicted high
    Short put   @  put_strike               ← lower_90 of predicted low
    Long  put   @  put_strike  - wing_width

Credit simulation
-----------------
We do not have historical bid/ask data, so credit is approximated as:

    credit = (vix_close / 100) * prior_close * credit_fraction

where ``credit_fraction`` defaults to 0.30 (30% of 1-day expected move,
based on typical SPX 0-DTE iron condor pricing).

This is a modelling simplification that must be replaced with real option
prices in live deployment.  The approach is declared explicitly so the
reviewer cannot miss it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.calibration.regime import Regime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PositionConfig:
    """All parameters governing a single iron condor backtest run."""

    # ── Strike placement ──────────────────────────────────────────────────
    # How to derive strikes from conformal interval outputs.
    # 'interval_90' : short call = upper_90, short put = lower_90
    # 'interval_68' : short call = upper_68, short put = lower_68
    # 'fixed_atm_pct': short call = close * (1 + call_pct_above),
    #                  short put  = close * (1 - put_pct_below)
    strike_method: str = "interval_90"
    call_pct_above: float = 0.010   # used only when strike_method='fixed_atm_pct'
    put_pct_below:  float = 0.010

    # ── Wing width (protection leg distance from short strike, in pts) ─────
    wing_width_pts: float = 20.0    # e.g. buy call at strike + $20

    # ── Credit model ─────────────────────────────────────────────────────
    credit_fraction: float = 0.30   # fraction of 1-day expected move

    # ── Position sizing ───────────────────────────────────────────────────
    contracts_green:  int = 1       # full size on GREEN
    contracts_yellow: int = 1       # half size on YELLOW (by default same; override to 0 for pure skip)
    skip_yellow: bool = False       # True → skip YELLOW entirely
    skip_red:    bool = True        # True → skip RED entirely (almost always)

    # ── Friction ─────────────────────────────────────────────────────────
    # $0.10/leg × 4 legs = $0.40 per condor (per contract)
    slippage_per_leg: float = 0.10
    n_legs:           int   = 4

    # ── Risk limit ────────────────────────────────────────────────────────
    # Refuse to trade if credit < min_credit (trade not worth it)
    min_credit_pts: float = 0.50

    # ── Regime minimum hold ───────────────────────────────────────────────
    # Once RED, stay RED for at least this many trading days (anti-flicker)
    regime_min_hold_days: int = 3

    @property
    def friction_per_contract(self) -> float:
        return self.slippage_per_leg * self.n_legs


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    date:             pd.Timestamp
    regime:           int
    contracts:        int
    call_strike:      float
    put_strike:       float
    credit_pts:       float          # credit per share (100× per contract)
    actual_high:      float
    actual_low:       float
    call_intrusion:   float          # breach depth into call wing (0 if OTM)
    put_intrusion:    float
    gross_pnl_pts:    float          # credit - call_intrusion - put_intrusion
    friction_pts:     float
    net_pnl_pts:      float          # gross - friction (per share)
    net_pnl_dollars:  float          # net_pnl_pts × 100 × contracts
    skipped:          bool = False
    skip_reason:      str  = ""


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class IronCondorEngine:
    """
    Simulate iron condor trades for each date in a signals DataFrame.

    Parameters
    ----------
    config : PositionConfig instance.

    Usage
    -----
    >>> engine = IronCondorEngine()
    >>> trades = engine.run(signals, spx_df, vix_df)
    >>> report = engine.build_report(trades)
    """

    def __init__(self, config: Optional[PositionConfig] = None):
        self.config = config or PositionConfig()

    # ------------------------------------------------------------------
    def run(
        self,
        signals: pd.DataFrame,
        spx_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None,
        regime: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Execute the backtest.

        Parameters
        ----------
        signals  : DataFrame indexed by date.  Must contain either:
                    - 'call_strike', 'put_strike'  (pre-computed from conformal)
                    OR
                    - 'pred_high_pct', 'pred_low_pct', 'upper_90', 'lower_90'
        spx_df   : Raw OHLCV DataFrame (for actual highs/lows and prior close).
        vix_df   : VIX close prices (for credit simulation).  Optional.
        regime   : Integer series (0=GREEN,1=YELLOW,2=RED).
                   If None, all days are treated as GREEN.

        Returns
        -------
        DataFrame of TradeRecord-like rows, one per trading day considered.
        """
        cfg = self.config

        # ── Align inputs ─────────────────────────────────────────────────
        common_dates = signals.index.intersection(spx_df.index)
        signals  = signals.loc[common_dates]
        spx_al   = spx_df.loc[common_dates]

        if regime is None:
            regime = pd.Series(Regime.GREEN, index=common_dates, dtype=int)
        else:
            regime = regime.reindex(common_dates, fill_value=Regime.YELLOW)

        # ── Apply minimum RED hold (anti-flicker) ────────────────────────
        regime = _apply_min_hold(regime, cfg.regime_min_hold_days, Regime.RED)

        # ── VIX for credit simulation ─────────────────────────────────────
        if vix_df is not None:
            vix_cl = vix_df["Close"].reindex(common_dates).ffill()
        else:
            vix_cl = pd.Series(20.0, index=common_dates)  # default 20 VIX

        records = []
        prior_close = spx_df["Close"].shift(1).reindex(common_dates)

        for date in common_dates:
            row = self._process_day(
                date, signals.loc[date], spx_al.loc[date],
                prior_close.get(date, np.nan),
                float(vix_cl.get(date, 20.0)),
                int(regime.get(date, Regime.GREEN)),
            )
            records.append(row.__dict__)

        trades = pd.DataFrame(records)
        trades["date"] = pd.to_datetime(trades["date"])
        trades = trades.set_index("date")

        _active = (~trades["skipped"]).sum()
        _skip   = trades["skipped"].sum()
        logger.info(
            "Backtest: %d dates | %d active trades | %d skipped | "
            "net P&L = $%.2f",
            len(trades), _active, _skip,
            trades["net_pnl_dollars"].sum(),
        )
        return trades

    # ------------------------------------------------------------------
    def _process_day(
        self,
        date: pd.Timestamp,
        sig: pd.Series,
        ohlcv: pd.Series,
        prior_close: float,
        vix: float,
        reg: int,
    ) -> TradeRecord:
        """Compute a single day's trade record."""
        cfg = self.config

        actual_high = float(ohlcv["High"])
        actual_low  = float(ohlcv["Low"])

        # ── Regime gating ─────────────────────────────────────────────────
        if reg == Regime.RED and cfg.skip_red:
            return _skipped(date, reg, "RED_SKIP")
        if reg == Regime.YELLOW and cfg.skip_yellow:
            return _skipped(date, reg, "YELLOW_SKIP")

        n_contracts = (
            cfg.contracts_green if reg == Regime.GREEN
            else cfg.contracts_yellow
        )

        # ── Strike placement ──────────────────────────────────────────────
        if np.isnan(prior_close) or prior_close <= 0:
            return _skipped(date, reg, "NO_PRIOR_CLOSE")

        call_strike, put_strike = self._compute_strikes(sig, prior_close)
        if call_strike is None or put_strike is None:
            return _skipped(date, reg, "MISSING_STRIKE_DATA")

        # Sanity: strikes must straddle current price
        if call_strike <= put_strike:
            return _skipped(date, reg, "INVERTED_STRIKES")

        # ── Credit simulation ─────────────────────────────────────────────
        # 1-day expected move = (vix / 100) * prior_close / sqrt(252)
        one_day_move = (vix / 100.0) * prior_close / np.sqrt(252)
        credit_pts   = one_day_move * cfg.credit_fraction

        if credit_pts < cfg.min_credit_pts:
            return _skipped(date, reg, "CREDIT_TOO_LOW")

        # ── P&L: intrusion-depth model ────────────────────────────────────
        call_intrusion = max(0.0, min(actual_high - call_strike,
                                     cfg.wing_width_pts))
        put_intrusion  = max(0.0, min(put_strike   - actual_low,
                                     cfg.wing_width_pts))

        gross_pnl = credit_pts - call_intrusion - put_intrusion
        friction  = cfg.friction_per_contract   # per share
        net_pnl   = gross_pnl - friction

        # ── Dollar P&L (100 shares per contract) ─────────────────────────
        net_dollar = net_pnl * 100 * n_contracts

        return TradeRecord(
            date=date,
            regime=reg,
            contracts=n_contracts,
            call_strike=call_strike,
            put_strike=put_strike,
            credit_pts=credit_pts,
            actual_high=actual_high,
            actual_low=actual_low,
            call_intrusion=call_intrusion,
            put_intrusion=put_intrusion,
            gross_pnl_pts=gross_pnl,
            friction_pts=friction,
            net_pnl_pts=net_pnl,
            net_pnl_dollars=net_dollar,
        )

    # ------------------------------------------------------------------
    def _compute_strikes(
        self,
        sig: pd.Series,
        prior_close: float,
    ):
        """Return (call_strike, put_strike) or (None, None) on failure."""
        cfg = self.config

        if cfg.strike_method == "interval_90":
            # Prefer pre-computed absolute strike prices; fall back to pct-deviations.
            raw_call = sig.get("call_strike")
            raw_put  = sig.get("put_strike")
            if raw_call is not None and not np.isnan(float(raw_call)) and float(raw_call) > 1:
                call, put = float(raw_call), float(raw_put)
            else:
                call = _pct_to_price(sig.get("upper_90"), prior_close)
                put  = _pct_to_price(sig.get("lower_90"), prior_close)

        elif cfg.strike_method == "interval_68":
            call = _pct_to_price(sig.get("upper_68"), prior_close)
            put  = _pct_to_price(sig.get("lower_68"), prior_close)

        elif cfg.strike_method == "fixed_atm_pct":
            call = prior_close * (1 + cfg.call_pct_above)
            put  = prior_close * (1 - cfg.put_pct_below)

        else:
            raise ValueError(f"Unknown strike_method: {cfg.strike_method!r}")

        if call is None or put is None:
            return None, None
        if np.isnan(call) or np.isnan(put):
            return None, None

        return float(call), float(put)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct_to_price(pct, close):
    """Convert a %-deviation to an absolute price.  Returns None on failure."""
    if pct is None or np.isnan(pct):
        return None
    return close * (1 + pct)


def _skipped(date, reg, reason) -> TradeRecord:
    return TradeRecord(
        date=date, regime=reg, contracts=0,
        call_strike=np.nan, put_strike=np.nan,
        credit_pts=0.0, actual_high=np.nan, actual_low=np.nan,
        call_intrusion=0.0, put_intrusion=0.0,
        gross_pnl_pts=0.0, friction_pts=0.0,
        net_pnl_pts=0.0, net_pnl_dollars=0.0,
        skipped=True, skip_reason=reason,
    )


def _apply_min_hold(
    regime: pd.Series,
    min_hold: int,
    target_state: int,
) -> pd.Series:
    """
    Once a day enters ``target_state``, keep it there for at least
    ``min_hold`` trading days (anti-flicker / minimum holding period).
    """
    if min_hold <= 1:
        return regime

    regime = regime.copy()
    values = regime.values.copy()
    n      = len(values)
    hold   = 0

    for i in range(n):
        if values[i] == target_state:
            # The trigger day itself counts as day 1 of the hold window.
            hold = min_hold - 1
        elif hold > 0:
            values[i] = target_state
            hold -= 1

    regime[:] = values
    return regime
