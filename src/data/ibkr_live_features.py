"""
src/data/ibkr_live_features.py
================================
High-level wrapper around ibkr_fetcher that:
  1. Opens a fresh IB connection
  2. Fetches the full IBKR snapshot (SPX, VIX, 0DTE chain, Greeks)
  3. Closes the connection
  4. Returns a flat dict ready to be merged into a feature row

This module intentionally manages its own connection lifecycle so that
callers (e.g. live_fetcher.py) don't need to handle IB state.

Functions
---------
get_ibkr_features(spx_price=None) -> dict
    Returns a dict with ibkr_* keys for feature pipeline integration.
    Returns all-None dict if IB Gateway is unreachable.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Keys emitted by this module — consumers can rely on these always existing.
IBKR_FEATURE_KEYS = [
    "ibkr_spx_price",
    "ibkr_vix",
    "ibkr_atm_iv",
    "ibkr_iv_skew_25d",
    "ibkr_put_call_ratio",
    "ibkr_gex",
    "ibkr_atm_gamma",
    "ibkr_call_wall",
    "ibkr_put_wall",
    "ibkr_data_quality",
]


def _null_features() -> dict:
    """Return a dict with all IBKR feature keys set to None.

    Used as a safe fallback when IB Gateway is unavailable.
    """
    result = {k: None for k in IBKR_FEATURE_KEYS}
    result["ibkr_data_quality"] = "UNAVAILABLE"
    return result


def get_ibkr_features(spx_price: Optional[float] = None) -> dict:
    """Connect to IB Gateway, fetch a live snapshot, disconnect, return features.

    Opens a **new** connection each call and always cleans up — safe to
    call from a cron/scheduler context where no persistent IB instance
    exists.

    Parameters
    ----------
    spx_price : float, optional
        If supplied, skips the live SPX price request and uses this value
        for option-chain filtering and Greek computations.  Useful when
        the caller already has a fresh price from another source.

    Returns
    -------
    dict
        Keys (all prefixed ``ibkr_``):

        =================== =========================================
        ibkr_spx_price      Live SPX index price
        ibkr_vix            Live VIX level
        ibkr_atm_iv         ATM implied volatility (call/put average)
        ibkr_iv_skew_25d    25-delta put IV minus 25-delta call IV
        ibkr_put_call_ratio Put OI / Call OI (or volume if OI = 0)
        ibkr_gex            Net gamma exposure ($ notional)
        ibkr_atm_gamma      Gamma at the ATM strike
        ibkr_call_wall      Strike with highest call open interest
        ibkr_put_wall       Strike with highest put open interest
        ibkr_data_quality   'FULL' | 'PARTIAL' | 'DEGRADED' |
                            'NO_CHAIN' | 'ERROR' | 'UNAVAILABLE'
        =================== =========================================

        All numeric values may be ``None`` if data was unavailable.
    """
    ib = None
    try:
        # ── Import here to keep the module importable when ib_insync is absent ──
        from src.data.ibkr_fetcher import connect_ib, disconnect_ib, fetch_ibkr_snapshot

        logger.info("get_ibkr_features: connecting to IB Gateway …")
        ib = connect_ib()

        snapshot = fetch_ibkr_snapshot(ib, spx_price=spx_price)

        # Map snapshot keys → ibkr_ feature keys
        features = {
            "ibkr_spx_price":      snapshot.get("spx_price"),
            "ibkr_vix":            snapshot.get("vix"),
            "ibkr_atm_iv":         snapshot.get("atm_iv"),
            "ibkr_iv_skew_25d":    snapshot.get("iv_skew_25d"),
            "ibkr_put_call_ratio": snapshot.get("put_call_ratio"),
            "ibkr_gex":            snapshot.get("gex"),
            "ibkr_atm_gamma":      snapshot.get("atm_gamma"),
            "ibkr_call_wall":      snapshot.get("call_wall"),
            "ibkr_put_wall":       snapshot.get("put_wall"),
            "ibkr_data_quality":   snapshot.get("data_quality", "ERROR"),
        }

        logger.info(
            "get_ibkr_features: quality=%s spx=%s vix=%s atm_iv=%s",
            features["ibkr_data_quality"],
            features["ibkr_spx_price"],
            features["ibkr_vix"],
            features["ibkr_atm_iv"],
        )
        return features

    except ConnectionError as exc:
        logger.warning("get_ibkr_features: IB Gateway unreachable — %s", exc)
        return _null_features()

    except ImportError as exc:
        logger.warning("get_ibkr_features: ib_insync not available — %s", exc)
        return _null_features()

    except Exception as exc:
        logger.error("get_ibkr_features: unexpected error — %s", exc, exc_info=True)
        null = _null_features()
        null["ibkr_data_quality"] = "ERROR"
        return null

    finally:
        # Always disconnect, even on error
        if ib is not None:
            try:
                from src.data.ibkr_fetcher import disconnect_ib
                disconnect_ib(ib)
            except Exception as e:
                logger.warning("get_ibkr_features: cleanup disconnect failed: %s", e)
