"""
src/data/ibkr_fetcher.py
=========================
IBKR live market data fetcher using ib_insync.

Connects to IB Gateway (paper or live) and pulls:
  - SPX spot price
  - VIX spot price
  - SPXW 0DTE option chain with full Greeks
  - Computed Greeks summary (IV skew, GEX, put/call ratio, walls)

All functions are wrapped in try/except — they return partial/None
results on failure and never raise into the calling pipeline.

Functions
---------
connect_ib()                    Connect to IB Gateway, return IB instance.
disconnect_ib(ib)               Clean disconnect.
fetch_spx_price(ib)             Live SPX last/mid price.
fetch_vix_price(ib)             Live VIX price.
fetch_spx_0dte_chain(ib, spx)   SPXW 0DTE option chain as DataFrame.
compute_greeks_summary(chain)   Aggregate Greeks metrics dict.
fetch_ibkr_snapshot(ib, spx)    Master function — all of the above.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── IBKR connection defaults (overridden by settings.py) ──────────────────────
_DEFAULT_HOST      = "127.0.0.1"
_DEFAULT_PORT      = 4002
_DEFAULT_CLIENT_ID = 10
_DEFAULT_TIMEOUT   = 10     # seconds to wait for IB Gateway handshake
_DATA_TIMEOUT      = 5      # seconds to wait per market data request

# ── Option chain fetch parameters ─────────────────────────────────────────────
_SPXW_BATCH_SIZE   = 50     # contracts per reqMktData batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ibkr_settings() -> tuple[str, int, int, int]:
    """Load host/port/client_id/timeout from settings.py if available."""
    try:
        from config.settings import (
            IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, IBKR_TIMEOUT,
        )
        return IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, IBKR_TIMEOUT
    except Exception:
        return _DEFAULT_HOST, _DEFAULT_PORT, _DEFAULT_CLIENT_ID, _DEFAULT_TIMEOUT


def _today_str() -> str:
    """Return today's date as YYYYMMDD (IB expiry format)."""
    return date.today().strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect_ib() -> "IB":  # type: ignore[name-defined]
    """Connect to IB Gateway and return a connected IB instance.

    Parameters are loaded from ``config/settings.py`` if available,
    otherwise the module-level defaults are used.

    Returns
    -------
    ib_insync.IB
        A connected IB client instance.

    Raises
    ------
    ConnectionError
        If the gateway is unreachable or the handshake times out.
    """
    from ib_insync import IB, util

    host, port, client_id, timeout = _load_ibkr_settings()

    logger.info(
        "connect_ib: connecting to %s:%d (clientId=%d, timeout=%ds)",
        host, port, client_id, timeout,
    )

    ib = IB()
    # util.startLoop() is only needed in Jupyter / asyncio contexts;
    # ib.connect() handles its own event loop in script mode.
    try:
        ib.connect(host, port, clientId=client_id, timeout=timeout)
    except Exception as exc:
        raise ConnectionError(
            f"IB Gateway connection failed ({host}:{port}): {exc}"
        ) from exc

    if not ib.isConnected():
        raise ConnectionError(
            f"IB Gateway at {host}:{port} did not confirm connection."
        )

    logger.info(
        "connect_ib: connected — server version %s, account %s",
        ib.client.serverVersion(),
        next(iter(ib.managedAccounts()), "?"),
    )
    return ib


def disconnect_ib(ib: "IB") -> None:  # type: ignore[name-defined]
    """Disconnect from IB Gateway cleanly.

    Safe to call even if *ib* is already disconnected.
    """
    try:
        if ib is not None and ib.isConnected():
            ib.disconnect()
            logger.info("disconnect_ib: disconnected cleanly.")
    except Exception as exc:
        logger.warning("disconnect_ib: error during disconnect: %s", exc)


# ---------------------------------------------------------------------------
# Price fetchers
# ---------------------------------------------------------------------------

def fetch_spx_price(ib: "IB") -> Optional[float]:  # type: ignore[name-defined]
    """Fetch the live SPX index price from IBKR.

    Returns the last traded price; falls back to mid-quote if last is
    unavailable (as is common outside regular trading hours).

    Parameters
    ----------
    ib : ib_insync.IB
        A connected IB instance.

    Returns
    -------
    float or None
        SPX price, or None if the request fails.
    """
    try:
        from ib_insync import Contract

        spx = Contract(
            symbol="SPX",
            secType="IND",
            exchange="CBOE",
            currency="USD",
        )
        [spx] = ib.qualifyContracts(spx)

        ticker = ib.reqMktData(spx, genericTickList="", snapshot=True)
        # Wait up to _DATA_TIMEOUT seconds for a price
        ib.sleep(0)   # flush pending events
        deadline = datetime.now().timestamp() + _DATA_TIMEOUT
        while datetime.now().timestamp() < deadline:
            if (ticker.last and not np.isnan(ticker.last)) or \
               (ticker.bid and ticker.ask and
                    not np.isnan(ticker.bid) and not np.isnan(ticker.ask)):
                break
            ib.sleep(0.25)

        price: Optional[float] = None

        def _valid(v) -> bool:
            """True if v is a real, non-sentinel price (IB uses -1 for N/A)."""
            return v is not None and not np.isnan(v) and v > 0

        if _valid(ticker.last):
            price = float(ticker.last)
        elif _valid(ticker.bid) and _valid(ticker.ask):
            price = (float(ticker.bid) + float(ticker.ask)) / 2.0
        elif _valid(ticker.close):
            price = float(ticker.close)

        ib.cancelMktData(spx)

        if price is None:
            logger.warning("fetch_spx_price: no valid price returned (market may be closed).")
        else:
            logger.info("fetch_spx_price: SPX = %.2f", price)
        return price

    except Exception as exc:
        logger.error("fetch_spx_price failed: %s", exc)
        return None


def fetch_vix_price(ib: "IB") -> Optional[float]:  # type: ignore[name-defined]
    """Fetch the live VIX index price from IBKR.

    Parameters
    ----------
    ib : ib_insync.IB
        A connected IB instance.

    Returns
    -------
    float or None
        VIX level, or None if the request fails.
    """
    try:
        from ib_insync import Contract

        vix = Contract(
            symbol="VIX",
            secType="IND",
            exchange="CBOE",
            currency="USD",
        )
        [vix] = ib.qualifyContracts(vix)

        ticker = ib.reqMktData(vix, genericTickList="", snapshot=True)
        ib.sleep(0)
        deadline = datetime.now().timestamp() + _DATA_TIMEOUT
        while datetime.now().timestamp() < deadline:
            if (ticker.last and not np.isnan(ticker.last)) or \
               (ticker.bid and ticker.ask and
                    not np.isnan(ticker.bid) and not np.isnan(ticker.ask)):
                break
            ib.sleep(0.25)

        price: Optional[float] = None

        def _valid(v) -> bool:
            """True if v is a real, non-sentinel price (IB uses -1 for N/A)."""
            return v is not None and not np.isnan(v) and v > 0

        if _valid(ticker.last):
            price = float(ticker.last)
        elif _valid(ticker.bid) and _valid(ticker.ask):
            price = (float(ticker.bid) + float(ticker.ask)) / 2.0
        elif _valid(ticker.close):
            price = float(ticker.close)

        ib.cancelMktData(vix)

        if price is None:
            logger.warning("fetch_vix_price: no valid price returned (market may be closed).")
        else:
            logger.info("fetch_vix_price: VIX = %.2f", price)
        return price

    except Exception as exc:
        logger.error("fetch_vix_price failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Option chain fetcher
# ---------------------------------------------------------------------------

def fetch_spx_0dte_chain(
    ib: "IB",  # type: ignore[name-defined]
    spx_price: Optional[float] = None,
) -> pd.DataFrame:
    """Fetch the SPXW 0DTE (today's expiry) option chain from IBKR.

    Uses ``reqSecDefOptParams`` to discover available strikes for today's
    SPXW expiry, then batches ``reqMktData`` snapshot calls to fill in
    market data (bid/ask/IV/Greeks).

    Parameters
    ----------
    ib        : ib_insync.IB — connected IB instance.
    spx_price : float, optional — current SPX spot (used to filter strikes
                to a ±20% band around spot to keep the request manageable).

    Returns
    -------
    pd.DataFrame
        Columns: strike, right, bid, ask, mid, iv, delta, gamma,
                 theta, vega, oi, volume
        Empty DataFrame on failure.
    """
    try:
        from ib_insync import Option

        expiry = _today_str()
        logger.info("fetch_spx_0dte_chain: fetching SPXW chain for %s", expiry)

        # ── Discover strikes & expirations via reqSecDefOptParams ──────────────
        opt_params = ib.reqSecDefOptParams(
            underlyingSymbol="SPXW",
            futFopExchange="",
            underlyingSecType="IND",
            underlyingConId=0,
        )

        if not opt_params:
            # Fallback: try SPX instead
            opt_params = ib.reqSecDefOptParams(
                underlyingSymbol="SPX",
                futFopExchange="",
                underlyingSecType="IND",
                underlyingConId=0,
            )

        # Locate the chain that includes today's expiry
        today_chain = None
        for params in opt_params:
            if expiry in (params.expirations or []):
                today_chain = params
                break

        if today_chain is None:
            logger.warning(
                "fetch_spx_0dte_chain: no SPXW/SPX chain found for expiry %s. "
                "Market may be closed or 0DTE options not listed today.",
                expiry,
            )
            return pd.DataFrame()

        all_strikes = sorted(today_chain.strikes)
        logger.info(
            "fetch_spx_0dte_chain: found %d strikes on exchange %s",
            len(all_strikes), today_chain.exchange,
        )

        # ── Filter strikes to a sensible band around spot ──────────────────────
        if spx_price and spx_price > 0:
            lo = spx_price * 0.80
            hi = spx_price * 1.20
            strikes = [s for s in all_strikes if lo <= s <= hi]
            logger.info(
                "fetch_spx_0dte_chain: filtered to %d strikes in [%.0f, %.0f]",
                len(strikes), lo, hi,
            )
        else:
            strikes = all_strikes

        if not strikes:
            logger.warning("fetch_spx_0dte_chain: no strikes after filtering.")
            return pd.DataFrame()

        exchange = today_chain.exchange or "SMART"

        # ── Build contracts for both calls and puts ────────────────────────────
        contracts = []
        for strike in strikes:
            for right in ("C", "P"):
                contracts.append(
                    Option(
                        symbol="SPXW",
                        lastTradeDateOrContractMonth=expiry,
                        strike=strike,
                        right=right,
                        exchange=exchange,
                        currency="USD",
                    )
                )

        # Qualify contracts in batches (IB has ~50 contract limit per qualify call)
        qualified: list = []
        for i in range(0, len(contracts), _SPXW_BATCH_SIZE):
            batch = contracts[i : i + _SPXW_BATCH_SIZE]
            try:
                q = ib.qualifyContracts(*batch)
                qualified.extend([c for c in q if c.conId])
            except Exception as e:
                logger.warning("Qualify batch %d failed: %s", i // _SPXW_BATCH_SIZE, e)

        if not qualified:
            logger.warning("fetch_spx_0dte_chain: no contracts qualified.")
            return pd.DataFrame()

        logger.info("fetch_spx_0dte_chain: requesting market data for %d contracts", len(qualified))

        # ── Request market data in batches ────────────────────────────────────
        tickers = []
        for i in range(0, len(qualified), _SPXW_BATCH_SIZE):
            batch = qualified[i : i + _SPXW_BATCH_SIZE]
            batch_tickers = [
                ib.reqMktData(c, genericTickList="106", snapshot=True)
                for c in batch
            ]
            tickers.extend(batch_tickers)
            ib.sleep(0)   # flush

        # Wait for Greeks to populate
        ib.sleep(_DATA_TIMEOUT)
        ib.sleep(0)

        # ── Parse results into a DataFrame ─────────────────────────────────────
        rows = []
        for tkr in tickers:
            c = tkr.contract
            if not c:
                continue

            # Extract Greeks (modelGreeks or bidGreeks or askGreeks)
            greeks = tkr.modelGreeks or tkr.lastGreeks or tkr.bidGreeks
            iv     = float(greeks.impliedVol) if greeks and greeks.impliedVol and not np.isnan(greeks.impliedVol) else np.nan
            delta  = float(greeks.delta)      if greeks and greeks.delta      and not np.isnan(greeks.delta)      else np.nan
            gamma  = float(greeks.gamma)      if greeks and greeks.gamma      and not np.isnan(greeks.gamma)      else np.nan
            theta  = float(greeks.theta)      if greeks and greeks.theta      and not np.isnan(greeks.theta)      else np.nan
            vega   = float(greeks.vega)       if greeks and greeks.vega       and not np.isnan(greeks.vega)       else np.nan

            bid = float(tkr.bid) if tkr.bid and not np.isnan(tkr.bid) else np.nan
            ask = float(tkr.ask) if tkr.ask and not np.isnan(tkr.ask) else np.nan
            mid = (bid + ask) / 2.0 if not (np.isnan(bid) or np.isnan(ask)) else np.nan

            oi     = int(tkr.optionOpenInterest) if tkr.optionOpenInterest and not np.isnan(tkr.optionOpenInterest) else 0
            volume = int(tkr.volume)             if tkr.volume             and not np.isnan(tkr.volume)             else 0

            rows.append({
                "strike": float(c.strike),
                "right":  c.right,       # 'C' or 'P'
                "bid":    bid,
                "ask":    ask,
                "mid":    mid,
                "iv":     iv,
                "delta":  delta,
                "gamma":  gamma,
                "theta":  theta,
                "vega":   vega,
                "oi":     oi,
                "volume": volume,
            })

        # Cancel all market data subscriptions
        for tkr in tickers:
            try:
                ib.cancelMktData(tkr.contract)
            except Exception:
                pass

        chain_df = pd.DataFrame(rows)
        if chain_df.empty:
            logger.warning("fetch_spx_0dte_chain: resulting DataFrame is empty.")
        else:
            logger.info(
                "fetch_spx_0dte_chain: %d rows fetched (%d strikes × 2 rights)",
                len(chain_df), chain_df["strike"].nunique(),
            )
        return chain_df

    except Exception as exc:
        logger.error("fetch_spx_0dte_chain failed: %s", exc, exc_info=True)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Greeks summary
# ---------------------------------------------------------------------------

def compute_greeks_summary(
    chain_df: pd.DataFrame,
    spx_price: Optional[float] = None,
) -> dict:
    """Compute aggregate option-market metrics from the 0DTE chain.

    Parameters
    ----------
    chain_df  : DataFrame from ``fetch_spx_0dte_chain``.
    spx_price : float, optional — SPX spot used for ATM strike selection.
                If None, uses the median of available strikes.

    Returns
    -------
    dict with keys:
        atm_iv          — average of ATM call+put implied volatility
        iv_skew_25d     — 25-delta put IV minus 25-delta call IV
        put_call_ratio  — sum(put OI) / sum(call OI)
        gex             — net gamma exposure (calls - puts) in $ millions
        atm_gamma       — gamma at the ATM strike (call average)
        call_wall       — strike with highest call open interest
        put_wall        — strike with highest put open interest
    """
    result: dict = {
        "atm_iv":         None,
        "iv_skew_25d":    None,
        "put_call_ratio": None,
        "gex":            None,
        "atm_gamma":      None,
        "call_wall":      None,
        "put_wall":       None,
    }

    if chain_df is None or chain_df.empty:
        logger.warning("compute_greeks_summary: received empty chain_df.")
        return result

    try:
        calls = chain_df[chain_df["right"] == "C"].copy()
        puts  = chain_df[chain_df["right"] == "P"].copy()

        # ── Spot / ATM strike ──────────────────────────────────────────────────
        all_strikes = sorted(chain_df["strike"].unique())
        if spx_price and spx_price > 0:
            atm_strike = min(all_strikes, key=lambda s: abs(s - spx_price))
        else:
            atm_strike = float(np.median(all_strikes))

        # ── ATM IV ────────────────────────────────────────────────────────────
        atm_calls = calls[calls["strike"] == atm_strike]
        atm_puts  = puts[puts["strike"]  == atm_strike]

        atm_call_iv = atm_calls["iv"].dropna().mean()
        atm_put_iv  = atm_puts["iv"].dropna().mean()
        valid_ivs   = [v for v in [atm_call_iv, atm_put_iv] if not np.isnan(v)]
        result["atm_iv"] = float(np.mean(valid_ivs)) if valid_ivs else None

        # ── ATM Gamma ─────────────────────────────────────────────────────────
        atm_gamma_vals = atm_calls["gamma"].dropna()
        result["atm_gamma"] = float(atm_gamma_vals.mean()) if not atm_gamma_vals.empty else None

        # ── IV Skew (25-delta) ─────────────────────────────────────────────────
        # Find the put nearest to -0.25 delta and call nearest to +0.25 delta
        puts_with_delta  = puts[puts["delta"].notna()].copy()
        calls_with_delta = calls[calls["delta"].notna()].copy()

        put_25d_iv  = None
        call_25d_iv = None

        if not puts_with_delta.empty:
            # Put deltas are negative; find closest to -0.25
            puts_with_delta["delta_dist"] = (puts_with_delta["delta"] - (-0.25)).abs()
            best_put = puts_with_delta.loc[puts_with_delta["delta_dist"].idxmin()]
            if not np.isnan(best_put["iv"]):
                put_25d_iv = float(best_put["iv"])

        if not calls_with_delta.empty:
            calls_with_delta["delta_dist"] = (calls_with_delta["delta"] - 0.25).abs()
            best_call = calls_with_delta.loc[calls_with_delta["delta_dist"].idxmin()]
            if not np.isnan(best_call["iv"]):
                call_25d_iv = float(best_call["iv"])

        if put_25d_iv is not None and call_25d_iv is not None:
            result["iv_skew_25d"] = put_25d_iv - call_25d_iv

        # ── Put/Call Ratio (by OI, falling back to volume) ────────────────────
        put_oi  = puts["oi"].sum()
        call_oi = calls["oi"].sum()
        if call_oi > 0:
            result["put_call_ratio"] = float(put_oi) / float(call_oi)
        else:
            # Fallback to volume
            put_vol  = puts["volume"].sum()
            call_vol = calls["volume"].sum()
            if call_vol > 0:
                result["put_call_ratio"] = float(put_vol) / float(call_vol)

        # ── GEX (Gamma Exposure) ──────────────────────────────────────────────
        # GEX = Σ(gamma × OI × 100 × SPX) per strike
        # Calls are positive (dealers long gamma), puts are negative
        spot = spx_price if (spx_price and spx_price > 0) else 5000.0  # fallback

        call_gex = calls.apply(
            lambda r: r["gamma"] * r["oi"] * 100 * spot
            if not np.isnan(r["gamma"]) else 0.0,
            axis=1,
        ).sum()

        put_gex = puts.apply(
            lambda r: r["gamma"] * r["oi"] * 100 * spot
            if not np.isnan(r["gamma"]) else 0.0,
            axis=1,
        ).sum()

        result["gex"] = float(call_gex - put_gex)

        # ── Call Wall / Put Wall (peak OI strike) ─────────────────────────────
        if not calls.empty and calls["oi"].sum() > 0:
            result["call_wall"] = float(calls.loc[calls["oi"].idxmax(), "strike"])

        if not puts.empty and puts["oi"].sum() > 0:
            result["put_wall"] = float(puts.loc[puts["oi"].idxmax(), "strike"])

        logger.info(
            "compute_greeks_summary: atm_iv=%.4f skew=%.4f pc_ratio=%.3f "
            "gex=%.0f call_wall=%.0f put_wall=%.0f",
            result.get("atm_iv") or 0.0,
            result.get("iv_skew_25d") or 0.0,
            result.get("put_call_ratio") or 0.0,
            result.get("gex") or 0.0,
            result.get("call_wall") or 0.0,
            result.get("put_wall") or 0.0,
        )

    except Exception as exc:
        logger.error("compute_greeks_summary failed: %s", exc, exc_info=True)

    return result


# ---------------------------------------------------------------------------
# Master snapshot function
# ---------------------------------------------------------------------------

def fetch_ibkr_snapshot(
    ib: "IB",  # type: ignore[name-defined]
    spx_price: Optional[float] = None,
) -> dict:
    """Master snapshot: fetch SPX, VIX, 0DTE chain and compute all Greeks.

    This is the single entry-point for external callers that already have
    a connected IB instance.

    Parameters
    ----------
    ib        : ib_insync.IB — connected IB instance.
    spx_price : float, optional — if provided, skips the live SPX fetch
                and uses this value for chain filtering/Greek computation.

    Returns
    -------
    dict with keys:
        spx_price       — float or None
        vix             — float or None
        chain           — pd.DataFrame (may be empty)
        atm_iv          — float or None
        iv_skew_25d     — float or None
        put_call_ratio  — float or None
        gex             — float or None
        atm_gamma       — float or None
        call_wall       — float or None
        put_wall        — float or None
        data_quality    — 'FULL' | 'PARTIAL' | 'NO_CHAIN' | 'ERROR'
        timestamp       — ISO timestamp string
    """
    snapshot: dict = {
        "spx_price":      None,
        "vix":            None,
        "chain":          pd.DataFrame(),
        "atm_iv":         None,
        "iv_skew_25d":    None,
        "put_call_ratio": None,
        "gex":            None,
        "atm_gamma":      None,
        "call_wall":      None,
        "put_wall":       None,
        "data_quality":   "ERROR",
        "timestamp":      datetime.now().isoformat(),
    }

    try:
        # ── SPX price ──────────────────────────────────────────────────────────
        if spx_price is not None:
            snapshot["spx_price"] = spx_price
        else:
            snapshot["spx_price"] = fetch_spx_price(ib)

        # ── VIX price ──────────────────────────────────────────────────────────
        snapshot["vix"] = fetch_vix_price(ib)

        # ── 0DTE chain ─────────────────────────────────────────────────────────
        chain = fetch_spx_0dte_chain(ib, snapshot["spx_price"])
        snapshot["chain"] = chain

        if chain.empty:
            snapshot["data_quality"] = "NO_CHAIN"
            logger.warning("fetch_ibkr_snapshot: chain empty — Greeks unavailable.")
            return snapshot

        # ── Greeks summary ─────────────────────────────────────────────────────
        greeks = compute_greeks_summary(chain, snapshot["spx_price"])
        snapshot.update(greeks)

        # ── Data quality assessment ────────────────────────────────────────────
        none_count = sum(1 for k, v in snapshot.items()
                         if k not in ("chain", "timestamp", "data_quality") and v is None)
        if none_count == 0:
            snapshot["data_quality"] = "FULL"
        elif none_count <= 3:
            snapshot["data_quality"] = "PARTIAL"
        else:
            snapshot["data_quality"] = "DEGRADED"

        logger.info(
            "fetch_ibkr_snapshot complete: quality=%s spx=%.2f vix=%.2f atm_iv=%.4f",
            snapshot["data_quality"],
            snapshot.get("spx_price") or 0.0,
            snapshot.get("vix") or 0.0,
            snapshot.get("atm_iv") or 0.0,
        )

    except Exception as exc:
        logger.error("fetch_ibkr_snapshot failed: %s", exc, exc_info=True)
        snapshot["data_quality"] = "ERROR"

    return snapshot
