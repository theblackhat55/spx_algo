"""
src/execution/broker.py
========================
Task 33 — Interactive Brokers execution module (paper-trade first).

Uses ib_async (PyPI: ib-async>=2.1.0), the officially maintained
successor to ib_insync after the original author's passing.

CRITICAL SAFETY CONTROLS
--------------------------
* Execution only when EXECUTION_MODE != 'LIVE' defaults to paper.
* Max 1 condor per signal (configurable via N_CONTRACTS env var).
* Regime RED → skip entirely.
* data_quality DEGRADED → kill switch, skip execution.
* 60-second fill timeout; cancels unfilled orders.
* All actions logged to output/trades/trade_log_YYYYMMDD.json.

Usage
-----
    from src.execution.broker import IBKRBroker, build_condor_from_signal
    broker = IBKRBroker(paper=True)
    broker.connect()
    contract, order = build_condor_from_signal(signal, broker)
    trade = broker.place_condor_order(contract, order)
    broker.disconnect()
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAPER_PORT = 7497
LIVE_PORT  = 7496
TWS_HOST   = "127.0.0.1"
CLIENT_ID  = 10

FILL_TIMEOUT_SEC   = 60
MAX_CONTRACTS      = int(os.getenv("N_CONTRACTS", "1"))
EXECUTION_MODE     = os.getenv("EXECUTION_MODE", "PAPER").upper()

SPX_SYMBOL  = "SPX"
SPX_EXCH    = "CBOE"
SPX_CURR    = "USD"
SPX_SECTYPE = "OPT"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_expiry_dte(dte: int = 0) -> str:
    """Return SPX option expiry in YYYYMMDD format.

    dte=0 → same-day (0-DTE), dte=1 → next trading day (1-DTE).
    """
    from pandas.tseries.offsets import BDay
    import pandas as pd
    target = (pd.Timestamp.today() + BDay(dte)).date()
    return target.strftime("%Y%m%d")


def _log_trade(record: Dict[str, Any], output_dir: Optional[Path] = None) -> None:
    if output_dir is None:
        try:
            from config.settings import OUTPUT_DIR
            output_dir = Path(OUTPUT_DIR) / "trades"
        except Exception:
            output_dir = Path("output") / "trades"

    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f"trade_log_{date.today().strftime('%Y%m%d')}.json"

    logs: List[Dict] = []
    if fname.exists():
        try:
            logs = json.loads(fname.read_text())
        except Exception:
            pass

    logs.append(record)
    fname.write_text(json.dumps(logs, indent=2, default=str, sort_keys=True))
    logger.info("Trade log updated → %s", fname)


# ---------------------------------------------------------------------------
# IBKR Broker class
# ---------------------------------------------------------------------------

class IBKRBroker:
    """
    Thin wrapper around ib_async for iron-condor execution.

    All public methods are synchronous; async I/O is handled internally.

    Parameters
    ----------
    paper : bool
        If True, connect to paper-trading port (7497).
        If False AND EXECUTION_MODE='LIVE', use live port (7496).
    host  : str   TWS / IB Gateway hostname.
    client_id : int   Unique client ID (avoid conflicts with TWS charts).
    """

    def __init__(
        self,
        paper: bool = True,
        host:  str  = TWS_HOST,
        client_id: int = CLIENT_ID,
        output_dir: Optional[Path] = None,
    ):
        self.paper     = paper
        self.host      = host
        self.client_id = client_id
        self.output_dir = output_dir
        self._ib       = None          # ib_async.IB instance
        self._connected = False

        if not paper and EXECUTION_MODE != "LIVE":
            raise RuntimeError(
                "paper=False requires EXECUTION_MODE=LIVE env var as safety confirmation."
            )

    # ------------------------------------------------------------------
    def connect(self, max_attempts: int = 3) -> bool:
        """Connect to TWS/IBGateway with retry logic."""
        try:
            from ib_async import IB
        except ImportError:
            logger.error("ib_async not installed: pip install ib-async>=2.1.0")
            return False

        port = PAPER_PORT if self.paper else LIVE_PORT

        for attempt in range(1, max_attempts + 1):
            try:
                ib = IB()
                ib.connect(self.host, port, clientId=self.client_id, timeout=10)
                self._ib = ib
                self._connected = True
                logger.info("IBKRBroker connected: host=%s port=%d paper=%s",
                            self.host, port, self.paper)
                return True
            except Exception as exc:
                logger.warning("Connect attempt %d/%d failed: %s", attempt, max_attempts, exc)
                if attempt < max_attempts:
                    time.sleep(5 * attempt)

        logger.error("IBKRBroker: all %d connect attempts failed", max_attempts)
        return False

    # ------------------------------------------------------------------
    def disconnect(self) -> None:
        if self._ib and self._connected:
            try:
                self._ib.disconnect()
            except Exception:
                pass
        self._connected = False
        logger.info("IBKRBroker disconnected")

    # ------------------------------------------------------------------
    def build_iron_condor_contract(
        self,
        short_call_strike: float,
        long_call_strike:  float,
        short_put_strike:  float,
        long_put_strike:   float,
        expiry:            str,     # YYYYMMDD
        n_contracts:       int = 1,
    ):
        """Build a 4-leg SPX iron-condor combo contract.

        Returns (contract, order) tuple ready for placeOrder().

        Note: SPX options are European-style, cash-settled — no assignment risk.
        """
        try:
            from ib_async import Contract, ComboLeg, Order
        except ImportError:
            raise ImportError("ib_async not installed")

        if not self._connected or self._ib is None:
            raise RuntimeError("Not connected to IBKR")

        legs_data = [
            (short_call_strike, "C", "SELL"),
            (long_call_strike,  "C", "BUY"),
            (short_put_strike,  "P", "SELL"),
            (long_put_strike,   "P", "BUY"),
        ]

        combo_legs: List[ComboLeg] = []
        for strike, right, action in legs_data:
            contract = Contract(
                symbol=SPX_SYMBOL,
                secType=SPX_SECTYPE,
                exchange=SPX_EXCH,
                currency=SPX_CURR,
                lastTradeDateOrContractMonth=expiry,
                strike=strike,
                right=right,
            )
            details = self._ib.reqContractDetails(contract)
            if not details:
                raise ValueError(f"No contract found: {SPX_SYMBOL} {right} {strike} {expiry}")
            con_id = details[0].contract.conId
            leg = ComboLeg(
                conId=con_id,
                ratio=1,
                action=action,
                exchange=SPX_EXCH,
            )
            combo_legs.append(leg)

        combo = Contract(
            symbol=SPX_SYMBOL,
            secType="BAG",
            exchange=SPX_EXCH,
            currency=SPX_CURR,
            comboLegs=combo_legs,
        )

        return combo, n_contracts

    # ------------------------------------------------------------------
    def place_condor_order(
        self,
        combo_contract,
        credit_limit:  float,
        n_contracts:   int = 1,
        timeout:       int = FILL_TIMEOUT_SEC,
    ) -> Dict[str, Any]:
        """Place a limit order for the iron condor at *credit_limit* net credit.

        Waits up to *timeout* seconds for a fill, then cancels.

        Returns a trade record dict.
        """
        try:
            from ib_async import LimitOrder
        except ImportError:
            raise ImportError("ib_async not installed")

        if not self._connected:
            raise RuntimeError("Not connected to IBKR")

        order = LimitOrder(
            action="SELL",          # sell the spread for a credit
            totalQuantity=n_contracts,
            lmtPrice=round(credit_limit, 2),
        )

        record = {
            "timestamp":     datetime.utcnow().isoformat() + "Z",
            "action":        "PLACE_ORDER",
            "mode":          "PAPER" if self.paper else "LIVE",
            "credit_limit":  credit_limit,
            "n_contracts":   n_contracts,
            "status":        "PENDING",
            "fill_price":    None,
            "order_id":      None,
            "error":         None,
        }

        try:
            trade = self._ib.placeOrder(combo_contract, order)
            record["order_id"] = trade.order.orderId
            logger.info("Order placed: id=%s credit_limit=%.2f", trade.order.orderId, credit_limit)

            # Poll for fill
            deadline = time.time() + timeout
            while time.time() < deadline:
                self._ib.sleep(1)
                if trade.orderStatus.status == "Filled":
                    record["status"]     = "FILLED"
                    record["fill_price"] = trade.orderStatus.avgFillPrice
                    logger.info("Order %s FILLED at %.4f", trade.order.orderId,
                                record["fill_price"])
                    break
                if trade.orderStatus.status in ("Cancelled", "ApiCancelled"):
                    record["status"] = "CANCELLED"
                    break
            else:
                # Timeout — cancel
                self._ib.cancelOrder(order)
                record["status"] = "TIMEOUT_CANCELLED"
                logger.warning("Order %s not filled within %ds — cancelled",
                               trade.order.orderId, timeout)

        except Exception as exc:
            record["status"] = "ERROR"
            record["error"]  = str(exc)
            logger.error("placeOrder failed: %s", exc)

        _log_trade(record, self.output_dir)
        return record

    # ------------------------------------------------------------------
    def monitor_position(self, order_id: int) -> Dict[str, Any]:
        """Return current fill status for *order_id*."""
        if not self._connected:
            return {"order_id": order_id, "status": "DISCONNECTED"}

        trades = self._ib.trades()
        for t in trades:
            if t.order.orderId == order_id:
                return {
                    "order_id":   order_id,
                    "status":     t.orderStatus.status,
                    "filled":     t.orderStatus.filled,
                    "remaining":  t.orderStatus.remaining,
                    "avg_price":  t.orderStatus.avgFillPrice,
                }
        return {"order_id": order_id, "status": "NOT_FOUND"}

    # ------------------------------------------------------------------
    def close_position(
        self,
        combo_contract,
        debit_limit:  float,
        n_contracts:  int = 1,
        timeout:      int = FILL_TIMEOUT_SEC,
    ) -> Dict[str, Any]:
        """Close an existing condor by buying back at *debit_limit*."""
        try:
            from ib_async import LimitOrder
        except ImportError:
            raise ImportError("ib_async not installed")

        order = LimitOrder(
            action="BUY",
            totalQuantity=n_contracts,
            lmtPrice=round(debit_limit, 2),
        )

        record = {
            "timestamp":    datetime.utcnow().isoformat() + "Z",
            "action":       "CLOSE_POSITION",
            "debit_limit":  debit_limit,
            "n_contracts":  n_contracts,
            "status":       "PENDING",
            "fill_price":   None,
        }

        try:
            trade = self._ib.placeOrder(combo_contract, order)
            deadline = time.time() + timeout
            while time.time() < deadline:
                self._ib.sleep(1)
                if trade.orderStatus.status == "Filled":
                    record["status"]     = "FILLED"
                    record["fill_price"] = trade.orderStatus.avgFillPrice
                    break
            else:
                self._ib.cancelOrder(order)
                record["status"] = "TIMEOUT_CANCELLED"

        except Exception as exc:
            record["status"] = "ERROR"
            record["error"]  = str(exc)

        _log_trade(record, self.output_dir)
        return record


# ---------------------------------------------------------------------------
# Convenience: build condor from FullSignal
# ---------------------------------------------------------------------------

def build_condor_from_signal(signal, broker: IBKRBroker, dte: int = 0):
    """Extract strikes from FullSignal and build the combo contract.

    Safety checks applied before calling broker.build_iron_condor_contract().
    """
    # Kill switch: regime RED
    regime = getattr(signal, "regime", "RED")
    if regime == "RED":
        raise ValueError("Regime is RED — execution blocked by safety gate")

    # Kill switch: degraded data quality
    quality = getattr(signal, "data_quality", "DEGRADED")
    if quality == "DEGRADED":
        raise ValueError("Data quality DEGRADED — kill switch activated")

    # Extract strikes
    sc = getattr(signal, "ic_short_call", None)
    lc = getattr(signal, "ic_long_call",  None)
    sp = getattr(signal, "ic_short_put",  None)
    lp = getattr(signal, "ic_long_put",   None)

    if any(v is None for v in (sc, lc, sp, lp)):
        raise ValueError(f"Missing strike levels: short_call={sc}, long_call={lc}, "
                         f"short_put={sp}, long_put={lp}")

    n = MAX_CONTRACTS
    if regime == "YELLOW":
        n = max(1, n // 2)
        logger.info("YELLOW regime: halving position size to %d contract(s)", n)

    expiry = _next_expiry_dte(dte)

    contract, n_used = broker.build_iron_condor_contract(
        short_call_strike=round(sc, 0),
        long_call_strike=round(lc, 0),
        short_put_strike=round(sp, 0),
        long_put_strike=round(lp, 0),
        expiry=expiry,
        n_contracts=n,
    )
    return contract, n_used
