#!/usr/bin/env python3
"""
scripts/test_ibkr_data.py
==========================
Standalone integration test for the IBKR live data module.

Connects to IB Gateway, fetches SPX price, VIX, SPXW 0DTE option chain,
computes all Greeks summary metrics, and prints a formatted report.

Usage:
    /root/spx_algo/.venv/bin/python3 /root/spx_algo/scripts/test_ibkr_data.py

Exit codes:
    0 — Success (connected and fetched at least SPX + VIX)
    1 — Connection failed or critical error
    2 — Partial data (connected, but chain/Greeks unavailable)
"""
from __future__ import annotations

import logging
import sys
import os
from datetime import datetime

# ── Make sure project root is on the path ─────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Configure logging (INFO to stdout) ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_ibkr_data")

# Suppress noisy ib_insync internal logging
logging.getLogger("ib_insync").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)


def fmt(val, fmt_str: str = ".4f", fallback: str = "N/A") -> str:
    """Format a numeric value or return fallback string."""
    if val is None:
        return fallback
    try:
        return format(float(val), fmt_str)
    except (TypeError, ValueError):
        return str(val)


def print_section(title: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_row(label: str, value: str, unit: str = "") -> None:
    print(f"  {label:<35} {value:>12}  {unit}")


def main() -> int:
    print("=" * 60)
    print("  IBKR Live Data Module — Integration Test")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    print("=" * 60)

    ib = None
    exit_code = 0

    try:
        # ── Step 1: Load settings ──────────────────────────────────────────────
        print_section("1. Loading settings")
        try:
            from config.settings import (
                IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, IBKR_TIMEOUT
            )
        except Exception:
            IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, IBKR_TIMEOUT = \
                "127.0.0.1", 4002, 10, 10

        print_row("Host", IBKR_HOST)
        print_row("Port", str(IBKR_PORT))
        print_row("Client ID", str(IBKR_CLIENT_ID))
        print_row("Timeout", f"{IBKR_TIMEOUT}s")

        # ── Step 2: Connect ────────────────────────────────────────────────────
        print_section("2. Connecting to IB Gateway")
        from src.data.ibkr_fetcher import (
            connect_ib,
            disconnect_ib,
            fetch_spx_price,
            fetch_vix_price,
            fetch_spx_0dte_chain,
            compute_greeks_summary,
            fetch_ibkr_snapshot,
        )

        try:
            ib = connect_ib()
            print_row("Connection", "✓ CONNECTED")
            from ib_insync import IB
            server_ver = ib.client.serverVersion()
            accounts   = ib.managedAccounts()
            print_row("Server version", str(server_ver))
            print_row("Managed accounts", ", ".join(accounts))
        except ConnectionError as e:
            print(f"\n  ✗ FAILED to connect: {e}")
            print("  → Is IB Gateway running on 127.0.0.1:4002?")
            return 1

        # ── Step 3: Fetch SPX price ────────────────────────────────────────────
        print_section("3. SPX Index Price")
        spx = fetch_spx_price(ib)
        if spx is not None:
            print_row("SPX price", fmt(spx, ".2f"), "pts")
            print_row("Status", "✓ OK")
        else:
            print_row("SPX price", "N/A")
            print_row("Status", "⚠ No data (market may be closed)")

        # ── Step 4: Fetch VIX price ────────────────────────────────────────────
        print_section("4. VIX Index Price")
        vix = fetch_vix_price(ib)
        if vix is not None:
            print_row("VIX level", fmt(vix, ".2f"))
            print_row("Status", "✓ OK")
        else:
            print_row("VIX level", "N/A")
            print_row("Status", "⚠ No data (market may be closed)")

        if spx is None and vix is None:
            print("\n  ⚠ Both SPX and VIX returned no data.")
            print("  This usually means the market is closed or outside regular hours.")

        # ── Step 5: Fetch 0DTE option chain ───────────────────────────────────
        print_section("5. SPXW 0DTE Option Chain")
        chain = fetch_spx_0dte_chain(ib, spx)

        if chain.empty:
            print_row("Chain rows", "0")
            print_row("Status", "⚠ No chain data")
            print("  → 0DTE options may not be available outside market hours")
            print("  → Or today may not be an SPX expiry day")
            exit_code = 2
        else:
            n_strikes = chain["strike"].nunique()
            n_calls   = len(chain[chain["right"] == "C"])
            n_puts    = len(chain[chain["right"] == "P"])
            iv_valid  = chain["iv"].notna().sum()
            dlt_valid = chain["delta"].notna().sum()

            print_row("Total rows", str(len(chain)))
            print_row("Unique strikes", str(n_strikes))
            print_row("Calls", str(n_calls))
            print_row("Puts", str(n_puts))
            print_row("Rows with valid IV", str(iv_valid))
            print_row("Rows with valid delta", str(dlt_valid))

            # Print a mini chain sample (5 strikes around ATM)
            if spx:
                all_strikes = sorted(chain["strike"].unique())
                atm = min(all_strikes, key=lambda s: abs(s - spx))
                atm_idx = all_strikes.index(atm)
                sample_strikes = all_strikes[
                    max(0, atm_idx - 2) : min(len(all_strikes), atm_idx + 3)
                ]
                print(f"\n  {'Strike':>8}  {'R':>2}  {'Bid':>8}  {'Ask':>8}  "
                      f"{'Mid':>8}  {'IV':>8}  {'Delta':>8}  {'Gamma':>10}  {'OI':>8}")
                print("  " + "-" * 85)
                for s in sample_strikes:
                    for right in ("C", "P"):
                        row = chain[(chain["strike"] == s) & (chain["right"] == right)]
                        if row.empty:
                            continue
                        r = row.iloc[0]
                        print(
                            f"  {s:>8.1f}  {right:>2}  "
                            f"{fmt(r.get('bid'), '.2f'):>8}  "
                            f"{fmt(r.get('ask'), '.2f'):>8}  "
                            f"{fmt(r.get('mid'), '.2f'):>8}  "
                            f"{fmt(r.get('iv'), '.4f'):>8}  "
                            f"{fmt(r.get('delta'), '.4f'):>8}  "
                            f"{fmt(r.get('gamma'), '.6f'):>10}  "
                            f"{int(r.get('oi', 0)):>8}"
                        )

        # ── Step 6: Compute Greeks summary ─────────────────────────────────────
        print_section("6. Greeks Summary Metrics")
        if not chain.empty:
            greeks = compute_greeks_summary(chain, spx)
            print_row("ATM Implied Volatility", fmt(greeks.get("atm_iv"), ".4f"))
            print_row("IV Skew (25-delta put-call)", fmt(greeks.get("iv_skew_25d"), ".4f"))
            print_row("Put/Call Ratio (OI)", fmt(greeks.get("put_call_ratio"), ".4f"))
            print_row("GEX (net gamma exposure)", fmt(greeks.get("gex"), ",.0f"), "$")
            print_row("ATM Gamma", fmt(greeks.get("atm_gamma"), ".8f"))
            print_row("Call Wall (peak call OI)", fmt(greeks.get("call_wall"), ".0f"), "pts")
            print_row("Put Wall (peak put OI)", fmt(greeks.get("put_wall"), ".0f"), "pts")

            all_present = all(
                greeks.get(k) is not None
                for k in ["atm_iv", "iv_skew_25d", "put_call_ratio",
                           "gex", "atm_gamma", "call_wall", "put_wall"]
            )
            print_row("All metrics populated", "✓ YES" if all_present else "⚠ PARTIAL")
        else:
            print("  (skipped — no chain data)")

        # ── Step 7: Full snapshot via get_ibkr_features ───────────────────────
        print_section("7. get_ibkr_features() — pipeline integration test")
        disconnect_ib(ib)
        ib = None  # Will reconnect inside get_ibkr_features

        from src.data.ibkr_live_features import get_ibkr_features
        features = get_ibkr_features(spx_price=spx)

        for key, val in features.items():
            if key == "ibkr_data_quality":
                print_row(key, str(val))
            elif isinstance(val, float):
                print_row(key, fmt(val, ".4f"))
            else:
                print_row(key, str(val) if val is not None else "N/A")

        quality = features.get("ibkr_data_quality", "ERROR")
        print(f"\n  Data quality: {quality}")

        # ── Final summary ──────────────────────────────────────────────────────
        print_section("Test Summary")
        checks = {
            "IB Gateway connection":   True,
            "SPX price":               spx is not None,
            "VIX price":               vix is not None,
            "0DTE chain returned":     not chain.empty,
            "ibkr_features() works":   features.get("ibkr_data_quality") != "ERROR",
        }

        all_ok = True
        critical_fail = False
        for check, passed in checks.items():
            symbol = "✓" if passed else "✗"
            print(f"  {symbol}  {check}")
            if not passed:
                all_ok = False
                if check in ("IB Gateway connection",):
                    critical_fail = True

        if critical_fail:
            print("\n  ✗ CRITICAL FAILURE — see above\n")
            return 1
        elif not all_ok:
            print("\n  ⚠ PARTIAL SUCCESS — some data unavailable (normal outside market hours)\n")
            return 2 if exit_code == 2 else 0
        else:
            print("\n  ✓ ALL CHECKS PASSED\n")
            return 0

    except Exception as exc:
        logger.error("Unexpected error in test: %s", exc, exc_info=True)
        print(f"\n  ✗ UNEXPECTED ERROR: {exc}\n")
        return 1

    finally:
        if ib is not None:
            try:
                from src.data.ibkr_fetcher import disconnect_ib
                disconnect_ib(ib)
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
