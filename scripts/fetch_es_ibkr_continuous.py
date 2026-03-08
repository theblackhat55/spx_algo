#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from ib_async import IB, Contract, util


OUTFILE = Path("data/raw/es_ibkr_cont_daily.parquet")


def make_contfut_contract() -> Contract:
    c = Contract()
    c.symbol = "ES"
    c.secType = "CONTFUT"
    c.exchange = "CME"
    c.currency = "USD"
    return c


async def main() -> None:
    load_dotenv()

    host = os.getenv("IBKR_HOST", "127.0.0.1")
    port = int(os.getenv("IBKR_PORT", "7497"))
    client_id = int(os.getenv("IBKR_CLIENT_ID", "1"))

    ib = IB()
    print(f"Connecting to IBKR at {host}:{port} clientId={client_id} ...")
    await ib.connectAsync(host=host, port=port, clientId=client_id, timeout=10)

    contract = make_contfut_contract()
    details = await ib.reqContractDetailsAsync(contract)
    if not details:
        raise RuntimeError("No contract details returned for ES CONTFUT")

    print("Resolved contract:", details[0].contract)

    # IMPORTANT: endDateTime must be empty string for continuous futures
    bars = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime="",
        durationStr="10 Y",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=1,
        keepUpToDate=False,
    )

    if not bars:
        raise RuntimeError("No historical bars returned from IBKR")

    df = util.df(bars)
    if df.empty:
        raise RuntimeError("Historical bars converted to empty DataFrame")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"date": "Date"})
        df = df.set_index("Date")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    keep = [c for c in ["open", "high", "low", "close", "volume", "average", "barCount"] if c in df.columns]
    df = df[keep].copy()

    rename = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "average": "Average",
        "barCount": "BarCount",
    }
    df = df.rename(columns=rename)

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTFILE)

    print(f"Saved IBKR continuous ES daily data to: {OUTFILE}")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(df.tail(5))

    ib.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
