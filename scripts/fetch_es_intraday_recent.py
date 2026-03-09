#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

OUT = Path("data/raw/es_5m_recent.parquet")
TICKER = "ES=F"


def main() -> None:
    # Yahoo recent intraday only
    df = yf.download(TICKER, period="60d", interval="5m", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("Failed to download recent ES 5m data")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df.index = pd.to_datetime(df.index)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)

    print(f"Saved ES 5m recent data to: {OUT}")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df.index.min()} -> {df.index.max()}")
    print(df.tail(5)[["Open", "High", "Low", "Close"]])


if __name__ == "__main__":
    main()
