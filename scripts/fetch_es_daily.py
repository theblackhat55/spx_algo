#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import yfinance as yf
import pandas as pd


OUT = Path("data/raw/es_daily.parquet")
TICKER = "ES=F"
START = "2000-01-01"


def main() -> None:
    df = yf.download(TICKER, start=START, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("Failed to download ES daily data from Yahoo Finance")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)

    print(f"Saved ES daily data to: {OUT}")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(df.tail(3)[["Open", "High", "Low", "Close"]])


if __name__ == "__main__":
    main()
