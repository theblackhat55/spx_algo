#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import databento as db
import pandas as pd
from dotenv import load_dotenv


DEFAULT_START = "2024-01-01"
DEFAULT_END = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
DEFAULT_DATASET = "GLBX.MDP3"
DEFAULT_SCHEMA = "ohlcv-1m"
DEFAULT_SYMBOL = "ES.c.0"
DEFAULT_STYPE_IN = "continuous"
DEFAULT_OUT = "data/raw/es_databento_1m.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch ES 1-minute OHLCV from Databento.")
    p.add_argument("--start", default=DEFAULT_START, help="Start date/time, e.g. 2024-01-01")
    p.add_argument("--end", default=DEFAULT_END, help="End date/time, e.g. 2026-03-08")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--schema", default=DEFAULT_SCHEMA)
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--stype-in", default=DEFAULT_STYPE_IN, dest="stype_in")
    p.add_argument("--out", default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    load_dotenv(Path(".env"))
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY not found in .env")

    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = db.Historical(api_key)

    print("Requesting Databento data...")
    print(
        {
            "dataset": args.dataset,
            "schema": args.schema,
            "symbol": args.symbol,
            "stype_in": args.stype_in,
            "start": args.start,
            "end": args.end,
            "out": str(out_path),
        }
    )

    data = client.timeseries.get_range(
        dataset=args.dataset,
        schema=args.schema,
        symbols=args.symbol,
        stype_in=args.stype_in,
        start=args.start,
        end=args.end,
    )

    df = data.to_df()

    if df.empty:
        raise RuntimeError("No rows returned from Databento.")

    if "ts_event" in df.columns:
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
        df = df.set_index("ts_event")

    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    df.to_parquet(out_path)

    print(f"Saved Databento ES 1m data to: {out_path}")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df.index.min()} -> {df.index.max()}")
    print(f"Columns: {list(df.columns)}")
    print("\nHead:")
    print(df.head(5))
    print("\nTail:")
    print(df.tail(5))


if __name__ == "__main__":
    main()
