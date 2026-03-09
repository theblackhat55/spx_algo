#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import databento as db
import pandas as pd
from dotenv import load_dotenv


DEFAULT_DATASET = "GLBX.MDP3"
DEFAULT_SYMBOL = "ES.c.0"
DEFAULT_STYPE_IN = "continuous"
DEFAULT_SCHEMA = "ohlcv-1d"
DEFAULT_OUT_DIR = Path("output/analysis/databento")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Query Databento OHLC data for a specific date."
    )
    p.add_argument("--date", required=True, help="Date to query, e.g. 2026-03-04")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--stype-in", default=DEFAULT_STYPE_IN, dest="stype_in")
    p.add_argument("--schema", default=DEFAULT_SCHEMA, choices=["ohlcv-1d", "ohlcv-1m"])
    p.add_argument("--summary", action="store_true", help="For ohlcv-1m, print aggregated session summaries")
    p.add_argument("--out", default=None, help="Optional output JSON/CSV base path")
    return p.parse_args()


def _session_aggregate(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    out = {
        "open": float(df["open"].iloc[0]),
        "high": float(df["high"].max()),
        "low": float(df["low"].min()),
        "close": float(df["close"].iloc[-1]),
        "volume": float(df["volume"].sum()) if "volume" in df.columns else None,
        "rows": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
    }
    return out


def build_summary(df: pd.DataFrame, query_date: pd.Timestamp) -> dict:
    if df.empty:
        return {"date": str(query_date.date()), "full_day": {}, "overnight_preopen": {}, "cash_session_like": {}}

    # Use UTC windows that match the existing feature engineering conventions closely.
    # For a target cash date D:
    # - overnight/preopen-like window: [00:00, 14:29] UTC on D from the queried day slice
    # - cash-session-like window: [14:30, 21:00] UTC on D
    full_day = _session_aggregate(df)

    overnight_mask = (
        (df.index >= pd.Timestamp(query_date.date(), tz="UTC")) &
        (df.index < pd.Timestamp(query_date.date(), tz="UTC") + pd.Timedelta(hours=14, minutes=30))
    )
    cash_mask = (
        (df.index >= pd.Timestamp(query_date.date(), tz="UTC") + pd.Timedelta(hours=14, minutes=30)) &
        (df.index < pd.Timestamp(query_date.date(), tz="UTC") + pd.Timedelta(hours=21))
    )

    overnight_df = df.loc[overnight_mask]
    cash_df = df.loc[cash_mask]

    return {
        "date": str(query_date.date()),
        "full_day": full_day,
        "overnight_preopen": _session_aggregate(overnight_df),
        "cash_session_like": _session_aggregate(cash_df),
    }


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY not found in .env")

    query_date = pd.Timestamp(args.date)
    start = query_date.strftime("%Y-%m-%d")
    end = (query_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    client = db.Historical(api_key)

    data = client.timeseries.get_range(
        dataset=args.dataset,
        schema=args.schema,
        symbols=args.symbol,
        stype_in=args.stype_in,
        start=start,   # inclusive
        end=end,       # exclusive
    )

    df = data.to_df()
    if df.empty:
        print("No data returned.")
        return

    if "ts_event" in df.columns:
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
        df = df.set_index("ts_event")
    else:
        df.index = pd.to_datetime(df.index, utc=True)

    df = df.sort_index()

    print("\n=== DATABENTO DATE QUERY ===")
    print("date:", args.date)
    print("dataset:", args.dataset)
    print("symbol:", args.symbol)
    print("stype_in:", args.stype_in)
    print("schema:", args.schema)
    print("rows:", len(df))
    print("range:", df.index.min(), "->", df.index.max())
    print("columns:", list(df.columns))

    if args.schema == "ohlcv-1d":
        print("\nDaily OHLC row(s):")
        print(df)
    else:
        print("\nHead:")
        print(df.head(10))
        print("\nTail:")
        print(df.tail(10))

    summary = None
    if args.schema == "ohlcv-1m" and args.summary:
        summary = build_summary(df, query_date)
        print("\n=== SESSION SUMMARY ===")
        print(json.dumps(summary, indent=2))

    out_base = args.out
    if out_base:
        out_base = Path(out_base)
    else:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "daily" if args.schema == "ohlcv-1d" else "intraday"
        out_base = DEFAULT_OUT_DIR / f"databento_{args.symbol.replace('.', '_')}_{args.date}_{suffix}"

    if args.schema == "ohlcv-1d":
        csv_path = out_base.with_suffix(".csv")
        df.to_csv(csv_path)
        print(f"\nSaved CSV to: {csv_path}")
    else:
        parquet_path = out_base.with_suffix(".parquet")
        df.to_parquet(parquet_path)
        print(f"\nSaved parquet to: {parquet_path}")

        if summary is not None:
            summary_path = out_base.with_name(out_base.name + "_summary").with_suffix(".json")
            summary_path.write_text(json.dumps(summary, indent=2))
            print(f"Saved summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()
