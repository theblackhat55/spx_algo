#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import databento as db
import pandas as pd
from dotenv import load_dotenv


DEFAULT_START = "2024-01-01"
DEFAULT_DATASET = "GLBX.MDP3"
DEFAULT_SCHEMA = "ohlcv-1m"
DEFAULT_SYMBOL = "ES.c.0"
DEFAULT_STYPE_IN = "continuous"
DEFAULT_OUT = "data/raw/es_databento_1m.parquet"


def _default_end() -> str:
    # Conservative default: recent enough for daily delta, but not "now".
    return (pd.Timestamp.utcnow().floor("min") - pd.Timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch ES 1-minute OHLCV from Databento.")
    p.add_argument("--start", default=DEFAULT_START, help="Start date/time, e.g. 2024-01-01")
    p.add_argument("--end", default=None, help="Exclusive end date/time, e.g. 2026-03-13T00:00:00Z")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--schema", default=DEFAULT_SCHEMA)
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--stype-in", default=DEFAULT_STYPE_IN, dest="stype_in")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--full-refresh", action="store_true", help="Ignore existing parquet and fetch full range")
    return p.parse_args()


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if "ts_event" in out.columns:
        out["ts_event"] = pd.to_datetime(out["ts_event"], utc=True)
        out = out.set_index("ts_event")

    out.index = pd.to_datetime(out.index, utc=True)
    out = out.sort_index()
    return out


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return _normalize_df(df)


def _dedupe_concat(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if old_df.empty and new_df.empty:
        return pd.DataFrame()
    if old_df.empty:
        merged = new_df.copy()
    elif new_df.empty:
        merged = old_df.copy()
    else:
        merged = pd.concat([old_df, new_df], axis=0)

    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def _to_utc_ts(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _extract_retry_end_from_error(msg: str) -> str | None:
    # Handles examples like:
    # "Try again with an end time before 2026-03-12T13:07:14.982315000Z."
    # "data available up to '2026-03-12 20:50:00+00:00'"
    patterns = [
        r"before\s+([0-9T:\-\.]+Z)",
        r"up to '([^']+)'",
    ]
    for pat in patterns:
        m = re.search(pat, msg)
        if m:
            ts = _to_utc_ts(m.group(1))
            return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    return None


def _fetch_range(client, *, dataset, schema, symbol, stype_in, start, end):
    return client.timeseries.get_range(
        dataset=dataset,
        schema=schema,
        symbols=symbol,
        stype_in=stype_in,
        start=start,
        end=end,
    )


def main() -> None:
    load_dotenv(Path(".env"))
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY not found in .env")

    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    end = args.end or _default_end()
    existing = pd.DataFrame()
    fetch_start = args.start

    if not args.full_refresh:
        existing = _load_existing(out_path)
        if not existing.empty:
            last_ts = existing.index.max()
            fetch_start = (last_ts + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("Existing rows:", len(existing))
    if not existing.empty:
        print("Existing range:", existing.index.min(), "->", existing.index.max())

    fetch_start_ts = _to_utc_ts(fetch_start)
    end_ts = _to_utc_ts(end)

    if fetch_start_ts >= end_ts:
        print("No delta to fetch; start >= end.")
        merged = existing.sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        if merged.empty:
            raise RuntimeError("No existing data and no delta window available.")
        merged.to_parquet(out_path)
        print(f"Saved ES 1m data to: {out_path}")
        print(f"Rows: {len(merged)}")
        print(f"Date range: {merged.index.min()} -> {merged.index.max()}")
        return

    client = db.Historical(api_key)

    request_info = {
        "dataset": args.dataset,
        "schema": args.schema,
        "symbol": args.symbol,
        "stype_in": args.stype_in,
        "start": fetch_start,
        "end": end,
        "out": str(out_path),
        "mode": "full-refresh" if args.full_refresh else "delta-append",
    }

    print("Requesting Databento data...")
    print(request_info)

    try:
        data = _fetch_range(
            client,
            dataset=args.dataset,
            schema=args.schema,
            symbol=args.symbol,
            stype_in=args.stype_in,
            start=fetch_start,
            end=end,
        )
    except Exception as exc:
        msg = str(exc)
        retry_end = _extract_retry_end_from_error(msg)
        if retry_end:
            retry_end_ts = _to_utc_ts(retry_end)
            if fetch_start_ts >= retry_end_ts:
                print("No permissible delta window remains after applying Databento limit.")
                merged = existing.sort_index()
                merged = merged[~merged.index.duplicated(keep="last")]
                if merged.empty:
                    raise
                merged.to_parquet(out_path)
                print(f"Saved ES 1m data to: {out_path}")
                print(f"Rows: {len(merged)}")
                print(f"Date range: {merged.index.min()} -> {merged.index.max()}")
                return

            print("Initial request failed; retrying with Databento-permitted end:", retry_end)
            data = _fetch_range(
                client,
                dataset=args.dataset,
                schema=args.schema,
                symbol=args.symbol,
                stype_in=args.stype_in,
                start=fetch_start,
                end=retry_end,
            )
        else:
            raise

    new_df = _normalize_df(data.to_df())

    print("New rows fetched:", len(new_df))
    if not new_df.empty:
        print("New range:", new_df.index.min(), "->", new_df.index.max())

    merged = _dedupe_concat(existing, new_df)

    if merged.empty:
        raise RuntimeError("No rows available after merge.")

    merged.to_parquet(out_path)

    print(f"Saved ES 1m data to: {out_path}")
    print(f"Rows: {len(merged)}")
    print(f"Date range: {merged.index.min()} -> {merged.index.max()}")
    print(f"Columns: {list(merged.columns)}")
    print("\nTail:")
    print(merged.tail(5))


if __name__ == "__main__":
    main()
