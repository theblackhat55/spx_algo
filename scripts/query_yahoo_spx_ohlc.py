#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_OUTDIR = Path("output/analysis/yahoo_spx")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Query SPX OHLC from Yahoo Finance (^GSPC) for a single date or a date range."
    )
    p.add_argument("--date", help="Single trading date, e.g. 2026-03-04")
    p.add_argument("--start-date", help="Range start date, e.g. 2026-03-01")
    p.add_argument("--end-date", help="Range end date, e.g. 2026-03-06")
    p.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help=f"Output directory (default: {DEFAULT_OUTDIR})",
    )
    p.add_argument(
        "--auto-adjust",
        action="store_true",
        help="Pass auto_adjust=True to yfinance (default False).",
    )
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp, bool]:
    has_single = bool(args.date)
    has_range = bool(args.start_date or args.end_date)

    if has_single and has_range:
        raise SystemExit("Use either --date OR --start-date/--end-date, not both.")

    if has_single:
        d = pd.Timestamp(args.date).normalize()
        return d, d, True

    if args.start_date and args.end_date:
        s = pd.Timestamp(args.start_date).normalize()
        e = pd.Timestamp(args.end_date).normalize()
        if e < s:
            raise SystemExit("--end-date must be >= --start-date")
        return s, e, False

    raise SystemExit("Provide either --date YYYY-MM-DD OR --start-date YYYY-MM-DD --end-date YYYY-MM-DD")


def _download_gspc(start_date: pd.Timestamp, end_date: pd.Timestamp, auto_adjust: bool) -> pd.DataFrame:
    # Yahoo end is effectively exclusive in practice, so add 1 day to include end_date
    end_exclusive = end_date + pd.Timedelta(days=1)

    df = yf.download(
        "^GSPC",
        start=start_date.strftime("%Y-%m-%d"),
        end=end_exclusive.strftime("%Y-%m-%d"),
        auto_adjust=auto_adjust,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns if returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).strip("_")
            for col in df.columns.to_flat_index()
        ]

    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc.startswith("open"):
            rename_map[c] = "open"
        elif lc.startswith("high"):
            rename_map[c] = "high"
        elif lc.startswith("low"):
            rename_map[c] = "low"
        elif lc.startswith("close") and "adj" not in lc:
            rename_map[c] = "close"
        elif lc.startswith("adj close") or lc.startswith("adj_close"):
            rename_map[c] = "adj_close"
        elif lc.startswith("volume"):
            rename_map[c] = "volume"

    df = df.rename(columns=rename_map)

    keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    df = df[keep].copy()

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"

    return df.sort_index()


def _single_day_summary(df: pd.DataFrame, requested_date: pd.Timestamp) -> dict:
    if requested_date in df.index:
        row = df.loc[requested_date]
        out = {
            "requested_date": requested_date.strftime("%Y-%m-%d"),
            "found": True,
            "open": None if pd.isna(row.get("open")) else float(row.get("open")),
            "high": None if pd.isna(row.get("high")) else float(row.get("high")),
            "low": None if pd.isna(row.get("low")) else float(row.get("low")),
            "close": None if pd.isna(row.get("close")) else float(row.get("close")),
            "adj_close": None if "adj_close" not in row.index or pd.isna(row.get("adj_close")) else float(row.get("adj_close")),
            "volume": None if "volume" not in row.index or pd.isna(row.get("volume")) else float(row.get("volume")),
        }
    else:
        out = {
            "requested_date": requested_date.strftime("%Y-%m-%d"),
            "found": False,
            "reason": "Date not present in Yahoo response (likely weekend/holiday/non-trading day).",
        }
    return out


def main() -> None:
    args = _parse_args()
    start_date, end_date, is_single = _validate_args(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = _download_gspc(start_date, end_date, auto_adjust=args.auto_adjust)

    if df.empty:
        print("No data returned from Yahoo Finance for ^GSPC.")
        raise SystemExit(1)

    if is_single:
        summary = _single_day_summary(df, start_date)
        stem = f"{start_date.strftime('%Y-%m-%d')}_yahoo_spx"
        csv_path = outdir / f"{stem}.csv"
        json_path = outdir / f"{stem}.json"
    else:
        summary = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "rows": int(len(df)),
            "date_min": df.index.min().strftime("%Y-%m-%d"),
            "date_max": df.index.max().strftime("%Y-%m-%d"),
            "columns": list(df.columns),
        }
        stem = f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_yahoo_spx"
        csv_path = outdir / f"{stem}.csv"
        json_path = outdir / f"{stem}.json"

    df.to_csv(csv_path)
    json_path.write_text(json.dumps(summary, indent=2))

    print("Saved CSV :", csv_path)
    print("Saved JSON:", json_path)
    print()

    if is_single:
        print("=== YAHOO SPX SINGLE-DAY CHECK ===")
        print(json.dumps(summary, indent=2))
        print()
        print(df.tail())
    else:
        print("=== YAHOO SPX RANGE CHECK ===")
        print(json.dumps(summary, indent=2))
        print()
        print(df.head())
        print()
        print(df.tail())


if __name__ == "__main__":
    main()
