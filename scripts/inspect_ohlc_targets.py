#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.targets.ohlc_targets import build_ohlc_component_targets


def main():
    spx_path = REPO_ROOT / "data" / "raw" / "spx_daily.parquet"
    if not spx_path.exists():
        raise FileNotFoundError(f"SPX parquet not found: {spx_path}")

    spx = pd.read_parquet(spx_path)
    spx.index = pd.to_datetime(spx.index)

    out = build_ohlc_component_targets(spx, dropna=False)

    print("\n=== OHLC TARGETS HEAD ===")
    print(out.head(5))
    print("\n=== OHLC TARGETS TAIL ===")
    print(out.tail(5))
    print("\n=== NON-NULL COUNTS ===")
    print(out.notnull().sum())
    print("\n=== SUMMARY ===")
    print(out.describe().T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
