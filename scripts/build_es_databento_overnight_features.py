#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.es_databento_overnight import build_es_databento_overnight_features


RAW_PATH = Path("data/raw/es_databento_1m.parquet")
OUT_PATH = Path("data/processed/es_databento_overnight_features.parquet")


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw Databento file: {RAW_PATH}")

    df = pd.read_parquet(RAW_PATH)
    if "ts_event" in df.columns:
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
        df = df.set_index("ts_event")
    else:
        df.index = pd.to_datetime(df.index, utc=True)

    feat = build_es_databento_overnight_features(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(OUT_PATH)

    print(f"Saved Databento overnight features to: {OUT_PATH}")
    print(f"Rows: {len(feat)}")
    print(f"Date range: {feat.index.min().date()} -> {feat.index.max().date()}")
    print(f"Columns: {list(feat.columns)}")
    print("\nHead:")
    print(feat.head(5))
    print("\nTail:")
    print(feat.tail(5))


if __name__ == "__main__":
    main()
