#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.es_overnight import build_es_overnight_features_from_5m

INFILE = Path("data/raw/es_5m_recent.parquet")
OUTFILE = Path("data/processed/es_overnight_features.parquet")


def main() -> None:
    if not INFILE.exists():
        raise FileNotFoundError(f"Missing intraday file: {INFILE}")

    df = pd.read_parquet(INFILE)
    feat = build_es_overnight_features_from_5m(df)

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(OUTFILE)

    print(f"Saved ES overnight features to: {OUTFILE}")
    print(f"Rows: {len(feat)}")
    if not feat.empty:
        print(f"Date range: {feat.index.min().date()} -> {feat.index.max().date()}")
        print(feat.tail(5))


if __name__ == "__main__":
    main()
