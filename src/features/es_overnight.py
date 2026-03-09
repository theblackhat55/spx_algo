from __future__ import annotations

import pandas as pd


def build_es_overnight_features_from_5m(es_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily overnight ES features from 5-minute bars.

    Assumptions:
    - input index is timezone-naive or timezone-aware timestamps
    - columns include Open/High/Low/Close
    - overnight window approximated as 16:00 previous day -> 09:30 current day
    """
    df = es_5m.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required intraday columns: {sorted(missing)}")

    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)

    df.index = idx
    df = df.sort_index()

    rows = []
    trading_dates = sorted(pd.Index(df.index.date).unique())

    for d in trading_dates:
        current = pd.Timestamp(d)
        overnight_start = (current - pd.Timedelta(days=1)).replace(hour=16, minute=0, second=0)
        overnight_end = current.replace(hour=9, minute=30, second=0)

        window = df.loc[(df.index > overnight_start) & (df.index <= overnight_end)].copy()
        if window.empty:
            continue

        prev_session = df.loc[df.index <= overnight_start]
        if prev_session.empty:
            continue

        prev_close = float(prev_session["Close"].iloc[-1])
        first_open = float(window["Open"].iloc[0])
        last_close = float(window["Close"].iloc[-1])
        high = float(window["High"].max())
        low = float(window["Low"].min())

        rows.append(
            {
                "Date": current.normalize(),
                "es_overnight_gap_pct": (first_open / prev_close) - 1.0,
                "es_overnight_ret": (last_close / prev_close) - 1.0,
                "es_overnight_range_pct": (high - low) / prev_close,
                "es_overnight_high_from_prev_close": (high / prev_close) - 1.0,
                "es_overnight_low_from_prev_close": 1.0 - (low / prev_close),
                "es_overnight_close_vs_open": (last_close / first_open) - 1.0,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "es_overnight_gap_pct",
                "es_overnight_ret",
                "es_overnight_range_pct",
                "es_overnight_high_from_prev_close",
                "es_overnight_low_from_prev_close",
                "es_overnight_close_vs_open",
            ]
        )

    out["Date"] = pd.to_datetime(out["Date"])
    out = out.set_index("Date").sort_index()
    return out


def load_es_overnight_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()
