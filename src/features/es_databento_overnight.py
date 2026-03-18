from __future__ import annotations

import pandas as pd


OVERNIGHT_START_UTC_HOUR = 23  # previous calendar day 23:00 UTC
PREOPEN_END_HOUR = 12
PREOPEN_END_MINUTE = 29
PREOPEN_LAST_60_START = (11, 30)
PREOPEN_LAST_30_START = (12, 0)


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out = out.sort_index()
    return out


def _session_date(ts: pd.Timestamp) -> pd.Timestamp.date:
    # Bars from 23:00 UTC onward belong to the *next* SPX cash date.
    if ts.hour >= OVERNIGHT_START_UTC_HOUR:
        return (ts + pd.Timedelta(days=1)).date()
    return ts.date()


def build_es_databento_overnight_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_utc_index(df_1m)

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df[list(required)].copy()
    work["session_date"] = [ _session_date(ts) for ts in work.index ]

    rows = []
    for session_date, g in work.groupby("session_date", sort=True):
        # keep only overnight + preopen portion:
        # previous day 23:00 UTC through current day 14:29 UTC
        mask = (
            (g.index.hour >= OVERNIGHT_START_UTC_HOUR)
            | (g.index.hour < PREOPEN_END_HOUR)
            | (
                (g.index.hour == PREOPEN_END_HOUR)
                & (g.index.minute <= PREOPEN_END_MINUTE)
            )
        )
        s = g.loc[mask].copy()
        if s.empty:
            continue

        overnight_open = float(s["open"].iloc[0])
        overnight_close = float(s["close"].iloc[-1])
        overnight_high = float(s["high"].max())
        overnight_low = float(s["low"].min())
        overnight_volume = float(s["volume"].sum())

        overnight_ret = (overnight_close / overnight_open) - 1.0 if overnight_open else 0.0
        overnight_range_pct = ((overnight_high - overnight_low) / overnight_open) if overnight_open else 0.0

        preopen_60 = s.loc[
            (
                (s.index.hour == PREOPEN_LAST_60_START[0])
                & (s.index.minute >= PREOPEN_LAST_60_START[1])
            )
            | (
                (s.index.hour == PREOPEN_END_HOUR)
                & (s.index.minute <= PREOPEN_END_MINUTE)
            )
        ]
        preopen_30 = s.loc[
            (
                (s.index.hour == PREOPEN_LAST_30_START[0])
                & (s.index.minute >= PREOPEN_LAST_30_START[1])
            )
            | (
                (s.index.hour == PREOPEN_END_HOUR)
                & (s.index.minute <= PREOPEN_END_MINUTE)
            )
        ]

        if preopen_60.empty:
            preopen_ret_last_60m = 0.0
            preopen_range_last_60m = 0.0
        else:
            p60_open = float(preopen_60["open"].iloc[0])
            p60_close = float(preopen_60["close"].iloc[-1])
            p60_high = float(preopen_60["high"].max())
            p60_low = float(preopen_60["low"].min())
            preopen_ret_last_60m = (p60_close / p60_open) - 1.0 if p60_open else 0.0
            preopen_range_last_60m = ((p60_high - p60_low) / p60_open) if p60_open else 0.0

        if preopen_30.empty:
            preopen_ret_last_30m = 0.0
        else:
            p30_open = float(preopen_30["open"].iloc[0])
            p30_close = float(preopen_30["close"].iloc[-1])
            preopen_ret_last_30m = (p30_close / p30_open) - 1.0 if p30_open else 0.0

        overnight_mid = (overnight_high + overnight_low) / 2.0
        preopen_close_vs_overnight_mid = ((overnight_close / overnight_mid) - 1.0) if overnight_mid else 0.0

        rows.append(
            {
                "date": pd.Timestamp(session_date),
                "es_overnight_open": overnight_open,
                "es_overnight_close": overnight_close,
                "es_overnight_high": overnight_high,
                "es_overnight_low": overnight_low,
                "es_overnight_volume": overnight_volume,
                "es_overnight_ret": overnight_ret,
                "es_overnight_range_pct": overnight_range_pct,
                "es_preopen_ret_last_60m": preopen_ret_last_60m,
                "es_preopen_ret_last_30m": preopen_ret_last_30m,
                "es_preopen_range_last_60m": preopen_range_last_60m,
                "es_preopen_close_vs_overnight_mid": preopen_close_vs_overnight_mid,
            }
        )

    out = pd.DataFrame(rows).set_index("date").sort_index()

    out["es_overnight_high_from_prev_close"] = out["es_overnight_high"].shift(0)
    out["es_overnight_low_from_prev_close"] = out["es_overnight_low"].shift(0)

    # convert absolute high/low to returns from previous overnight close proxy
    prev_close = out["es_overnight_close"].shift(1)
    valid = prev_close.notna() & (prev_close != 0)
    out.loc[valid, "es_overnight_high_from_prev_close"] = (
        out.loc[valid, "es_overnight_high"] / prev_close.loc[valid] - 1.0
    )
    out.loc[valid, "es_overnight_low_from_prev_close"] = (
        out.loc[valid, "es_overnight_low"] / prev_close.loc[valid] - 1.0
    )
    out.loc[~valid, "es_overnight_high_from_prev_close"] = 0.0
    out.loc[~valid, "es_overnight_low_from_prev_close"] = 0.0

    return out


def load_es_databento_overnight_features(path: str | None = None) -> pd.DataFrame:
    p = path or "data/processed/es_databento_overnight_features.parquet"
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df
