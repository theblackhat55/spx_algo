from __future__ import annotations

import pandas as pd


REQUIRED_ES_COLS = {"Open", "High", "Low", "Close"}


def _normalize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Flatten possible MultiIndex columns from yfinance
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    # Normalize common lowercase variants
    rename_map = {c: c.title() for c in out.columns if c.lower() in {"open", "high", "low", "close", "volume"}}
    out = out.rename(columns=rename_map)
    return out


def compute_es_features(es_df: pd.DataFrame, spx_df: pd.DataFrame | None = None) -> pd.DataFrame:
    es = _normalize_ohlc_columns(es_df)
    missing = REQUIRED_ES_COLS - set(es.columns)
    if missing:
        raise ValueError(f"ES data missing required columns: {sorted(missing)}")

    es = es.copy()
    es.index = pd.to_datetime(es.index)
    es = es.sort_index()

    feat = pd.DataFrame(index=es.index)

    # Basic ES daily structure
    feat["es_ret_1d"] = es["Close"].pct_change(1, fill_method=None)
    feat["es_ret_3d"] = es["Close"].pct_change(3)
    feat["es_ret_5d"] = es["Close"].pct_change(5)

    feat["es_open_gap_pct"] = (es["Open"] / es["Close"].shift(1)) - 1.0
    feat["es_intraday_ret"] = (es["Close"] / es["Open"]) - 1.0
    feat["es_range_pct"] = (es["High"] - es["Low"]) / es["Close"].shift(1)

    feat["es_high_from_open"] = (es["High"] / es["Open"]) - 1.0
    feat["es_low_from_open"] = 1.0 - (es["Low"] / es["Open"])

    # Rolling volatility/range context
    feat["es_realized_vol_5"] = feat["es_ret_1d"].rolling(5).std()
    feat["es_realized_vol_20"] = feat["es_ret_1d"].rolling(20).std()
    feat["es_avg_range_5"] = feat["es_range_pct"].rolling(5).mean()
    feat["es_avg_range_20"] = feat["es_range_pct"].rolling(20).mean()

    # Relative state
    feat["es_close_vs_ma5"] = (es["Close"] / es["Close"].rolling(5).mean()) - 1.0
    feat["es_close_vs_ma20"] = (es["Close"] / es["Close"].rolling(20).mean()) - 1.0

    if spx_df is not None and not spx_df.empty:
        spx = spx_df.copy()
        spx.index = pd.to_datetime(spx.index)
        spx = spx.sort_index()

        # Join SPX close for simple cross-market relationship features
        base = pd.DataFrame(index=feat.index)
        base["es_close"] = es["Close"]
        # support either Close or close if upstream changes
        spx_close_col = "Close" if "Close" in spx.columns else ("close" if "close" in spx.columns else None)
        if spx_close_col is not None:
            spx_close = spx[[spx_close_col]].rename(columns={spx_close_col: "spx_close"})
            base = base.join(spx_close, how="left")
            feat["es_minus_spx_ret_1d"] = feat["es_ret_1d"] - base["spx_close"].pct_change(1, fill_method=None)
            feat["es_spx_ratio_vs_ma20"] = (
                (base["es_close"] / base["spx_close"]) /
                (base["es_close"] / base["spx_close"]).rolling(20).mean()
            ) - 1.0

    return feat


def load_es_daily_features(es_path: str, spx_df: pd.DataFrame | None = None) -> pd.DataFrame:
    es = pd.read_parquet(es_path)
    if "Date" in es.columns:
        es["Date"] = pd.to_datetime(es["Date"])
        es = es.set_index("Date")
    es.index = pd.to_datetime(es.index)
    return compute_es_features(es, spx_df=spx_df)
