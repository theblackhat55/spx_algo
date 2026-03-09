#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_OUTDIR = Path("output/analysis/forecast_vs_actual")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare a forecast JSON against actual SPX (^GSPC) OHLC from Yahoo Finance."
    )
    p.add_argument(
        "--forecast-json",
        required=True,
        help="Path to forecast JSON, e.g. output/forecasts/backtests/2026-03-04_from_2026-03-03_gap_augmented_hybrid_ohlc_forecast.json",
    )
    p.add_argument(
        "--date",
        help="Optional explicit comparison date YYYY-MM-DD. If omitted, uses forecast_for_date from JSON.",
    )
    p.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help=f"Directory for comparison reports (default: {DEFAULT_OUTDIR})",
    )
    p.add_argument(
        "--auto-adjust",
        action="store_true",
        help="Pass auto_adjust=True to yfinance (default False). For OHLC comparison, default False is usually preferred.",
    )
    return p.parse_args()


def _load_forecast(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Forecast JSON not found: {path}")
    return json.loads(path.read_text())


def _infer_compare_date(forecast: dict, explicit_date: str | None) -> pd.Timestamp:
    if explicit_date:
        return pd.Timestamp(explicit_date).normalize()

    for key in ["forecast_for_date", "date", "forecast_date"]:
        if key in forecast:
            return pd.Timestamp(forecast[key]).normalize()

    raise SystemExit(
        "Could not infer comparison date from forecast JSON. Pass --date YYYY-MM-DD."
    )


def _extract_predicted_ohlc(forecast: dict) -> dict:
    if "predicted_ohlc" in forecast and isinstance(forecast["predicted_ohlc"], dict):
        src = forecast["predicted_ohlc"]
    else:
        src = forecast

    aliases = {
        "open": ["open", "pred_open", "predicted_open"],
        "high": ["high", "pred_high", "predicted_high"],
        "low": ["low", "pred_low", "predicted_low"],
        "close": ["close", "pred_close", "predicted_close"],
    }

    out = {}
    for target, names in aliases.items():
        value = None
        for name in names:
            if name in src:
                value = src[name]
                break
        if value is None:
            raise SystemExit(f"Could not find predicted {target} in forecast JSON.")
        out[target] = float(value)

    return out


def _fetch_yahoo_spx_for_date(compare_date: pd.Timestamp, auto_adjust: bool) -> dict:
    end_exclusive = compare_date + pd.Timedelta(days=1)

    df = yf.download(
        "^GSPC",
        start=compare_date.strftime("%Y-%m-%d"),
        end=end_exclusive.strftime("%Y-%m-%d"),
        auto_adjust=auto_adjust,
        progress=False,
    )

    if df is None or df.empty:
        raise SystemExit(
            f"No Yahoo Finance data returned for ^GSPC on {compare_date.strftime('%Y-%m-%d')}"
        )

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
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"

    if compare_date not in df.index:
        raise SystemExit(
            f"Requested date {compare_date.strftime('%Y-%m-%d')} not found in Yahoo response."
        )

    row = df.loc[compare_date]

    return {
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "adj_close": float(row["adj_close"]) if "adj_close" in row.index and pd.notna(row["adj_close"]) else None,
        "volume": float(row["volume"]) if "volume" in row.index and pd.notna(row["volume"]) else None,
    }


def _build_comparison(compare_date: pd.Timestamp, predicted: dict, actual: dict, forecast_path: Path) -> dict:
    errors = {}
    abs_errors = {}
    ape = {}

    for key in ["open", "high", "low", "close"]:
        err = predicted[key] - actual[key]
        errors[key] = err
        abs_errors[key] = abs(err)
        ape[key] = abs(err) / abs(actual[key]) if actual[key] not in (0, None) else None

    report = {
        "date": compare_date.strftime("%Y-%m-%d"),
        "forecast_json": str(forecast_path),
        "predicted_ohlc": predicted,
        "actual_spx_ohlc": {
            "open": actual["open"],
            "high": actual["high"],
            "low": actual["low"],
            "close": actual["close"],
        },
        "actual_adj_close": actual.get("adj_close"),
        "actual_volume": actual.get("volume"),
        "errors_pred_minus_actual": errors,
        "abs_errors": abs_errors,
        "mape_by_field": ape,
        "mae_mean_ohlc": sum(abs_errors.values()) / 4.0,
    }
    return report


def main() -> None:
    args = _parse_args()
    forecast_path = Path(args.forecast_json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    forecast = _load_forecast(forecast_path)
    compare_date = _infer_compare_date(forecast, args.date)
    predicted = _extract_predicted_ohlc(forecast)
    actual = _fetch_yahoo_spx_for_date(compare_date, auto_adjust=args.auto_adjust)

    report = _build_comparison(compare_date, predicted, actual, forecast_path)

    out_path = outdir / f"{compare_date.strftime('%Y-%m-%d')}_forecast_vs_yahoo_spx.json"
    out_path.write_text(json.dumps(report, indent=2))

    print("Saved comparison report:", out_path)
    print()
    print("=== FORECAST VS YAHOO SPX CHECK ===")
    print("date:", report["date"])
    print("forecast_json:", report["forecast_json"])
    print()
    print("Predicted OHLC:")
    for k, v in report["predicted_ohlc"].items():
        print(f"  {k}: {v:.4f}")
    print()
    print("Actual SPX OHLC:")
    for k, v in report["actual_spx_ohlc"].items():
        print(f"  {k}: {v:.4f}")
    print()
    print("Errors (pred - actual):")
    for k, v in report["errors_pred_minus_actual"].items():
        print(f"  {k}: {v:.4f}")
    print()
    print("Absolute errors:")
    for k, v in report["abs_errors"].items():
        print(f"  {k}: {v:.4f}")
    print()
    print(f"Mean OHLC MAE: {report['mae_mean_ohlc']:.4f}")


if __name__ == "__main__":
    main()
