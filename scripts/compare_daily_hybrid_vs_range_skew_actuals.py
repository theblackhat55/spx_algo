#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yfinance as yf


DEFAULT_HYBRID = Path("output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json")
DEFAULT_RANGE_SKEW = Path("output/forecasts/latest_gap_augmented_range_skew_forecast.json")
DEFAULT_OUTDIR = Path("output/reports/daily_forecast_comparison")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text())


def _extract_predicted_ohlc(payload: Dict[str, Any]) -> Dict[str, float]:
    candidate_keys = [
        "predicted_ohlc",
        "final_predicted_ohlc",
        "hybrid_predicted_ohlc",
        "predicted_values",
    ]

    for key in candidate_keys:
        if key in payload and isinstance(payload[key], dict):
            d = payload[key]

            if all(k in d for k in ["open", "high", "low", "close"]):
                return {k: float(d[k]) for k in ["open", "high", "low", "close"]}

            if all(k in d for k in ["pred_open", "pred_high", "pred_low", "pred_close"]):
                return {
                    "open": float(d["pred_open"]),
                    "high": float(d["pred_high"]),
                    "low": float(d["pred_low"]),
                    "close": float(d["pred_close"]),
                }

    if all(k in payload for k in ["pred_open", "pred_high", "pred_low", "pred_close"]):
        return {
            "open": float(payload["pred_open"]),
            "high": float(payload["pred_high"]),
            "low": float(payload["pred_low"]),
            "close": float(payload["pred_close"]),
        }

    if all(k in payload for k in ["open", "high", "low", "close"]):
        return {
            "open": float(payload["open"]),
            "high": float(payload["high"]),
            "low": float(payload["low"]),
            "close": float(payload["close"]),
        }

    raise KeyError(
        "Could not find supported predicted OHLC fields in forecast JSON. "
        "Checked predicted_ohlc, final_predicted_ohlc, hybrid_predicted_ohlc, "
        "predicted_values, and flat pred_open/pred_high/pred_low/pred_close."
    )


def _extract_forecast_date(payload: Dict[str, Any]) -> str:
    for key in ["forecast_for_date", "forecast_date", "date"]:
        if key in payload and payload[key]:
            return str(payload[key])
    raise KeyError("Could not determine forecast date from forecast JSON")


def _fetch_yahoo_actual(date_str: str, auto_adjust: bool = False) -> Dict[str, float]:
    dt = pd.Timestamp(date_str)
    start = dt.strftime("%Y-%m-%d")
    end = (dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(
        "^GSPC",
        start=start,
        end=end,
        interval="1d",
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError(f"No Yahoo Finance data returned for ^GSPC on {date_str}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    row = df.iloc[0]
    return {
        "open": float(row["Open"]),
        "high": float(row["High"]),
        "low": float(row["Low"]),
        "close": float(row["Close"]),
        "volume": float(row["Volume"]) if "Volume" in row.index and pd.notna(row["Volume"]) else None,
    }


def _compute_metrics(pred: Dict[str, float], actual: Dict[str, float]) -> Dict[str, float]:
    out = {}
    abs_errs = []
    sq_errs = []

    for field in ["open", "high", "low", "close"]:
        err = float(pred[field] - actual[field])
        ae = abs(err)
        se = err ** 2
        out[f"{field}_error"] = err
        out[f"{field}_abs_error"] = ae
        out[f"{field}_ape"] = (ae / abs(actual[field]) * 100.0) if actual[field] != 0 else None
        abs_errs.append(ae)
        sq_errs.append(se)

    out["mean_ohlc_mae"] = float(sum(abs_errs) / 4.0)
    out["ohlc_rmse"] = float((sum(sq_errs) / 4.0) ** 0.5)

    pred_range = pred["high"] - pred["low"]
    actual_range = actual["high"] - actual["low"]
    out["pred_range"] = float(pred_range)
    out["actual_range"] = float(actual_range)
    out["range_error"] = float(pred_range - actual_range)
    out["range_abs_error"] = abs(out["range_error"])

    out["high_coverage"] = float(actual["high"] <= pred["high"])
    out["low_coverage"] = float(actual["low"] >= pred["low"])
    out["inside_range_coverage"] = float(
        (actual["high"] <= pred["high"]) and (actual["low"] >= pred["low"])
    )

    return out


def _winner(hybrid_metrics: Dict[str, float], rs_metrics: Dict[str, float]) -> Dict[str, str]:
    lower_better = {
        "open_abs_error", "high_abs_error", "low_abs_error", "close_abs_error",
        "mean_ohlc_mae", "ohlc_rmse", "range_abs_error"
    }
    higher_better = {
        "high_coverage", "low_coverage", "inside_range_coverage"
    }

    verdict = {}
    for k in sorted(lower_better):
        hv = hybrid_metrics.get(k)
        rv = rs_metrics.get(k)
        if hv is None or rv is None:
            verdict[k] = "n/a"
        elif rv < hv:
            verdict[k] = "range_skew"
        elif hv < rv:
            verdict[k] = "hybrid"
        else:
            verdict[k] = "tie"

    for k in sorted(higher_better):
        hv = hybrid_metrics.get(k)
        rv = rs_metrics.get(k)
        if hv is None or rv is None:
            verdict[k] = "n/a"
        elif rv > hv:
            verdict[k] = "range_skew"
        elif hv > rv:
            verdict[k] = "hybrid"
        else:
            verdict[k] = "tie"

    return verdict


def main():
    ap = argparse.ArgumentParser(description="Compare daily hybrid vs range+skew forecasts against Yahoo SPX actuals")
    ap.add_argument("--hybrid-json", default=str(DEFAULT_HYBRID))
    ap.add_argument("--range-skew-json", default=str(DEFAULT_RANGE_SKEW))
    ap.add_argument("--date", default=None, help="Override forecast date YYYY-MM-DD")
    ap.add_argument("--allow-date-override", action="store_true", help="Allow explicit override of forecast date mismatch")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--auto-adjust", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hybrid_payload = _load_json(Path(args.hybrid_json))
    rs_payload = _load_json(Path(args.range_skew_json))

    hybrid_pred = _extract_predicted_ohlc(hybrid_payload)
    rs_pred = _extract_predicted_ohlc(rs_payload)

    hybrid_date = _extract_forecast_date(hybrid_payload)
    rs_date = _extract_forecast_date(rs_payload)
    eval_date = args.date or hybrid_date

    if hybrid_date != rs_date:
        raise ValueError(
            f"Forecast date mismatch between files: hybrid={hybrid_date}, range_skew={rs_date}"
        )

    if args.date is not None and eval_date != hybrid_date and not args.allow_date_override:
        raise ValueError(
            f"Refusing to compare forecast files dated {hybrid_date} against actuals for {eval_date} "
            f"without --allow-date-override"
        )

    try:
        actual = _fetch_yahoo_actual(eval_date, auto_adjust=args.auto_adjust)
    except ValueError as e:
        msg = str(e)
        print(f"Actual OHLC not available yet for {eval_date}: {msg}")
        print("Re-run after market close and Yahoo Finance daily data update.")
        return

    hybrid_metrics = _compute_metrics(hybrid_pred, actual)
    rs_metrics = _compute_metrics(rs_pred, actual)
    winners = _winner(hybrid_metrics, rs_metrics)

    report = {
        "forecast_date": eval_date,
        "hybrid_forecast_path": str(Path(args.hybrid_json)),
        "range_skew_forecast_path": str(Path(args.range_skew_json)),
        "actual_source": "Yahoo Finance ^GSPC",
        "actual_ohlc": actual,
        "hybrid_predicted_ohlc": hybrid_pred,
        "range_skew_predicted_ohlc": rs_pred,
        "hybrid_metrics": hybrid_metrics,
        "range_skew_metrics": rs_metrics,
        "metric_diff_range_skew_minus_hybrid": {
            k: (rs_metrics[k] - hybrid_metrics[k])
            for k in rs_metrics.keys()
            if k in hybrid_metrics and rs_metrics[k] is not None and hybrid_metrics[k] is not None
        },
        "winner_by_metric": winners,
    }

    json_path = outdir / f"{eval_date}_hybrid_vs_range_skew_actuals.json"
    json_path.write_text(json.dumps(report, indent=2))

    row = {
        "forecast_date": eval_date,
        "actual_open": actual["open"],
        "actual_high": actual["high"],
        "actual_low": actual["low"],
        "actual_close": actual["close"],

        "hybrid_open": hybrid_pred["open"],
        "hybrid_high": hybrid_pred["high"],
        "hybrid_low": hybrid_pred["low"],
        "hybrid_close": hybrid_pred["close"],

        "range_skew_open": rs_pred["open"],
        "range_skew_high": rs_pred["high"],
        "range_skew_low": rs_pred["low"],
        "range_skew_close": rs_pred["close"],
    }
    for prefix, metrics in [("hybrid", hybrid_metrics), ("range_skew", rs_metrics)]:
        for k, v in metrics.items():
            row[f"{prefix}_{k}"] = v

    scorecard_csv = outdir / "daily_hybrid_vs_range_skew_scorecard.csv"
    new_df = pd.DataFrame([row])

    if scorecard_csv.exists():
        old = pd.read_csv(scorecard_csv)
        combined = pd.concat([old, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["forecast_date"], keep="last").sort_values("forecast_date")
    else:
        combined = new_df

    combined.to_csv(scorecard_csv, index=False)

    print(f"Saved JSON report: {json_path}")
    print(f"Updated scorecard: {scorecard_csv}")
    print("\n=== DAILY FORECAST COMPARISON ===")
    print(f"forecast_date: {eval_date}")
    print(f"actual OHLC:       {actual}")
    print(f"hybrid OHLC:       {hybrid_pred}")
    print(f"range+skew OHLC:   {rs_pred}")
    print("\n--- Mean OHLC MAE ---")
    print(f"hybrid:      {hybrid_metrics['mean_ohlc_mae']:.4f}")
    print(f"range+skew:  {rs_metrics['mean_ohlc_mae']:.4f}")
    print("\n--- Range MAE ---")
    print(f"hybrid:      {hybrid_metrics['range_abs_error']:.4f}")
    print(f"range+skew:  {rs_metrics['range_abs_error']:.4f}")
    print("\n--- Inside Range Coverage ---")
    print(f"hybrid:      {hybrid_metrics['inside_range_coverage']:.0f}")
    print(f"range+skew:  {rs_metrics['inside_range_coverage']:.0f}")


if __name__ == "__main__":
    main()
