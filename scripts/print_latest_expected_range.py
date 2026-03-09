#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_HYBRID = Path("output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json")
DEFAULT_RANGE_SKEW = Path("output/forecasts/latest_gap_augmented_range_skew_forecast.json")


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing forecast file: {path}")
    return json.loads(path.read_text())


def _extract_ohlc(payload: dict, label: str):
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

    raise KeyError(f"Could not extract OHLC from {label}")


def _forecast_date(payload: dict):
    for key in ["forecast_for_date", "forecast_date", "date"]:
        if key in payload:
            return str(payload[key])
    return "unknown"


def _feature_date(payload: dict):
    for key in ["generated_from_feature_date", "feature_date", "as_of_date"]:
        if key in payload:
            return str(payload[key])
    return "unknown"


def _fmt_ohlc(ohlc: dict):
    rng = ohlc["high"] - ohlc["low"]
    up = ohlc["high"] - ohlc["open"]
    down = ohlc["open"] - ohlc["low"]
    return {
        "open": round(ohlc["open"], 2),
        "high": round(ohlc["high"], 2),
        "low": round(ohlc["low"], 2),
        "close": round(ohlc["close"], 2),
        "range": round(rng, 2),
        "up_from_open": round(up, 2),
        "down_from_open": round(down, 2),
    }


def main():
    ap = argparse.ArgumentParser(description="Print latest expected SPX range from hybrid and range+skew forecasts")
    ap.add_argument("--hybrid-json", default=str(DEFAULT_HYBRID))
    ap.add_argument("--range-skew-json", default=str(DEFAULT_RANGE_SKEW))
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    args = ap.parse_args()

    hybrid_payload = _load_json(Path(args.hybrid_json))
    rs_payload = _load_json(Path(args.range_skew_json))

    hybrid_ohlc = _extract_ohlc(hybrid_payload, "hybrid forecast")
    rs_ohlc = _extract_ohlc(rs_payload, "range+skew forecast")

    hybrid_forecast_date = _forecast_date(hybrid_payload)
    rs_forecast_date = _forecast_date(rs_payload)
    if hybrid_forecast_date != rs_forecast_date:
        raise SystemExit(
            f"Forecast date mismatch: hybrid={hybrid_forecast_date}, range_skew={rs_forecast_date}"
        )

    hybrid_summary = _fmt_ohlc(hybrid_ohlc)
    rs_summary = _fmt_ohlc(rs_ohlc)

    result = {
        "forecast_for_date": _forecast_date(rs_payload),
        "generated_from_feature_date": _feature_date(rs_payload),
        "hybrid": hybrid_summary,
        "range_skew": rs_summary,
        "range_skew_overlay": rs_payload.get("range_skew_overlay", {}),
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=== LATEST EXPECTED SPX RANGE ===")
    print("forecast_for_date:", result["forecast_for_date"])
    print("generated_from_feature_date:", result["generated_from_feature_date"])
    print()

    print("--- Hybrid forecast ---")
    print(
        f"open {hybrid_summary['open']:.2f}, high {hybrid_summary['high']:.2f}, "
        f"low {hybrid_summary['low']:.2f}, close {hybrid_summary['close']:.2f}"
    )
    print(
        f"range {hybrid_summary['range']:.2f} | "
        f"up_from_open {hybrid_summary['up_from_open']:.2f} | "
        f"down_from_open {hybrid_summary['down_from_open']:.2f}"
    )
    print()

    print("--- Range+skew forecast ---")
    print(
        f"open {rs_summary['open']:.2f}, high {rs_summary['high']:.2f}, "
        f"low {rs_summary['low']:.2f}, close {rs_summary['close']:.2f}"
    )
    print(
        f"range {rs_summary['range']:.2f} | "
        f"up_from_open {rs_summary['up_from_open']:.2f} | "
        f"down_from_open {rs_summary['down_from_open']:.2f}"
    )
    print()

    overlay = result["range_skew_overlay"]
    if overlay:
        print("--- Range+skew overlay ---")
        for k in [
            "pred_range",
            "pred_up_share_raw",
            "pred_up_share_model_clipped",
            "hybrid_up_share",
            "skew_alpha",
            "pred_up_share_blended",
        ]:
            if k in overlay:
                v = overlay[k]
                if isinstance(v, float):
                    print(f"{k}: {v:.6f}")
                else:
                    print(f"{k}: {v}")


if __name__ == "__main__":
    main()
