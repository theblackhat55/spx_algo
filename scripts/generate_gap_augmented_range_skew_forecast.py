#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


DEFAULT_HYBRID_FORECAST_JSON = Path("output/forecasts/latest_gap_augmented_hybrid_ohlc_forecast.json")
DEFAULT_HYBRID_PREDS_CSV = Path("output/backtests/gap_augmented_hybrid/predictions.csv")
DEFAULT_OUT = Path("output/forecasts/latest_gap_augmented_range_skew_forecast.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a gap-augmented forecast with learned range+skew overlay."
    )
    p.add_argument("--hybrid-forecast-json", default=str(DEFAULT_HYBRID_FORECAST_JSON))
    p.add_argument("--hybrid-predictions-csv", default=str(DEFAULT_HYBRID_PREDS_CSV))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--min-up-share", type=float, default=0.2)
    p.add_argument("--max-up-share", type=float, default=0.8)
    p.add_argument("--skew-alpha", type=float, default=0.3,
                   help="Blend weight for model skew vs hybrid original skew.")
    return p.parse_args()


def _load_hybrid_forecast(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing hybrid forecast JSON: {path}")
    return json.loads(path.read_text())


def _extract_forecast_ohlc(payload: dict[str, Any]) -> dict[str, float]:
    src = payload.get("predicted_ohlc", payload)

    aliases = {
        "open": ["open", "pred_open", "predicted_open"],
        "high": ["high", "pred_high", "predicted_high"],
        "low": ["low", "pred_low", "predicted_low"],
        "close": ["close", "pred_close", "predicted_close"],
    }

    out = {}
    for key, names in aliases.items():
        for name in names:
            if name in src:
                out[key] = float(src[name])
                break
        else:
            raise KeyError(f"Could not find {key} in hybrid forecast JSON")
    return out


def _load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing hybrid predictions CSV: {path}")

    df = pd.read_csv(path)
    required = {
        "forecast_date", "prev_close",
        "pred_open", "pred_high", "pred_low", "pred_close",
        "actual_open", "actual_high", "actual_low", "actual_close",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hybrid predictions CSV missing columns: {sorted(missing)}")

    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    return df.sort_values("forecast_date").reset_index(drop=True)


def _safe_div(a: pd.Series, b: pd.Series, default: float = 0.5) -> pd.Series:
    out = pd.Series(default, index=a.index, dtype=float)
    mask = b.abs() > 1e-9
    out.loc[mask] = a.loc[mask] / b.loc[mask]
    return out


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["pred_range"] = out["pred_high"] - out["pred_low"]
    out["actual_range"] = out["actual_high"] - out["actual_low"]

    out["pred_up_ext"] = np.maximum(out["pred_high"] - out["pred_open"], 0.0)
    out["pred_down_ext"] = np.maximum(out["pred_open"] - out["pred_low"], 0.0)

    out["actual_up_ext_from_pred_open"] = np.maximum(out["actual_high"] - out["pred_open"], 0.0)
    out["actual_down_ext_from_pred_open"] = np.maximum(out["pred_open"] - out["actual_low"], 0.0)

    out["pred_up_share"] = _safe_div(out["pred_up_ext"], out["pred_range"], default=0.5)
    out["actual_up_share_from_pred_open"] = _safe_div(
        out["actual_up_ext_from_pred_open"], out["actual_range"], default=0.5
    ).clip(0.0, 1.0)

    out["pred_gap_abs"] = np.abs(out["pred_open"] / out["prev_close"] - 1.0)
    out["pred_close_from_open"] = out["pred_close"] / out["pred_open"] - 1.0
    out["pred_close_from_open_abs"] = np.abs(out["pred_close_from_open"])

    out["actual_gap_abs"] = np.abs(out["actual_open"] / out["prev_close"] - 1.0)
    out["actual_close_from_open"] = out["actual_close"] / out["actual_open"] - 1.0
    out["actual_close_from_open_abs"] = np.abs(out["actual_close_from_open"])

    out["weekday"] = out["forecast_date"].dt.weekday
    out["month"] = out["forecast_date"].dt.month

    lag_cols = [
        "pred_range", "actual_range",
        "pred_gap_abs", "actual_gap_abs",
        "pred_close_from_open_abs", "actual_close_from_open_abs",
        "pred_up_share", "actual_up_share_from_pred_open",
    ]
    for col in lag_cols:
        out[f"{col}_lag1"] = out[col].shift(1)
        out[f"{col}_lag2"] = out[col].shift(2)
        out[f"{col}_ma5"] = out[col].rolling(5).mean().shift(1)

    return out


def _model_params() -> dict[str, Any]:
    return {
        "objective": "regression",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }


def _fit_model(train: pd.DataFrame, feature_cols: list[str], target_col: str) -> LGBMRegressor:
    model = LGBMRegressor(**_model_params())
    model.fit(train[feature_cols], train[target_col])
    return model


def _feature_importance(model: LGBMRegressor, feature_cols: list[str]) -> list[dict[str, Any]]:
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return imp.to_dict(orient="records")


def main() -> None:
    args = _parse_args()

    hybrid_forecast_path = Path(args.hybrid_forecast_json)
    hybrid_preds_path = Path(args.hybrid_predictions_csv)
    out_path = Path(args.out)

    hybrid_payload = _load_hybrid_forecast(hybrid_forecast_path)
    hybrid_ohlc = _extract_forecast_ohlc(hybrid_payload)

    history = _load_history(hybrid_preds_path)
    feat_df = _build_features(history).dropna().reset_index(drop=True)

    if len(feat_df) < 100:
        raise SystemExit(f"Not enough historical rows to train range/skew models: {len(feat_df)}")

    range_features = [
        "pred_range",
        "pred_up_share",
        "pred_gap_abs",
        "pred_close_from_open",
        "pred_close_from_open_abs",
        "weekday",
        "month",
        "pred_range_lag1",
        "pred_range_lag2",
        "pred_range_ma5",
        "actual_range_lag1",
        "actual_range_lag2",
        "actual_range_ma5",
        "pred_gap_abs_lag1",
        "pred_gap_abs_lag2",
        "pred_gap_abs_ma5",
        "pred_close_from_open_abs_lag1",
        "pred_close_from_open_abs_lag2",
        "pred_close_from_open_abs_ma5",
    ]
    range_features = [c for c in range_features if c in feat_df.columns]

    skew_features = [
        "pred_up_share",
        "pred_gap_abs",
        "pred_close_from_open",
        "pred_close_from_open_abs",
        "pred_range",
        "weekday",
        "month",
        "pred_up_share_lag1",
        "pred_up_share_lag2",
        "pred_up_share_ma5",
        "actual_up_share_from_pred_open_lag1",
        "actual_up_share_from_pred_open_lag2",
        "actual_up_share_from_pred_open_ma5",
        "pred_gap_abs_lag1",
        "pred_gap_abs_lag2",
        "pred_gap_abs_ma5",
        "pred_close_from_open_abs_lag1",
        "pred_close_from_open_abs_lag2",
        "pred_close_from_open_abs_ma5",
        "actual_range_lag1",
        "actual_range_ma5",
    ]
    skew_features = [c for c in skew_features if c in feat_df.columns]

    range_model = _fit_model(feat_df, range_features, "actual_range")
    skew_model = _fit_model(feat_df, skew_features, "actual_up_share_from_pred_open")

    latest = feat_df.iloc[-1].copy()

    # Replace the latest row's forecast-side values with the freshly generated hybrid forecast
    latest["pred_open"] = hybrid_ohlc["open"]
    latest["pred_high"] = hybrid_ohlc["high"]
    latest["pred_low"] = hybrid_ohlc["low"]
    latest["pred_close"] = hybrid_ohlc["close"]

    latest["pred_range"] = latest["pred_high"] - latest["pred_low"]
    latest["pred_up_ext"] = max(latest["pred_high"] - latest["pred_open"], 0.0)
    latest["pred_down_ext"] = max(latest["pred_open"] - latest["pred_low"], 0.0)
    latest["pred_up_share"] = (
        latest["pred_up_ext"] / latest["pred_range"] if latest["pred_range"] > 1e-9 else 0.5
    )
    latest["pred_gap_abs"] = abs(latest["pred_open"] / latest["prev_close"] - 1.0)
    latest["pred_close_from_open"] = latest["pred_close"] / latest["pred_open"] - 1.0
    latest["pred_close_from_open_abs"] = abs(latest["pred_close_from_open"])

    latest_df = pd.DataFrame([latest])

    pred_range = float(max(range_model.predict(latest_df[range_features])[0], 0.0))
    pred_up_share_raw = float(skew_model.predict(latest_df[skew_features])[0])
    pred_up_share_model = float(np.clip(pred_up_share_raw, args.min_up_share, args.max_up_share))

    hybrid_range = max(hybrid_ohlc["high"] - hybrid_ohlc["low"], 1e-9)
    hybrid_up_share = float(
        np.clip((hybrid_ohlc["high"] - hybrid_ohlc["open"]) / hybrid_range, 0.0, 1.0)
    )

    blended_up_share = float(
        np.clip(
            args.skew_alpha * pred_up_share_model + (1.0 - args.skew_alpha) * hybrid_up_share,
            args.min_up_share,
            args.max_up_share,
        )
    )

    pred_open = hybrid_ohlc["open"]
    pred_high = float(pred_open + pred_range * blended_up_share)
    pred_low = float(pred_open - pred_range * (1.0 - blended_up_share))
    pred_close = hybrid_ohlc["close"]

    out_payload = {
        "forecast_for_date": hybrid_payload.get("forecast_for_date"),
        "generated_from_feature_date": hybrid_payload.get("generated_from_feature_date"),
        "source_forecast_file": str(hybrid_forecast_path),
        "source_selection": {
            "open": "databento_gap_augmented_hybrid",
            "range": "learned_range_model",
            "skew": "learned_skew_model",
            "close": "databento_gap_augmented_hybrid",
        },
        "hybrid_predicted_ohlc": hybrid_ohlc,
        "range_skew_overlay": {
            "pred_range": pred_range,
            "pred_up_share_raw": pred_up_share_raw,
            "pred_up_share_model_clipped": pred_up_share_model,
            "hybrid_up_share": hybrid_up_share,
            "skew_alpha": args.skew_alpha,
            "pred_up_share_blended": blended_up_share,
        },
        "predicted_ohlc": {
            "open": pred_open,
            "high": pred_high,
            "low": pred_low,
            "close": pred_close,
        },
        "model_metadata": {
            "range_training_rows": int(len(feat_df)),
            "skew_training_rows": int(len(feat_df)),
            "range_feature_count": len(range_features),
            "skew_feature_count": len(skew_features),
            "range_feature_importance_top10": _feature_importance(range_model, range_features)[:10],
            "skew_feature_importance_top10": _feature_importance(skew_model, skew_features)[:10],
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2))

    print("Saved forecast:", out_path)
    print()
    print("=== GAP AUGMENTED RANGE+SKEW FORECAST CHECK ===")
    print("forecast_for_date:", out_payload["forecast_for_date"])
    print("generated_from_feature_date:", out_payload["generated_from_feature_date"])
    print("hybrid_predicted_ohlc:", out_payload["hybrid_predicted_ohlc"])
    print("range_skew_overlay:", out_payload["range_skew_overlay"])
    print("final_predicted_ohlc:", out_payload["predicted_ohlc"])


if __name__ == "__main__":
    main()
