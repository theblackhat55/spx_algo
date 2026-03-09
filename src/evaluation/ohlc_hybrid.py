from __future__ import annotations

from pathlib import Path
import json
from typing import Dict

import pandas as pd

from src.models.ohlc_forecaster import OHLC_TARGET_COLUMNS


def choose_best_component_source(
    model_component_metrics: dict,
    rolling_component_metrics: dict,
) -> Dict[str, str]:
    selection = {}
    for target in OHLC_TARGET_COLUMNS:
        model_mae = model_component_metrics[target]["mae"]
        rolling_mae = rolling_component_metrics[target]["mae"]
        selection[target] = "model" if model_mae <= rolling_mae else "rolling_baseline"
    return selection


def build_hybrid_components(
    model_components: pd.DataFrame,
    rolling_components: pd.DataFrame,
    selection: Dict[str, str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=model_components.index)
    for target in OHLC_TARGET_COLUMNS:
        source = selection[target]
        if source == "model":
            out[target] = model_components[target]
        elif source == "rolling_baseline":
            out[target] = rolling_components[target]
        else:
            raise ValueError(f"Unknown source for {target}: {source}")
    return out


def summarize_selection(
    selection: Dict[str, str],
    model_component_metrics: dict,
    rolling_component_metrics: dict,
) -> str:
    lines = ["# Hybrid Component Selection Summary", ""]
    for target in OHLC_TARGET_COLUMNS:
        mm = model_component_metrics[target]["mae"]
        rm = rolling_component_metrics[target]["mae"]
        chosen = selection[target]
        lines.append(f"## {target}")
        lines.append(f"- model_mae: {mm:.6f}")
        lines.append(f"- rolling_baseline_mae: {rm:.6f}")
        lines.append(f"- selected: {chosen}")
        lines.append("")
    return "\n".join(lines)


def save_text(text: str, output_file: str | Path) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(data: dict, output_file: str | Path) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_hybrid_predictions(
    output_file: str | Path,
    actual_ohlc: pd.DataFrame,
    prev_close: pd.Series,
    y_test: pd.DataFrame,
    model_components: pd.DataFrame,
    rolling_components: pd.DataFrame,
    hybrid_components: pd.DataFrame,
    model_ohlc: pd.DataFrame,
    rolling_ohlc: pd.DataFrame,
    hybrid_ohlc: pd.DataFrame,
) -> None:
    idx = y_test.index
    df = pd.DataFrame(index=idx)

    df["prev_close"] = prev_close.reindex(idx)
    df["actual_open"] = actual_ohlc.loc[idx, "Open"]
    df["actual_high"] = actual_ohlc.loc[idx, "High"]
    df["actual_low"] = actual_ohlc.loc[idx, "Low"]
    df["actual_close"] = actual_ohlc.loc[idx, "Close"]

    for col in OHLC_TARGET_COLUMNS:
        df[f"actual_{col}"] = y_test[col]
        df[f"model_{col}"] = model_components[col]
        df[f"rolling_{col}"] = rolling_components[col]
        df[f"hybrid_{col}"] = hybrid_components[col]

    for col in ["pred_open", "pred_high", "pred_low", "pred_close"]:
        df[f"model_{col}"] = model_ohlc[col]
        df[f"rolling_{col}"] = rolling_ohlc[col]
        df[f"hybrid_{col}"] = hybrid_ohlc[col]

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
