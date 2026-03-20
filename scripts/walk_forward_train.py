#!/usr/bin/env python3
"""
scripts/walk_forward_train.py
─────────────────────────────
Walk-forward training script for the SPX Algo regression models.  Each fold
uses an expanding training window (anchored to the first available date) and
validates on the immediately following out-of-sample window.

Correct module paths (verified against repo structure):
  • src.features.builder.build_feature_matrix  — builds feature matrix
  • src.targets.engineer.engineer_targets      — builds regression targets
  • src.targets.engineer.align_features_targets — aligns X/y on date index
  • src.pipeline.signal_generator._StackingEnsemble — fallback stacking model

Usage
─────
    python scripts/walk_forward_train.py [options]

Options
    --start-year INT    First year to include in training (default: 2010)
    --val-days   INT    OOS validation window in trading days per fold (default: 63)
    --folds      INT    Number of walk-forward folds (default: 8)
    --output-dir PATH   Directory to save model artefacts (default: output/models)
    --raw-dir    PATH   Directory containing raw parquet files (default: data/raw)
    --seed       INT    Random seed for reproducibility (default: 42)
    --verbose           Print per-fold metrics to stdout
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("walk_forward_train")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward training for SPX Algo models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start-year", type=int, default=2010,
                   help="First calendar year to include in the training universe")
    p.add_argument("--val-days", type=int, default=63,
                   help="OOS validation window per fold (trading days)")
    p.add_argument("--folds", type=int, default=8,
                   help="Number of walk-forward folds")
    p.add_argument("--output-dir", type=Path, default=Path("output/models"),
                   help="Destination for saved model artefacts")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                   help="Directory containing spx_daily.parquet / vix_daily.parquet")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-fold metrics to stdout")
    return p.parse_args()


def _legacy_guard():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--allow-legacy-output", action="store_true")
    args, _ = parser.parse_known_args()
    if not args.allow_legacy_output:
        raise SystemExit(
            "walk_forward_train.py is deprecated for production artifact writes. "
            "Use scripts/retrain_full_stack.py instead, or rerun with --allow-legacy-output "
            "only for research/backtest purposes."
        )


def main() -> None:
    _legacy_guard()
    args = _parse_args()

    import numpy as np
    import pandas as pd
    import joblib

    # ── Correct imports (verified against repo structure) ─────────────────────
    try:
        from src.features.builder import build_feature_matrix
        from src.targets.engineer import engineer_targets, align_features_targets
        from src.targets.splitter import WalkForwardSplitter
        from src.training.config import PRODUCTION_SPLIT_CONFIG
        from src.pipeline.signal_generator import _StackingEnsemble
    except ImportError as exc:
        logger.error("Failed to import project modules: %s", exc)
        sys.exit(1)

    spx_path = args.raw_dir / "spx_daily.parquet"
    if not spx_path.exists():
        logger.error("SPX parquet not found at %s", spx_path)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build full feature matrix and target table ────────────────────────────
    logger.info("Building feature matrix from %s …", args.raw_dir)
    try:
        all_features = build_feature_matrix(raw_dir=args.raw_dir)
    except Exception as exc:
        logger.error("build_feature_matrix failed: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info("Building regression targets …")
    try:
        all_targets = engineer_targets(spx_path=spx_path, save=False)
    except Exception as exc:
        logger.error("engineer_targets failed: %s", exc, exc_info=True)
        sys.exit(1)

    # Filter to start year
    all_features = all_features[all_features.index.year >= args.start_year]
    all_targets  = all_targets[all_targets.index.year  >= args.start_year]

    # Align on common dates and limit to rows with both X and y
    X_full, y_full = align_features_targets(
        all_features, all_targets,
        target_cols=["target_high", "target_low"],
    )
    n_rows = len(X_full)
    logger.info(
        "Aligned dataset: %d rows × %d features  (start year %d)",
        n_rows, X_full.shape[1], args.start_year,
    )

    # ── Walk-forward loop ──────────────────────────────────────────────────────
    split_cfg = PRODUCTION_SPLIT_CONFIG
    splitter = WalkForwardSplitter(split_cfg)
    folds = splitter.split(X_full)

    if args.folds:
        folds = folds[:args.folds]

    fold_metrics: list[dict] = []

    for fold, (tr_idx, te_idx) in enumerate(folds, start=1):
        X_train = X_full.iloc[tr_idx]
        y_train = y_full.iloc[tr_idx]
        X_val   = X_full.iloc[te_idx]
        y_val   = y_full.iloc[te_idx]

        if len(X_train) == 0 or len(X_val) == 0:
            logger.warning("Empty split for fold %d — skipping", fold)
            continue

        logger.info(
            "Fold %d/%d — train: %s→%s (%d rows) | val: %s→%s (%d rows)",
            fold, len(folds),
            X_train.index[0].date(), X_train.index[-1].date(), len(X_train),
            X_val.index[0].date(),   X_val.index[-1].date(),   len(X_val),
        )

        try:
            model_high = _StackingEnsemble(name=f"fold{fold}_high", seed=args.seed)
            model_high.fit(X_train.values, y_train["target_high"].values)

            model_low  = _StackingEnsemble(name=f"fold{fold}_low", seed=args.seed)
            model_low.fit(X_train.values, y_train["target_low"].values)

            mae_h = float(np.mean(np.abs(
                model_high.predict(X_val.values) - y_val["target_high"].values)))
            mae_l = float(np.mean(np.abs(
                model_low.predict(X_val.values) - y_val["target_low"].values)))

            fold_metrics.append({"fold": fold, "mae_high": mae_h, "mae_low": mae_l})
            if args.verbose:
                print(f"  Fold {fold}: MAE high={mae_h:.2f}  low={mae_l:.2f}")

        except Exception as exc:
            logger.error("Fold %d failed: %s", fold, exc, exc_info=True)

    # ── Train final model on ALL aligned data ─────────────────────────────────
    logger.info("Training final model on full dataset (%d rows) …", n_rows)
    try:
        final_high = _StackingEnsemble(name="final_high", seed=args.seed)
        final_high.fit(X_full.values, y_full["target_high"].values)
        final_low  = _StackingEnsemble(name="final_low",  seed=args.seed)
        final_low.fit(X_full.values, y_full["target_low"].values)

        # Name matches _load_model_artifact search order (first priority)
        joblib.dump(final_high, args.output_dir / "regressor_target_high_pct.pkl")
        joblib.dump(final_low,  args.output_dir / "regressor_target_low_pct.pkl")
        logger.info("Final models saved to %s", args.output_dir)
    except Exception as exc:
        logger.error("Final model training failed: %s", exc, exc_info=True)
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    if fold_metrics:
        import statistics
        avg_h = statistics.mean(m["mae_high"] for m in fold_metrics)
        avg_l = statistics.mean(m["mae_low"]  for m in fold_metrics)
        logger.info(
            "Walk-forward summary — %d folds | avg MAE high=%.2f  low=%.2f",
            len(fold_metrics), avg_h, avg_l,
        )
    logger.info("walk_forward_train.py complete")


if __name__ == "__main__":
    main()
