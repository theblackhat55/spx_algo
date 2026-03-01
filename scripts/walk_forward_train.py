#!/usr/bin/env python3
"""
scripts/walk_forward_train.py
─────────────────────────────
Walk-forward training script for the SPX Algo regression and classification
models.  Each fold uses an expanding training window (anchored to the first
available date) and validates on the immediately following out-of-sample window.

Usage
─────
    python scripts/walk_forward_train.py [options]

Options
    --start-year INT    First year to include in training (default: 2010)
    --val-days   INT    Out-of-sample validation window in trading days per fold
                        (default: 63 ≈ 1 quarter)
    --folds      INT    Number of walk-forward folds (default: 8)
    --output-dir PATH   Directory to save model artefacts (default: output/models)
    --data-dir   PATH   Directory containing raw parquet files (default: data/raw)
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
                   help="Out-of-sample validation window per fold (trading days)")
    p.add_argument("--folds", type=int, default=8,
                   help="Number of walk-forward folds")
    p.add_argument("--output-dir", type=Path, default=Path("output/models"),
                   help="Destination for saved model artefacts")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"),
                   help="Directory containing spx_daily.parquet and vix_daily.parquet")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-fold metrics")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    import numpy as np
    import pandas as pd
    import joblib

    try:
        from src.pipeline.signal_generator import SignalGenerator, _StackingEnsemble
        from src.features.targets.engineer import build_targets
        from src.features.engineer import build_features
    except ImportError as exc:
        logger.error("Failed to import project modules: %s", exc)
        sys.exit(1)

    spx_path = args.data_dir / "spx_daily.parquet"
    if not spx_path.exists():
        logger.error("SPX parquet not found at %s", spx_path)
        sys.exit(1)

    spx_df = pd.read_parquet(spx_path)
    spx_df.index = pd.to_datetime(spx_df.index)
    spx_df = spx_df.sort_index()

    # Filter to start year
    spx_df = spx_df[spx_df.index.year >= args.start_year]
    n_rows = len(spx_df)
    logger.info("Loaded %d rows from %s (start year %d)", n_rows, spx_path, args.start_year)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Walk-forward loop ──────────────────────────────────────────────────────
    val_days  = args.val_days
    n_folds   = args.folds
    fold_metrics: list[dict] = []

    for fold in range(n_folds):
        # Validation window: count from end of series
        val_end   = n_rows - fold * val_days
        val_start = val_end - val_days
        if val_start <= 0:
            logger.warning("Not enough data for fold %d — stopping early", fold + 1)
            break

        train_slice = spx_df.iloc[:val_start]
        val_slice   = spx_df.iloc[val_start:val_end]

        logger.info(
            "Fold %d/%d — train: %s → %s (%d rows)  |  val: %s → %s (%d rows)",
            fold + 1, n_folds,
            train_slice.index[0].date(), train_slice.index[-1].date(), len(train_slice),
            val_slice.index[0].date(),   val_slice.index[-1].date(),   len(val_slice),
        )

        try:
            targets   = build_targets(train_slice)
            y_high    = targets["target_high"].dropna()
            y_low     = targets["target_low"].dropna()
            X_train   = build_features(train_slice).loc[y_high.index]

            model_high = _StackingEnsemble(name=f"fold{fold+1}_high", seed=args.seed)
            model_high.fit(X_train, y_high)

            model_low  = _StackingEnsemble(name=f"fold{fold+1}_low",  seed=args.seed)
            model_low.fit(X_train, y_low)

            # Validation MAE
            X_val    = build_features(val_slice).loc[
                build_targets(val_slice)["target_high"].dropna().index
            ]
            y_val_h  = build_targets(val_slice)["target_high"].dropna().loc[X_val.index]
            y_val_l  = build_targets(val_slice)["target_low"].dropna().loc[X_val.index]

            mae_h = float(np.mean(np.abs(model_high.predict(X_val) - y_val_h)))
            mae_l = float(np.mean(np.abs(model_low.predict(X_val)  - y_val_l)))

            fold_metrics.append({"fold": fold + 1, "mae_high": mae_h, "mae_low": mae_l})
            if args.verbose:
                print(f"  Fold {fold+1}: MAE high={mae_h:.2f}  low={mae_l:.2f}")

        except Exception as exc:
            logger.error("Fold %d failed: %s", fold + 1, exc, exc_info=True)

    # ── Save final model trained on ALL data ──────────────────────────────────
    logger.info("Training final model on full dataset (%d rows)", n_rows)
    try:
        all_targets = build_targets(spx_df)
        y_h = all_targets["target_high"].dropna()
        y_l = all_targets["target_low"].dropna()
        X_all = build_features(spx_df).loc[y_h.index]

        final_high = _StackingEnsemble(name="final_high", seed=args.seed)
        final_high.fit(X_all, y_h)
        final_low  = _StackingEnsemble(name="final_low",  seed=args.seed)
        final_low.fit(X_all, y_l)

        joblib.dump(final_high, args.output_dir / "regressor_target_high_pct.pkl")
        joblib.dump(final_low,  args.output_dir / "regressor_target_low_pct.pkl")
        logger.info("Saved final models to %s", args.output_dir)
    except Exception as exc:
        logger.error("Final model training failed: %s", exc, exc_info=True)
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    if fold_metrics:
        import statistics
        avg_h = statistics.mean(m["mae_high"] for m in fold_metrics)
        avg_l = statistics.mean(m["mae_low"]  for m in fold_metrics)
        logger.info(
            "Walk-forward summary — %d folds completed  |  avg MAE high=%.2f  low=%.2f",
            len(fold_metrics), avg_h, avg_l,
        )
    logger.info("walk_forward_train.py complete")


if __name__ == "__main__":
    main()
