#!/usr/bin/env python3
"""
scripts/retrain_full_stack.py
=============================
Retrain the full production model stack:
  - 6 regressors (xgboost, lightgbm, catboost, huber_xgboost, huber_lightgbm, ridge)
    × 2 targets (next_high_pct, next_low_pct)
  - Meta ridge stacking layer (high + low)
  - 2 classifiers (next_high_bin_050, next_low_bin_050)

Uses walk-forward training via Trainer class with 265 folds.
Saves all artifacts to output/models/.

Usage: python scripts/retrain_full_stack.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import copy
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retrain_full_stack")

MODEL_DIR = _REPO / "output" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print plan without training")
    args = parser.parse_args()

    t0_total = time.perf_counter()

    # ── Load data ─────────────────────────────────────────────────────────
    from src.features.builder import load_feature_matrix
    from src.targets.engineer import engineer_targets, align_features_targets
    from src.targets.splitter import WalkForwardSplitter
    from src.training.config import PRODUCTION_SPLIT_CONFIG
    from src.models.trainer import Trainer
    from src.models.tree_models import (
        XGBoostModel, LightGBMModel, CatBoostModel,
        HuberXGBoostModel, HuberLightGBMModel,
    )
    from src.models.linear_models import RidgeRegressionModel

    features = load_feature_matrix()
    targets = engineer_targets(spx_path=_REPO / "data" / "raw" / "spx_daily.parquet", save=False)

    logger.info("Features: %d rows × %d cols", *features.shape)
    logger.info("Targets: %d rows, columns: %s", len(targets), list(targets.columns))

    # Splitter config matching production (265 folds)
    split_cfg = PRODUCTION_SPLIT_CONFIG

    # ── Define model zoo ──────────────────────────────────────────────────
    regressor_models = [
        ("xgboost", XGBoostModel(task="regression")),
        ("lightgbm", LightGBMModel(task="regression")),
        ("catboost", CatBoostModel(task="regression")),
        ("huber_xgboost", HuberXGBoostModel()),
        ("huber_lightgbm", HuberLightGBMModel()),
        ("ridge", RidgeRegressionModel()),
    ]

    reg_targets = ["next_high_pct", "next_low_pct"]
    clf_targets = ["next_high_bin_050", "next_low_bin_050"]
    target_label = {"next_high_pct": "high", "next_low_pct": "low"}

    if args.dry_run:
        print(f"\nDRY RUN — would train:")
        print(f"  {len(regressor_models)} regressors × {len(reg_targets)} targets = {len(regressor_models)*len(reg_targets)} models")
        print(f"  {len(clf_targets)} classifiers")
        print(f"  2 meta ridge stacking layers")
        print(f"  Split config: {split_cfg}")
        return

    # ── Train regressors ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING REGRESSORS")
    print("=" * 60)

    all_metrics = []
    oos_preds = {}  # {(model_name, target_label): pd.Series} for meta stacking

    for target_col in reg_targets:
        label = target_label[target_col]
        X_aligned, y_aligned = align_features_targets(
            features, targets, target_cols=[target_col]
        )

        for model_name, model_proto in regressor_models:
            logger.info("Training: %s → %s", model_name, target_col)
            t0 = time.perf_counter()

            model = copy.deepcopy(model_proto)
            splitter = WalkForwardSplitter(split_cfg)
            trainer = Trainer(model=model, splitter=splitter)

            result = trainer.run(X_aligned, y_aligned, target_col=target_col)
            elapsed = time.perf_counter() - t0

            mae = float(np.mean(np.abs(result.oos_proba - result.oos_actual)))
            rmse = float(np.sqrt(np.mean((result.oos_proba - result.oos_actual) ** 2)))

            all_metrics.append({
                "model": model_name, "target": label,
                "mae": mae, "rmse": rmse, "time_s": elapsed,
            })

            # Save OOS predictions for meta stacking
            oos_preds[(model_name, label)] = result.oos_proba

            # Save final model (retrain on ALL data)
            final_model = copy.deepcopy(model_proto)
            final_model.fit(X_aligned, y_aligned[target_col])
            save_name = f"regressor_{model_name}_{label}.pkl"
            final_model.save(MODEL_DIR / save_name)

            logger.info("  %s_%s: MAE=%.6f RMSE=%.6f (%.1fs) → %s",
                        model_name, label, mae, rmse, elapsed, save_name)

    # ── Train meta ridge stacking ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING META RIDGE STACKING")
    print("=" * 60)

    for label in ["high", "low"]:
        target_col = f"next_{label}_pct"
        X_al, y_al = align_features_targets(features, targets, target_cols=[target_col])

        # Build stacking features from final model predictions
        stack_cols = {}
        for model_name, _ in regressor_models:
            mpath = MODEL_DIR / f"regressor_{model_name}_{label}.pkl"
            if mpath.exists():
                m = joblib.load(mpath)
                stack_cols[model_name] = pd.Series(m.predict(X_al), index=X_al.index)

        if not stack_cols:
            logger.warning("No model predictions for meta ridge %s — skipping", label)
            continue

        stack_df = pd.DataFrame(stack_cols)
        common_idx = stack_df.index.intersection(y_al.index)
        X_stack = stack_df.loc[common_idx].fillna(0)
        y_stack = y_al.loc[common_idx, target_col]

        meta = Ridge(alpha=1.0)
        meta.fit(X_stack.values, y_stack.values)

        meta_path = MODEL_DIR / f"meta_ridge_{label}.pkl"
        joblib.dump(meta, meta_path)
        logger.info("  Meta ridge %s: coefs=%s → %s",
                     label, dict(zip(stack_cols.keys(), meta.coef_.round(4))), meta_path)

    # ── Save regressor_target_high/low_pct.pkl (primary artifact) ─────────
    # Use the best single model (lowest MAE) as the primary artifact
    metrics_df = pd.DataFrame(all_metrics)
    for label in ["high", "low"]:
        subset = metrics_df[metrics_df["target"] == label]
        best = subset.loc[subset["mae"].idxmin()]
        best_name = best["model"]
        src_path = MODEL_DIR / f"regressor_{best_name}_{label}.pkl"
        dst_path = MODEL_DIR / f"regressor_target_{label}_pct.pkl"
        if src_path.exists():
            import shutil
            shutil.copy2(src_path, dst_path)
            logger.info("  Primary %s artifact: %s (MAE=%.6f) → %s",
                        label, best_name, best["mae"], dst_path.name)

    # ── Train classifiers ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING CLASSIFIERS")
    print("=" * 60)

    from src.models.tree_models import LightGBMModel as LGBMClf

    for target_col in clf_targets:
        X_aligned, y_aligned = align_features_targets(
            features, targets, target_cols=[target_col]
        )

        model = LGBMClf(task="classification")
        splitter = WalkForwardSplitter(split_cfg)
        trainer = Trainer(model=model, splitter=splitter)

        result = trainer.run(X_aligned, y_aligned, target_col=target_col)

        auc = result.overall.get("roc_auc", float("nan"))
        acc = result.overall.get("accuracy", float("nan"))
        f1 = result.overall.get("f1", float("nan"))

        # Save final classifier trained on all data
        final_clf = LGBMClf(task="classification")
        final_clf.fit(X_aligned, y_aligned[target_col])
        save_name = f"classifier_{target_col}.pkl"
        final_clf.save(MODEL_DIR / save_name)

        print(f"\n  Classifier: {target_col} ({result.n_folds} folds)")
        print(f"    AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}")
        print(f"    Saved -> {save_name}")

    # ── Summary ───────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t0_total

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE  ({total_time/60:.1f} minutes)")
    print("=" * 60)

    print(f"\nAll metrics:")
    print(metrics_df.to_string(index=False))


    _write_manifest(metrics_df, split_cfg, features)

    print(f"\nArtifacts in {MODEL_DIR}/:")
    for p in sorted(MODEL_DIR.glob("*.pkl")):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:<50s} {size_kb:>8.0f} KB")


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(metrics_df: pd.DataFrame, split_cfg, features: pd.DataFrame) -> None:
    manifest_path = MODEL_DIR / "manifest.json"

    def _best_for(label: str):
        subset = metrics_df[metrics_df["target"] == label]
        if subset.empty:
            return None
        best = subset.loc[subset["mae"].idxmin()]
        artifact = MODEL_DIR / f"regressor_target_{label}_pct.pkl"
        return {
            "model": str(best["model"]),
            "mae": float(best["mae"]),
            "rmse": float(best["rmse"]),
            "artifact": artifact.name,
            "artifact_sha256": _sha256_file(artifact),
        }

    feature_cols = list(features.columns)
    feature_hash = hashlib.sha256("\n".join(feature_cols).encode("utf-8")).hexdigest()

    manifest = {
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "script": "scripts/retrain_full_stack.py",
        "primary_high": _best_for("high"),
        "primary_low": _best_for("low"),
        "meta_high": {
            "artifact": "meta_ridge_high.pkl",
            "artifact_sha256": _sha256_file(MODEL_DIR / "meta_ridge_high.pkl"),
        },
        "meta_low": {
            "artifact": "meta_ridge_low.pkl",
            "artifact_sha256": _sha256_file(MODEL_DIR / "meta_ridge_low.pkl"),
        },
        "feature_matrix_rows": int(features.shape[0]),
        "feature_matrix_cols": int(features.shape[1]),
        "feature_matrix_end_date": str(features.index.max().date()) if len(features) else None,
        "feature_columns_hash": feature_hash,
        "feature_columns_count": len(feature_cols),
        "split_config": {
            "min_train_rows": split_cfg.min_train_rows,
            "test_rows": split_cfg.test_rows,
            "step_rows": split_cfg.step_rows,
            "gap_rows": split_cfg.gap_rows,
            "embargo_rows": split_cfg.embargo_rows,
        },
        "artifacts": {
            p.name: {
                "sha256": _sha256_file(p),
                "size_bytes": p.stat().st_size,
            }
            for p in sorted(MODEL_DIR.glob("*.pkl"))
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        import json
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written -> %s", manifest_path)


if __name__ == "__main__":
    main()
