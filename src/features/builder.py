"""
src/features/builder.py
========================
Task 10 — Feature matrix assembler.

Orchestrates all feature modules (Tasks 4–9), merges the results on
the date index, drops warm-up NaN rows, and saves the result to
``data/processed/features.parquet``.

Functions
---------
build_feature_matrix   — Full rebuild from raw Parquet files
load_feature_matrix    — Load the saved features.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPX_FILE,
    VIX_FILE,
    CALENDAR_FILE,
    FEATURES_FILE,
    GEX_FILE,
)
from src.features.proximity        import compute_proximity_features
from src.features.volatility       import compute_all_volatility_features
from src.features.technical        import compute_all_technical_features
from src.features.calendar_features import compute_calendar_features
from src.features.options_features  import compute_all_options_features
from src.features.lagged_targets    import compute_lagged_target_features
from src.data.calendar              import build_trading_calendar

logger = logging.getLogger(__name__)


def build_feature_matrix(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    spx_file: Path = SPX_FILE,
    vix_file: Path = VIX_FILE,
) -> pd.DataFrame:
    """Build the complete feature matrix from raw Parquet files.

    Steps
    -----
    1. Load SPX and VIX raw data.
    2. Build / load event calendar.
    3. Compute each feature group.
    4. Inner-join all feature DataFrames on the date index.
    5. Drop warm-up rows (any row with at least one NaN).
    6. Save to ``processed/features.parquet``.

    Parameters
    ----------
    raw_dir:
        Directory containing raw Parquet files.
    processed_dir:
        Directory for saving processed features.
    spx_file:
        Path to the SPX daily Parquet.
    vix_file:
        Path to the VIX daily Parquet.

    Returns
    -------
    Feature matrix DataFrame — no NaN values, DatetimeIndex.
    """
    # ── Load raw data ─────────────────────────────────────────────────────────
    if not spx_file.exists():
        raise FileNotFoundError(
            f"SPX file not found: {spx_file}\n"
            "Run src/data/fetcher.py first to download raw data."
        )
    if not vix_file.exists():
        raise FileNotFoundError(
            f"VIX file not found: {vix_file}\n"
            "Run src/data/fetcher.py first to download raw data."
        )

    logger.info("Loading SPX and VIX data …")
    spx = pd.read_parquet(spx_file)
    vix = pd.read_parquet(vix_file)

    # ── Build / load calendar ─────────────────────────────────────────────────
    logger.info("Building event calendar …")
    if CALENDAR_FILE.exists():
        from src.data.calendar import load_calendar
        calendar = load_calendar(CALENDAR_FILE)
    else:
        calendar = build_trading_calendar(
            trading_dates=spx.index,
            save_path=CALENDAR_FILE,
        )

    # ── Optional data sources ─────────────────────────────────────────────────
    gex_df = None
    if GEX_FILE.exists():
        try:
            gex_df = pd.read_csv(GEX_FILE, index_col=0, parse_dates=True)
            logger.info("GEX data loaded (%d rows).", len(gex_df))
        except Exception as exc:
            logger.warning("Could not load GEX file: %s", exc)

    # ── Compute feature groups ────────────────────────────────────────────────
    logger.info("Computing proximity features …")
    prox = compute_proximity_features(spx)

    logger.info("Computing volatility features …")
    vol = compute_all_volatility_features(spx, vix)

    logger.info("Computing technical features …")
    tech = compute_all_technical_features(spx)

    logger.info("Computing calendar features …")
    cal_feat = compute_calendar_features(spx, calendar)

    logger.info("Computing options features …")
    opts = compute_all_options_features(spx, vix, gex_df=gex_df)

    logger.info("Computing lagged-target features …")
    lagged = compute_lagged_target_features(spx)

    # ── Inner join on date index ──────────────────────────────────────────────
    logger.info("Merging feature groups …")
    frames = [prox, vol, tech, cal_feat, lagged]
    if not opts.empty:
        frames.append(opts)

    features = pd.concat(frames, axis=1, join="inner")

    # ── Drop warm-up NaN rows ─────────────────────────────────────────────────
    # First: remove any column that is entirely NaN
    # (e.g. GARCH when arch library is not installed)
    all_nan_cols = features.columns[features.isnull().all()].tolist()
    if all_nan_cols:
        logger.warning(
            "Dropping %d all-NaN column(s) before dropna: %s",
            len(all_nan_cols), all_nan_cols,
        )
        features = features.drop(columns=all_nan_cols)

    before = len(features)
    features = features.dropna()
    dropped = before - len(features)
    logger.info(
        "Dropped %d warm-up rows (%.1f%% of data).",
        dropped, 100 * dropped / max(before, 1),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    processed_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(processed_dir / "features.parquet")
    if len(features) == 0:
        raise RuntimeError(
            "Feature matrix is empty after dropna. "
            "Check that raw data has sufficient history and all feature "
            "modules are producing valid output."
        )

    logger.info(
        "Feature matrix saved → features.parquet  "
        "[%d rows × %d features, %s → %s]",
        len(features), features.shape[1],
        features.index[0].date(), features.index[-1].date(),
    )

    # ── Summary report ────────────────────────────────────────────────────────
    _print_feature_summary(features)

    return features


def _print_feature_summary(features: pd.DataFrame) -> None:
    """Print a structured summary of the feature matrix."""
    from config.feature_registry import FEATURE_REGISTRY

    print("\n" + "=" * 65)
    print("FEATURE MATRIX SUMMARY")
    print("=" * 65)
    print(f"  Rows      : {len(features):,}")
    print(f"  Features  : {features.shape[1]}")
    print(f"  Date range: {features.index[0].date()} → {features.index[-1].date()}")
    print(f"  NaN count : {features.isnull().sum().sum()}")
    print()

    # Group by category
    categories: dict[str, list[str]] = {}
    for col in features.columns:
        reg = FEATURE_REGISTRY.get(col, {})
        cat = reg.get("category", "unknown")
        categories.setdefault(cat, []).append(col)

    for cat, cols in sorted(categories.items()):
        print(f"  {cat:15s} ({len(cols):2d})  {', '.join(cols[:5])}"
              + (" …" if len(cols) > 5 else ""))
    print("=" * 65 + "\n")


def load_feature_matrix(
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Load the feature matrix from the saved Parquet file.

    Parameters
    ----------
    processed_dir:
        Directory containing ``features.parquet``.

    Returns
    -------
    Feature matrix DataFrame.

    Raises
    ------
    FileNotFoundError
        If ``features.parquet`` does not exist.
    """
    path = processed_dir / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {path}\n"
            "Run build_feature_matrix() first."
        )
    df = pd.read_parquet(path)
    logger.info(
        "Feature matrix loaded from %s  [%d rows × %d features]",
        path.name, len(df), df.shape[1],
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    build_feature_matrix()
