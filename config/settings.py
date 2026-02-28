"""
config/settings.py
==================
Single source of truth for every constant in the spx_algo project.
No magic numbers should appear anywhere else in the codebase.
"""
from pathlib import Path
import os

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# DATA SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

#: First date of historical data to download
DATA_START_DATE = "2000-01-01"

#: Yahoo Finance tickers (human-readable name → yfinance symbol)
YAHOO_TICKERS: dict[str, str] = {
    "spx":  "^GSPC",
    "es":   "ES=F",
    "vix":  "^VIX",
    "vvix": "^VVIX",
    "dxy":  "DX-Y.NYB",
    "tnx":  "^TNX",   # 10-Year Treasury Yield
    "tyx":  "^TYX",   # 30-Year Treasury Yield
    "irx":  "^IRX",   # 13-Week T-Bill
}

#: FRED series codes (series_id → human-readable name)
FRED_SERIES: dict[str, str] = {
    "T10Y2Y":        "yield_spread_10y2y",
    "DFF":           "fed_funds_rate",
    "BAMLH0A0HYM2":  "high_yield_spread",
}

# ── File paths ────────────────────────────────────────────────────────────────
RAW_DATA_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DATA_DIR  = PROJECT_ROOT / "data" / "external"
OUTPUT_DIR         = PROJECT_ROOT / "output"
MODELS_DIR         = OUTPUT_DIR / "models"
SIGNALS_DIR        = OUTPUT_DIR / "signals"
REPORTS_DIR        = OUTPUT_DIR / "reports"
PLOTS_DIR          = OUTPUT_DIR / "plots"

# Ensure critical output directories always exist at import time
for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
           MODELS_DIR, SIGNALS_DIR, REPORTS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Raw data filenames ────────────────────────────────────────────────────────
SPX_FILE    = RAW_DATA_DIR / "spx_daily.parquet"
ES_FILE     = RAW_DATA_DIR / "es_daily.parquet"
VIX_FILE    = RAW_DATA_DIR / "vix_daily.parquet"
MACRO_FILE  = RAW_DATA_DIR / "macro_fred.parquet"
VVIX_FILE   = RAW_DATA_DIR / "vvix_daily.parquet"
DXY_FILE    = RAW_DATA_DIR / "dxy_daily.parquet"
TNX_FILE    = RAW_DATA_DIR / "tnx_daily.parquet"
TYX_FILE    = RAW_DATA_DIR / "tyx_daily.parquet"
IRX_FILE    = RAW_DATA_DIR / "irx_daily.parquet"
CALENDAR_FILE = RAW_DATA_DIR / "calendar_events.csv"

# ── Processed data filenames ──────────────────────────────────────────────────
FEATURES_FILE = PROCESSED_DATA_DIR / "features.parquet"
TARGETS_FILE  = PROCESSED_DATA_DIR / "targets.parquet"

# ── External data ─────────────────────────────────────────────────────────────
GEX_FILE = EXTERNAL_DATA_DIR / "gex_daily.csv"

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

#: Lag periods for proximity variables (Jones 2015)
PROXIMITY_LAG_PERIODS: list[int] = [1, 2, 3, 5]

#: ATR window sizes in trading days
ATR_WINDOWS: list[int] = [5, 10, 20, 60]

#: EMA window sizes in trading days
EMA_WINDOWS: list[int] = [9, 21, 50, 100, 200]

#: RSI window sizes in trading days
RSI_WINDOWS: list[int] = [7, 14, 28]

#: Volume ratio rolling windows
VOLUME_RATIO_WINDOWS: list[int] = [5, 10, 20]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

#: Walk-forward training window (~3 years of trading days)
WF_TRAIN_WINDOW: int = 756

#: Walk-forward test/step window (~1 quarter of trading days)
WF_TEST_WINDOW: int = 63

#: Minimum number of training samples before a model is allowed to train
MIN_TRAIN_SAMPLES: int = 500

#: Production retraining frequency (~monthly)
RETRAIN_FREQUENCY_DAYS: int = 21

# ─────────────────────────────────────────────────────────────────────────────
# REGIME SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

#: Number of hidden states in the HMM
HMM_N_STATES: int = 4

#: VIX Z-score thresholds
# FIX Bug M1: align with regime.py defaults (vix_red_z=3.0, vix_yellow_z=1.5)
VIX_ZSCORE_RED:    float = 3.0
VIX_ZSCORE_YELLOW: float = 1.5

#: ATR expansion ratio thresholds  (ATR_5 / ATR_60)
# FIX Bug M1: align with regime.py defaults (atr_red_ratio=2.5, atr_yellow_ratio=1.5)
ATR_EXPAND_RED:    float = 2.5
ATR_EXPAND_YELLOW: float = 1.5

#: VVIX level thresholds
VVIX_RED:    float = 130.0
VVIX_YELLOW: float = 110.0

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

#: Primary prediction interval confidence (≈ 1 standard deviation)
CONFIDENCE_PRIMARY: float = 0.68

#: Secondary (wider) prediction interval confidence
CONFIDENCE_SECONDARY: float = 0.90

#: Multiplier for YELLOW-regime interval widths vs GREEN
YELLOW_INTERVAL_MULTIPLIER: float = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# API KEYS  (loaded from .env — never hardcoded here)
# ─────────────────────────────────────────────────────────────────────────────
FRED_API_KEY: str | None = os.getenv("FRED_API_KEY")
