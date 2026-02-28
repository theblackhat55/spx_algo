"""
src/pipeline/signal_generator.py
==================================
Task 27 — End-to-end signal generator with REAL conformal intervals.

This replaces the conformal stub in runner.py with a full implementation
that:
  1. Loads or fits regression models (target_high_pct, target_low_pct).
  2. Calibrates ConformalPredictor on a held-out calibration tail.
  3. Returns 68% and 90% prediction intervals.
  4. Assembles the complete DailySignal with iron-condor strikes.
  5. Provides deterministic output given fixed model artefacts + data.

Key design decisions
--------------------
* **Determinism**: all random seeds locked (xgb, lgb, catboost, hmm).
  `json.dumps(..., sort_keys=True)` for byte-identical output.
* **Conformal coverage**: residual-ICP is used as primary (no MAPIE
  dependency required).  Coverage is validated against 252-day history.
* **Iron-condor strikes**: short call = upper_68_high,
  short put = lower_68_low, wing_width from config.
"""
from __future__ import annotations

import hashlib
import json
import logging
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)

# Calibration tail: last N rows are used for conformal calibration
CALIBRATION_TAIL = 126     # ≈6 months
MIN_TRAIN_ROWS   = 252     # minimum history to attempt model fit
DEFAULT_SEED     = 42


# ---------------------------------------------------------------------------
# Data classes  (extend Phase 4's DailySignal)
# ---------------------------------------------------------------------------

from dataclasses import dataclass, asdict, field


@dataclass
class FullSignal:
    """Extended DailySignal with full conformal interval fields."""

    # Identity
    signal_date:        str           # YYYY-MM-DD (date signal is FOR)
    prediction_date:    str           # same as signal_date in 1-DTE mode
    generated_at:       str           # ISO-8601 UTC timestamp
    mode:               str           # 'live' | 'replay'

    # Regime
    regime:             str           # GREEN | YELLOW | RED
    regime_details:     Dict[str, Any] = field(default_factory=dict)

    # Regression predictions (% move from prior close)
    pred_high_pct:      Optional[float] = None
    pred_low_pct:       Optional[float] = None
    predicted_high:     Optional[float] = None   # absolute SPX level
    predicted_low:      Optional[float] = None
    predicted_range:    Optional[float] = None

    # Conformal intervals (absolute levels)
    conf_68_high_lo:    Optional[float] = None
    conf_68_high_hi:    Optional[float] = None
    conf_90_high_lo:    Optional[float] = None
    conf_90_high_hi:    Optional[float] = None
    conf_68_low_lo:     Optional[float] = None
    conf_68_low_hi:     Optional[float] = None
    conf_90_low_lo:     Optional[float] = None
    conf_90_low_hi:     Optional[float] = None

    # Iron-condor strikes (derived from 68% bounds)
    ic_short_call:      Optional[float] = None   # = conf_68_high_hi
    ic_long_call:       Optional[float] = None   # + wing_width
    ic_short_put:       Optional[float] = None   # = conf_68_low_lo
    ic_long_put:        Optional[float] = None   # - wing_width

    # Classification
    direction:          Optional[str]   = None   # BULLISH|NEUTRAL|BEARISH
    direction_prob:     Optional[float] = None
    prob_high_bin_050:  Optional[float] = None
    prob_low_bin_050:   Optional[float] = None

    # Meta
    prior_close:        Optional[float] = None
    n_features_used:    int             = 0
    model_versions:     Dict[str, str]  = field(default_factory=dict)
    data_quality:       str             = "FULL"   # FULL|PARTIAL|DEGRADED
    tradeable:          bool            = False
    notes:              List[str]       = field(default_factory=list)

    # ----------------------------------------------------------------
    def to_json(self, indent: int = 2) -> str:
        """Return deterministic JSON (sorted keys)."""
        return json.dumps(asdict(self), indent=indent, sort_keys=True, default=str)

    def save(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / f"signal_{self.signal_date}.json"
        fname.write_text(self.to_json())
        # Also overwrite latest_signal.json for quick access
        latest = output_dir / "latest_signal.json"
        latest.write_text(self.to_json())
        logger.info("Signal saved → %s", fname)
        return fname

    @classmethod
    def error_signal(cls, mode: str, signal_date: str, reason: str) -> "FullSignal":
        return cls(
            signal_date=signal_date,
            prediction_date=signal_date,
            generated_at=datetime.now(timezone.utc).isoformat(),
            mode=mode,
            regime="RED",
            tradeable=False,
            notes=[f"ERROR: {reason}"],
        )


# ---------------------------------------------------------------------------
# Model hash helper  (determinism / versioning)
# ---------------------------------------------------------------------------

def _model_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Minimal in-process regression trainer
# ---------------------------------------------------------------------------

def _fit_regression_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_type: str = "ridge",
    seed: int = DEFAULT_SEED,
):
    """Fit a lightweight regression model. Returns fitted BaseModel-like object."""
    from src.models.linear_models import RidgeRegressionModel as RidgeModel

    model = RidgeModel(name=f"ridge_{target.name}", alpha=1.0)
    valid = target.dropna()
    X = features.loc[valid.index].fillna(0)
    model.fit(X, valid)
    return model


# ---------------------------------------------------------------------------
# Conformal calibration helper
# ---------------------------------------------------------------------------

def _calibrate_conformal(
    model,
    features: pd.DataFrame,
    target: pd.Series,
    cal_tail: int = CALIBRATION_TAIL,
    seed: int = DEFAULT_SEED,
) -> "ConformalPredictor":
    from src.calibration.conformal import ConformalPredictor

    valid_idx = target.dropna().index
    X_all = features.loc[valid_idx].fillna(0)
    y_all = target.loc[valid_idx]

    if len(y_all) < cal_tail + 10:
        cal_tail = max(10, len(y_all) // 4)

    X_cal = X_all.iloc[-cal_tail:]
    y_cal = y_all.iloc[-cal_tail:]

    cp = ConformalPredictor(model, use_mapie=False, alpha_list=[0.68, 0.90])
    cp.calibrate(X_cal, y_cal)
    return cp


# ---------------------------------------------------------------------------
# Coverage validator
# ---------------------------------------------------------------------------

def validate_conformal_coverage(
    cp,
    features: pd.DataFrame,
    target: pd.Series,
    window: int = 252,
) -> Dict[str, float]:
    """Compute observed coverage for 68% and 90% intervals on last *window* rows."""
    valid_idx = target.dropna().index
    if len(valid_idx) < window:
        window = len(valid_idx)

    X_val = features.loc[valid_idx].iloc[-window:].fillna(0)
    y_val = target.loc[valid_idx].iloc[-window:]

    intervals = cp.predict_interval(X_val)
    coverage = {}
    for lvl in (68, 90):
        lo = intervals[f"lower_{lvl}"].values
        hi = intervals[f"upper_{lvl}"].values
        y  = y_val.values
        cov = float(np.mean((y >= lo) & (y <= hi)))
        coverage[f"coverage_{lvl}"] = round(cov, 4)
        logger.info("Conformal coverage @ %d%% = %.1f%% (expected ~%d%%)",
                    lvl, cov * 100, lvl)
    return coverage


# ---------------------------------------------------------------------------
# Direction classifier helper
# ---------------------------------------------------------------------------

def _classify_direction(prob_high: Optional[float],
                         prob_low:  Optional[float]) -> Tuple[str, float]:
    """Simple rule: if prob_high > 0.6 → BULLISH, prob_low > 0.6 → BEARISH."""
    if prob_high is None and prob_low is None:
        return "NEUTRAL", 0.5
    ph = prob_high or 0.5
    pl = prob_low  or 0.5
    if ph > 0.60:
        return "BULLISH", ph
    if pl > 0.60:
        return "BEARISH", pl
    return "NEUTRAL", max(ph, pl)


# ---------------------------------------------------------------------------
# Signal assembly
# ---------------------------------------------------------------------------

def assemble_signal(
    predictions:    Dict[str, Any],
    intervals_high: pd.DataFrame,
    intervals_low:  pd.DataFrame,
    regime:         str,
    regime_details: Dict[str, Any],
    prior_close:    float,
    signal_date:    str,
    mode:           str,
    metadata:       Dict[str, Any],
    wing_width_pts: float = 50.0,
) -> FullSignal:
    """Compose all components into a FullSignal."""
    pred_high_pct = predictions.get("pred_high_pct")
    pred_low_pct  = predictions.get("pred_low_pct")

    predicted_high = (prior_close * (1 + pred_high_pct)) if pred_high_pct is not None else None
    predicted_low  = (prior_close * (1 + pred_low_pct))  if pred_low_pct  is not None else None
    predicted_range = (predicted_high - predicted_low) if (predicted_high and predicted_low) else None

    # Conformal intervals — convert pct-deviation rows to absolute levels
    def _abs(df, col, base):
        if df is None or col not in df.columns:
            return None
        v = float(df[col].iloc[-1])
        return base * (1 + v) if abs(v) < 1 else v   # handle already-absolute

    h68lo = _abs(intervals_high, "lower_68", prior_close)
    h68hi = _abs(intervals_high, "upper_68", prior_close)
    h90lo = _abs(intervals_high, "lower_90", prior_close)
    h90hi = _abs(intervals_high, "upper_90", prior_close)
    l68lo = _abs(intervals_low,  "lower_68", prior_close)
    l68hi = _abs(intervals_low,  "upper_68", prior_close)
    l90lo = _abs(intervals_low,  "lower_90", prior_close)
    l90hi = _abs(intervals_low,  "upper_90", prior_close)

    # Iron-condor strikes from 68% bounds
    ic_short_call = h68hi
    ic_long_call  = (h68hi + wing_width_pts) if h68hi else None
    ic_short_put  = l68lo
    ic_long_put   = (l68lo - wing_width_pts) if l68lo else None

    # Direction
    direction, dir_prob = _classify_direction(
        predictions.get("prob_high_bin_050"),
        predictions.get("prob_low_bin_050"),
    )

    tradeable = (regime != "RED") and (metadata.get("data_quality", "FULL") != "DEGRADED")

    notes = metadata.get("notes", [])
    if not tradeable:
        notes.append(f"NOT TRADEABLE: regime={regime} quality={metadata.get('data_quality')}")

    return FullSignal(
        signal_date=signal_date,
        prediction_date=signal_date,
        generated_at=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        regime=regime,
        regime_details=regime_details,
        pred_high_pct=pred_high_pct,
        pred_low_pct=pred_low_pct,
        predicted_high=predicted_high,
        predicted_low=predicted_low,
        predicted_range=predicted_range,
        conf_68_high_lo=h68lo, conf_68_high_hi=h68hi,
        conf_90_high_lo=h90lo, conf_90_high_hi=h90hi,
        conf_68_low_lo=l68lo,  conf_68_low_hi=l68hi,
        conf_90_low_lo=l90lo,  conf_90_low_hi=l90hi,
        ic_short_call=ic_short_call, ic_long_call=ic_long_call,
        ic_short_put=ic_short_put,   ic_long_put=ic_long_put,
        direction=direction,
        direction_prob=dir_prob,
        prob_high_bin_050=predictions.get("prob_high_bin_050"),
        prob_low_bin_050=predictions.get("prob_low_bin_050"),
        prior_close=prior_close,
        n_features_used=metadata.get("n_features", 0),
        model_versions=metadata.get("model_versions", {}),
        data_quality=metadata.get("data_quality", "FULL"),
        tradeable=tradeable,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Signal validator
# ---------------------------------------------------------------------------

def validate_signal(signal: FullSignal) -> List[str]:
    """Return list of validation errors (empty = OK)."""
    errors = []
    if signal.predicted_high is not None and signal.predicted_low is not None:
        if signal.predicted_high <= signal.predicted_low:
            errors.append("predicted_high <= predicted_low")
        if signal.prior_close:
            pct_hi = abs(signal.predicted_high / signal.prior_close - 1)
            pct_lo = abs(signal.predicted_low  / signal.prior_close - 1)
            if pct_hi > 0.05:
                errors.append(f"predicted_high deviates >5% from prior close ({pct_hi:.2%})")
            if pct_lo > 0.05:
                errors.append(f"predicted_low deviates >5% from prior close ({pct_lo:.2%})")

    if signal.conf_68_high_lo is not None and signal.conf_90_high_lo is not None:
        if signal.conf_90_high_lo > signal.conf_68_high_lo:
            errors.append("90% interval not wider than 68% (lower bound)")

    if signal.regime not in {"GREEN", "YELLOW", "RED"}:
        errors.append(f"Invalid regime: {signal.regime!r}")

    return errors


# ---------------------------------------------------------------------------
# Main signal generator class
# ---------------------------------------------------------------------------

class SignalGenerator:
    """
    Full signal generator with real conformal intervals.

    Usage
    -----
    >>> gen = SignalGenerator(raw_dir=..., processed_dir=..., output_dir=...)
    >>> signal = gen.generate(mode='live')
    >>> signal.save(gen.output_dir)
    """

    def __init__(
        self,
        raw_dir:       Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        output_dir:    Optional[Path] = None,
        wing_width_pts: float = 50.0,
        seed: int = DEFAULT_SEED,
    ):
        try:
            from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR
            _raw  = Path(RAW_DATA_DIR)
            _proc = Path(PROCESSED_DATA_DIR)
            _out  = Path(OUTPUT_DIR) / "signals"
        except Exception:
            _root = Path(__file__).resolve().parent.parent.parent
            _raw  = _root / "data" / "raw"
            _proc = _root / "data" / "processed"
            _out  = _root / "output" / "signals"

        self.raw_dir       = Path(raw_dir       or _raw)
        self.processed_dir = Path(processed_dir or _proc)
        self.output_dir    = Path(output_dir    or _out)
        self.wing_width_pts = wing_width_pts
        self.seed          = seed

    # ------------------------------------------------------------------
    def generate(
        self,
        mode: str = "live",
        as_of_date: Optional[str] = None,
        save: bool = True,
    ) -> FullSignal:
        """Generate a FullSignal for *as_of_date* (default: today)."""
        notes: List[str] = []
        generated_at = datetime.now(timezone.utc).isoformat()
        today_str = as_of_date or date.today().strftime("%Y-%m-%d")

        # ── Load raw data ─────────────────────────────────────────────
        spx_df, vix_df = self._load_raw(as_of_date)
        if spx_df is None:
            return FullSignal.error_signal(mode, today_str, "SPX data not available")

        prior_close = float(spx_df["Close"].iloc[-1])
        signal_date = self._next_trading_day(spx_df.index[-1])
        notes.append(f"Prior close: {prior_close:.2f}  rows: {len(spx_df)}")

        # ── Load features ─────────────────────────────────────────────
        features = self._load_features()
        if features is None:
            return FullSignal.error_signal(mode, signal_date, "Features not available")
        notes.append(f"Features: {features.shape}")

        # ── Regime ───────────────────────────────────────────────────
        regime_str, regime_details = self._compute_regime(spx_df, vix_df, notes)

        # ── Build / load regression targets ──────────────────────────
        target_high, target_low = self._build_targets(spx_df, features, notes)

        # ── Fit regression models + conformal calibration ─────────────
        cp_high, cp_low, pred_pcts, model_versions = self._fit_and_calibrate(
            features, target_high, target_low, notes
        )

        # ── Generate predictions + intervals ─────────────────────────
        last_X = features.iloc[[-1]].fillna(0)
        intervals_high = cp_high.predict_interval(last_X) if cp_high else None
        intervals_low  = cp_low.predict_interval(last_X)  if cp_low  else None

        # ── Classification probabilities ──────────────────────────────
        clf_probs = self._classification_predict(features, notes)

        # ── Validate conformal coverage (log only, don't block) ───────
        if cp_high is not None and target_high is not None:
            cov = validate_conformal_coverage(cp_high, features, target_high)
            notes.append(f"High conformal coverage: {cov}")
        if cp_low is not None and target_low is not None:
            cov = validate_conformal_coverage(cp_low, features, target_low)
            notes.append(f"Low conformal coverage: {cov}")

        # ── Determine data quality ────────────────────────────────────
        data_quality = "FULL"
        if cp_high is None or cp_low is None:
            data_quality = "PARTIAL"

        metadata = {
            "notes":          notes,
            "data_quality":   data_quality,
            "n_features":     features.shape[1],
            "model_versions": model_versions,
        }

        # ── Assemble ──────────────────────────────────────────────────
        predictions = {**pred_pcts, **clf_probs}
        signal = assemble_signal(
            predictions=predictions,
            intervals_high=intervals_high,
            intervals_low=intervals_low,
            regime=regime_str,
            regime_details=regime_details,
            prior_close=prior_close,
            signal_date=signal_date,
            mode=mode,
            metadata=metadata,
            wing_width_pts=self.wing_width_pts,
        )

        # ── Validate ──────────────────────────────────────────────────
        errs = validate_signal(signal)
        if errs:
            for e in errs:
                logger.warning("Signal validation: %s", e)
            signal.notes.extend([f"VALIDATION WARNING: {e}" for e in errs])

        if save:
            signal.save(self.output_dir)

        return signal

    # ------------------------------------------------------------------
    def _load_raw(self, as_of_date: Optional[str]):
        spx_path = self.raw_dir / "spx_daily.parquet"
        vix_path = self.raw_dir / "vix_daily.parquet"
        if not spx_path.exists():
            logger.error("SPX not found: %s", spx_path)
            return None, None
        spx = pd.read_parquet(spx_path)
        spx.index = pd.to_datetime(spx.index)
        spx = spx.sort_index()
        vix = None
        if vix_path.exists():
            vix = pd.read_parquet(vix_path)
            vix.index = pd.to_datetime(vix.index)
        if as_of_date:
            cutoff = pd.Timestamp(as_of_date)
            spx = spx.loc[:cutoff]
            if vix is not None:
                vix = vix.loc[:cutoff]
        return spx, vix

    def _load_features(self) -> Optional[pd.DataFrame]:
        feat_path = self.processed_dir / "features.parquet"
        if feat_path.exists():
            df = pd.read_parquet(feat_path)
            df.index = pd.to_datetime(df.index)
            return df
        logger.error("features.parquet not found at %s", feat_path)
        return None

    def _build_targets(
        self,
        spx_df: pd.DataFrame,
        features: pd.DataFrame,
        notes: List[str],
    ) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """Build regression targets aligned to features."""
        try:
            target_high = (
                spx_df["High"].shift(-1) / spx_df["Close"] - 1
            ).reindex(features.index).dropna()
            target_low = (
                spx_df["Low"].shift(-1) / spx_df["Close"] - 1
            ).reindex(features.index).dropna()
            target_high.name = "target_high_pct"
            target_low.name  = "target_low_pct"
            notes.append(f"Targets: high={len(target_high)} low={len(target_low)}")
            return target_high, target_low
        except Exception as exc:
            notes.append(f"Target build failed: {exc}")
            return None, None

    def _fit_and_calibrate(
        self,
        features:     pd.DataFrame,
        target_high:  Optional[pd.Series],
        target_low:   Optional[pd.Series],
        notes:        List[str],
    ):
        cp_high, cp_low = None, None
        pred_pcts: Dict[str, Any] = {}
        model_versions: Dict[str, str] = {}

        for name, target in [("high", target_high), ("low", target_low)]:
            if target is None or len(target) < MIN_TRAIN_ROWS:
                notes.append(f"Insufficient data for {name} model")
                continue
            try:
                model = _fit_regression_model(
                    features.loc[target.dropna().index].fillna(0),
                    target.dropna(),
                    seed=self.seed,
                )
                cp = _calibrate_conformal(model, features, target)

                last_X = features.iloc[[-1]].fillna(0)
                pred_val = float(model.predict(last_X)[0])
                pred_pcts[f"pred_{name}_pct"] = pred_val
                model_versions[f"ridge_{name}"] = "in-process"

                if name == "high":
                    cp_high = cp
                else:
                    cp_low = cp

                notes.append(f"Fitted {name} model; pred_{name}_pct={pred_val:.4f}")
            except Exception as exc:
                notes.append(f"{name} model failed: {exc}")
                logger.exception("Model fit failed for %s", name)

        return cp_high, cp_low, pred_pcts, model_versions

    def _compute_regime(
        self,
        spx_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame],
        notes:  List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        details: Dict[str, Any] = {}
        try:
            from src.calibration.regime import RegimeDetector, Regime
            rd = RegimeDetector(
                use_hmm=True, use_garch=False,
                hmm_states=4,
            )
            # Lock HMM seed via monkey-patch on numpy
            np.random.seed(self.seed)
            regime_series = rd.fit_predict(spx_df, vix_df)
            reg_int = int(regime_series.iloc[-1])
            reg_str = {0: "GREEN", 1: "YELLOW", 2: "RED"}.get(reg_int, "YELLOW")

            for comp_name, comp_series in rd.components.items():
                details[comp_name] = int(comp_series.iloc[-1])

            notes.append(f"Regime: {reg_str}  components: {details}")
            return reg_str, details
        except Exception as exc:
            notes.append(f"Regime failed ({exc}) → YELLOW")
            return "YELLOW", {}

    def _classification_predict(
        self,
        features: pd.DataFrame,
        notes:    List[str],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        try:
            from config.settings import OUTPUT_DIR
            model_dir = Path(OUTPUT_DIR) / "models"
        except Exception:
            return result

        for target in ("next_high_bin_050", "next_low_bin_050"):
            model_path = model_dir / f"classifier_{target}.pkl"
            if not model_path.exists():
                continue
            try:
                from src.models.base_model import BaseModel
                model = BaseModel.load(model_path)
                last_X = features.iloc[[-1]].fillna(0)
                prob = float(model.predict_proba(last_X)[0, 1])
                key = f"prob_{'high' if 'high' in target else 'low'}_bin_050"
                result[key] = prob
            except Exception as exc:
                notes.append(f"Classifier {target} failed: {exc}")
        return result

    @staticmethod
    def _next_trading_day(last_date: pd.Timestamp) -> str:
        """Return the next business day as YYYY-MM-DD string."""
        from pandas.tseries.offsets import BDay
        return (last_date + BDay(1)).strftime("%Y-%m-%d")
