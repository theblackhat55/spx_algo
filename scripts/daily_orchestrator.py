#!/usr/bin/env python3
"""
scripts/daily_orchestrator.py
──────────────────────────────
High-level daily orchestrator for the SPX Algo.  Coordinates the complete
end-to-end daily run and sends status notifications.

Execution order
───────────────
  1. Validate market is open (skip weekends + holidays)
  2. Refresh raw market data (live_fetcher.run_daily_fetch)
  3. Rebuild features  (features.engineer.build_features)
  4. Generate signal   (signal_generator.SignalGenerator.generate)
  5. Log paper-trade signal row
  6. Log previous day's outcome (if actual OHLCV is available)
  7. Run drift check   (reconciler.Reconciler.reconcile_latest)
  8. Send Telegram / Slack / email summary (if configured)

This script is intended to be scheduled via cron, systemd, or the built-in
Scheduler (src/pipeline/scheduler.py).  It respects US market holidays using
the same _us_market_holidays() function as the rest of the codebase.

Usage
─────
    python scripts/daily_orchestrator.py [--mode paper|live] [--date YYYY-MM-DD]
    python scripts/daily_orchestrator.py --dry-run   # validate config, no execution

Environment variables
─────────────────────
    EXECUTION_MODE          paper (default) | live
    SIGNAL_OUTPUT_DIR       output/signals
    TELEGRAM_TOKEN          (optional)
    TELEGRAM_CHAT_ID        (optional)
    SLACK_WEBHOOK_URL       (optional)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import subprocess
import traceback
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_RAW_DIR = _REPO_ROOT / "data" / "raw"
_DATA_PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
_DEFAULT_SIGNAL_DIR = _REPO_ROOT / "output" / "signals"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("daily_orchestrator")


def _run_hybrid_forecast_step_nonfatal() -> None:
    """Run hybrid OHLC forecast generation without breaking legacy orchestration."""
    cmd = ["python", "scripts/run_hybrid_forecast_step.py"]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print("[WARN] Hybrid OHLC forecast step failed; continuing legacy flow.")
            if result.stderr:
                print(result.stderr)
    except Exception as exc:
        print(f"[WARN] Hybrid OHLC forecast step exception: {exc}")


def _run_gap_augmented_hybrid_forecast_step_nonfatal() -> None:
    """Run Databento gap-augmented hybrid OHLC forecast generation non-fatally."""
    cmd = ["python", "scripts/run_gap_augmented_hybrid_forecast_step.py"]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print("[WARN] Gap-augmented hybrid OHLC forecast step failed; continuing legacy flow.")
            if result.stderr:
                print(result.stderr)
    except Exception as exc:
        print(f"[WARN] Gap-augmented hybrid OHLC forecast step exception: {exc}")



# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_trading_day(d: date) -> bool:
    """Return True if *d* is a US equity trading day."""
    if d.weekday() >= 5:   # Saturday=5, Sunday=6
        return False
    try:
        from src.data.live_fetcher import _us_market_holidays
        return d not in _us_market_holidays(d.year)
    except Exception:
        return True   # fallback: assume trading day


def _send_notification(subject: str, body: str) -> None:
    """Best-effort notification via Telegram → Slack → stdout."""
    import os
    token   = os.getenv("TELEGRAM_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if token and chat_id:
        try:
            import urllib.request, urllib.parse
            msg = f"{subject}\n\n{body}"
            data = urllib.parse.urlencode({"chat_id": chat_id, "text": msg}).encode()
            req  = urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendMessage", data=data)
            urllib.request.urlopen(req, timeout=5)
            logger.info("Telegram notification sent")
            return
        except Exception as exc:
            logger.warning("Telegram notification failed: %s", exc)

    webhook = os.getenv("SLACK_WEBHOOK_URL", "")
    if webhook:
        try:
            import urllib.request, json as _json
            payload = _json.dumps({"text": f"*{subject}*\n{body}"}).encode()
            req = urllib.request.Request(webhook, payload,
                                         {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
            logger.info("Slack notification sent")
            return
        except Exception as exc:
            logger.warning("Slack notification failed: %s", exc)

    # Fallback: plain stdout
    print(f"\n{'='*60}\n{subject}\n{body}\n{'='*60}\n")


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_fetch() -> bool:
    logger.info("Step 2 — data fetch")
    try:
        from src.data.live_fetcher import run_daily_fetch
        run_daily_fetch()
        return True
    except Exception as exc:
        logger.error("Data fetch failed: %s", exc, exc_info=True)
        return False


def step_features() -> bool:
    logger.info("Step 3 — feature engineering")
    try:
        from src.features.builder  import build_feature_matrix

        feats = build_feature_matrix()
        _DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(_DATA_PROCESSED_DIR / "features.parquet")
        logger.info("Features: %d rows × %d cols", *feats.shape)
        return True
    except Exception as exc:
        logger.error("Feature engineering failed: %s", exc, exc_info=True)
        return False


def step_generate(as_of_date: str, mode: str, signal_dir: Path) -> Optional[object]:
    logger.info("Step 4 — signal generation (date=%s mode=%s)", as_of_date, mode)
    try:
        from src.pipeline.signal_generator import SignalGenerator

        gen = SignalGenerator()
        sig = gen.generate(mode=mode, as_of_date=as_of_date)
        if sig is None:
            logger.error("SignalGenerator returned None")
            return None

        signal_dir.mkdir(parents=True, exist_ok=True)
        out_path = signal_dir / f"signal_{as_of_date}.json"
        payload = sig.to_json()
        out_path.write_text(payload, encoding="utf-8")

        latest = signal_dir / "latest_signal.json"
        tmp_latest = signal_dir / "latest_signal.tmp.json"
        tmp_latest.write_text(payload, encoding="utf-8")
        tmp_latest.replace(latest)

        logger.info("Signal → %s  tradeable=%s  regime=%s",
                    out_path, sig.tradeable, sig.regime)
        return sig
    except Exception as exc:
        logger.error("Signal generation failed: %s", exc, exc_info=True)
        return None


def step_log_signal(signal) -> None:
    logger.info("Step 5 — paper-trade signal log")
    try:
        from src.execution.paper_logger import PaperTradeLogger

        pl  = PaperTradeLogger()
        row = signal.__dict__ if hasattr(signal, "__dict__") else dict(signal)
        pl.log_signal(row)
        logger.info("Signal row logged")
    except Exception as exc:
        logger.warning("Paper-trade log_signal skipped: %s", exc)


def step_log_outcome(as_of_date: str) -> None:
    """Log the *previous* trading day's outcome if actual data is available."""
    logger.info("Step 6 — log previous-day outcome")
    try:
        import pandas as pd
        from src.execution.paper_logger import PaperTradeLogger

        spx_path = _DATA_RAW_DIR / "spx_daily.parquet"
        spx = pd.read_parquet(spx_path)
        spx.index = pd.to_datetime(spx.index)
        # Find the row for the signal_date stored in the paper-trade log
        pl = PaperTradeLogger()
        if hasattr(pl, "read_log"):
            df = pl.read_log()
        elif hasattr(pl, "_read_df"):
            df = pl._read_df()
        else:
            raise AttributeError("PaperTradeLogger has neither read_log() nor _read_df()")
        if df.empty:
            logger.info("No signal rows found — skipping outcome logging")
            return
        # Log outcome for rows that have strikes but no actual_high yet
        pending = df[(df["call_strike"] != "") & (df["actual_high"] == "")]["date"].tolist()
        for trade_date in pending:
            if trade_date in spx.index.strftime("%Y-%m-%d").tolist():
                row_idx = spx.index[spx.index.strftime("%Y-%m-%d") == trade_date][0]
                ohlcv   = spx.loc[row_idx].to_dict()
                pl.log_outcome(trade_date, ohlcv)
                logger.info("Outcome logged for %s", trade_date)
    except Exception as exc:
        logger.warning("Outcome logging skipped: %s", exc)


def step_reconcile() -> None:
    logger.info("Step 7 — drift check / reconciliation")
    try:
        from src.pipeline.reconciler import Reconciler
        rec = Reconciler()
        rec.reconcile_latest()
        logger.info("Reconciliation complete")
    except Exception as exc:
        logger.warning("Reconciliation skipped: %s", exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPX Algo daily orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",  choices=["paper", "live"], default="paper")
    p.add_argument("--date",  default=None,
                   help="Override signal date (YYYY-MM-DD); defaults to today")
    p.add_argument("--signal-dir", type=Path, default=_DEFAULT_SIGNAL_DIR)
    p.add_argument("--dry-run", action="store_true",
                   help="Validate configuration only, no pipeline execution")
    return p.parse_args()


def main() -> None:
    _run_gap_augmented_hybrid_forecast_step_nonfatal()
    _run_hybrid_forecast_step_nonfatal()
    args  = _parse_args()
    today = args.date or date.today().strftime("%Y-%m-%d")

    signal_dir = Path(args.signal_dir)
    if not signal_dir.is_absolute():
        signal_dir = (_REPO_ROOT / signal_dir).resolve()

    logger.info("SPX Algo Daily Orchestrator — date=%s  mode=%s", today, args.mode)

    if args.dry_run:
        logger.info("Dry-run mode: validating imports only")
        try:
            from src.data.live_fetcher import run_daily_fetch          # noqa: F401
            from src.features.builder   import build_feature_matrix           # noqa: F401
            from src.pipeline.signal_generator import SignalGenerator   # noqa: F401
            from src.execution.paper_logger    import PaperTradeLogger  # noqa: F401
            logger.info("All imports OK — dry-run complete")
        except ImportError as exc:
            logger.error("Import check failed: %s", exc)
            sys.exit(1)
        return

    # ── Step 1: Market-open validation ────────────────────────────────────────
    run_date = date.fromisoformat(today)
    if not _is_trading_day(run_date):
        logger.info("Not a trading day (%s) — orchestrator exiting", today)
        sys.exit(0)

    errors: list[str] = []

    # ── Steps 2–3: Data + features ────────────────────────────────────────────
    if not step_fetch():
        errors.append("data-fetch failed")
    if not step_features():
        errors.append("feature-engineering failed")

    # ── Step 4: Signal ────────────────────────────────────────────────────────
    signal = step_generate(today, args.mode, signal_dir)
    if signal is None:
        errors.append("signal-generation failed")

    # ── Steps 5–7: Logging + reconciliation ───────────────────────────────────
    if signal is not None:
        step_log_signal(signal)
    step_log_outcome(today)
    step_reconcile()

    # ── Summary notification ──────────────────────────────────────────────────
    if errors:
        body = "Errors:\n" + "\n".join(f"  • {e}" for e in errors)
        _send_notification("⚠️ SPX Algo daily pipeline — PARTIAL FAILURE", body)
        logger.error("Pipeline completed with errors: %s", errors)
        sys.exit(1)
    else:
        regime    = getattr(signal, "regime",    "?") if signal else "?"
        tradeable = getattr(signal, "tradeable", "?") if signal else "?"
        ic_sc     = getattr(signal, "ic_short_call", "?") if signal else "?"
        ic_sp     = getattr(signal, "ic_short_put",  "?") if signal else "?"
        body = (
            f"Date: {today}  |  Regime: {regime}  |  Tradeable: {tradeable}\n"
            f"Short call: {ic_sc}  |  Short put: {ic_sp}"
        )
        _send_notification("✅ SPX Algo daily pipeline — SUCCESS", body)
        logger.info("Pipeline complete — %s", body)


if __name__ == "__main__":
    main()
