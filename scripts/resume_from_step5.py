#!/usr/bin/env python3
"""
scripts/resume_from_step5.py
─────────────────────────────
Resume the full daily pipeline from Step 5 (signal generation) onwards.

Pipeline steps
──────────────
  1. Data fetch          (run_daily_fetch)
  2. Feature engineering (build_features)
  3. Target engineering  (build_targets)
  4. Model training      (walk_forward or stacking ensemble)
  5. Signal generation   ← this script starts HERE
  6. Reconciliation / outcome logging
  7. Drift monitoring

Use this when steps 1–4 completed successfully on a previous run but signal
generation failed (e.g. a crash after data was already downloaded).

Usage
─────
    python scripts/resume_from_step5.py [--date YYYY-MM-DD] [--mode live|paper]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import date

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("resume_from_step5")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resume SPX Algo pipeline from Step 5 (signal generation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--date", default=None,
        help="Override signal date (YYYY-MM-DD); defaults to today",
    )
    p.add_argument(
        "--mode", choices=["live", "paper"], default="paper",
        help="Execution mode passed to SignalGenerator",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("output/signals"),
        help="Directory to write the signal JSON",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    as_of = args.date or date.today().strftime("%Y-%m-%d")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Resuming from Step 5 — date=%s  mode=%s", as_of, args.mode)

    # ── Step 5: Signal generation ──────────────────────────────────────────────
    try:
        from src.pipeline.signal_generator import SignalGenerator
    except ImportError as exc:
        logger.error("Cannot import SignalGenerator: %s", exc)
        sys.exit(1)

    gen = SignalGenerator(mode=args.mode)
    try:
        signal = gen.generate(as_of_date=as_of)
    except Exception as exc:
        logger.error("Signal generation failed: %s", exc, exc_info=True)
        sys.exit(1)

    if signal is None:
        logger.error("SignalGenerator returned None for %s", as_of)
        sys.exit(1)

    import json
    out_path = args.output_dir / f"signal_{as_of}.json"
    out_path.write_text(signal.to_json(), encoding="utf-8")

    # Update latest_signal.json symlink
    latest = args.output_dir / "latest_signal.json"
    if latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_path.name)

    logger.info(
        "Signal written → %s  (tradeable=%s  regime=%s)",
        out_path, signal.tradeable, signal.regime,
    )

    if not signal.tradeable:
        logger.warning("Signal is NOT tradeable — regime=%s  notes=%s",
                       signal.regime, signal.notes)

    # ── Step 6: Paper-trade log (if regime is not RED) ────────────────────────
    if signal.tradeable:
        try:
            from src.execution.paper_logger import PaperTradeLogger
            pl = PaperTradeLogger()
            pl.log_signal(signal.__dict__ if not hasattr(signal, "items") else signal)
            logger.info("Paper-trade signal row logged")
        except Exception as exc:
            logger.warning("Paper-trade logging skipped: %s", exc)

    logger.info("resume_from_step5.py complete")


if __name__ == "__main__":
    main()
