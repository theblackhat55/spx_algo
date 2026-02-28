"""
src/pipeline/scheduler.py
==========================
Task 23 — Daily signal scheduler with dead-man's switch.

Runs the full pipeline at a configured time each trading day (default 4:05 PM ET).
Sends an alert if the signal JSON is not written by the deadline (default 4:30 PM ET).

Dead-man's switch
-----------------
The ``DeadMansSwitch`` class monitors the output directory.  If signal_{today}.json
does not exist by the deadline timestamp, it fires an alert via:
    1. Console / logger (always)
    2. Email via smtplib (if SMTP credentials configured in .env)
    3. Slack webhook (if SLACK_WEBHOOK_URL configured in .env)

Usage
-----
Run as a background service:
    python -m src.pipeline.scheduler

Or integrate with system cron:
    5 21 * * 1-5  /usr/bin/python /path/to/spx_algo/src/pipeline/scheduler.py
    (21:05 UTC = 4:05 PM ET, adjusted for daylight saving time)

Environment variables (all optional)
--------------------------------------
SCHEDULE_TIME_ET       : "16:05"  — when to run the pipeline
DEADLINE_TIME_ET       : "16:30"  — dead-man's switch deadline
SMTP_HOST, SMTP_PORT   : SMTP relay
SMTP_USER, SMTP_PASS   : credentials
ALERT_EMAIL_TO         : recipient for dead-man's alerts
SLACK_WEBHOOK_URL      : Slack incoming webhook URL
"""
from __future__ import annotations

import logging
import os
import smtplib
import sys                          # FIX Bug C1: was missing
import time
import traceback
from datetime import datetime, date
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Config (environment-driven)
# ---------------------------------------------------------------------------

SCHEDULE_TIME_ET  = os.getenv("SCHEDULE_TIME_ET",  "16:05")
DEADLINE_TIME_ET  = os.getenv("DEADLINE_TIME_ET",  "16:30")
SIGNAL_DIR        = Path(os.getenv("SIGNAL_OUTPUT_DIR",
                                   str(Path(__file__).resolve().parent.parent.parent
                                       / "output" / "signals")))

SMTP_HOST         = os.getenv("SMTP_HOST",       "")
SMTP_PORT         = int(os.getenv("SMTP_PORT",   "587"))
SMTP_USER         = os.getenv("SMTP_USER",       "")
SMTP_PASS         = os.getenv("SMTP_PASS",       "")
ALERT_EMAIL_TO    = os.getenv("ALERT_EMAIL_TO",  "")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL","")


# ---------------------------------------------------------------------------
# Dead-man's switch
# ---------------------------------------------------------------------------

class DeadMansSwitch:
    """
    Watches the signal output directory.
    Fires an alert if today's signal file is missing by the deadline.

    Usage
    -----
    switch = DeadMansSwitch(deadline_time_et="16:30")
    switch.check()   # call at or after the deadline
    """

    def __init__(
        self,
        signal_dir:       Path = SIGNAL_DIR,
        deadline_time_et: str  = DEADLINE_TIME_ET,
    ):
        self.signal_dir       = Path(signal_dir)
        self.deadline_time_et = deadline_time_et

    # ------------------------------------------------------------------
    def today_signal_path(self) -> Path:
        today = date.today().strftime("%Y-%m-%d")
        return self.signal_dir / f"signal_{today}.json"

    def signal_exists(self) -> bool:
        return self.today_signal_path().exists()

    # ------------------------------------------------------------------
    def check(self) -> bool:
        """
        Check if today's signal exists.

        Returns True if OK, False if the switch fires (signal missing).
        Fires alerts on failure.
        """
        if self.signal_exists():
            logger.info("Dead-man check: signal OK — %s", self.today_signal_path())
            return True

        msg = (
            f"⚠️  SPX ALGO DEAD-MAN ALERT — {date.today()}\n\n"
            f"Expected signal file not found by {self.deadline_time_et} ET:\n"
            f"  {self.today_signal_path()}\n\n"
            f"The daily pipeline may have failed. "
            f"Check logs immediately before market open tomorrow."
        )
        logger.error("DEAD-MAN SWITCH FIRED: %s", msg)

        self._alert_email(msg)
        self._alert_slack(msg)
        return False

    # ------------------------------------------------------------------
    def _alert_email(self, body: str) -> None:
        if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, ALERT_EMAIL_TO]):
            logger.debug("Email alert skipped (SMTP not configured).")
            return
        try:
            mime = MIMEText(body)
            mime["Subject"] = f"[SPX ALGO] Dead-man alert — {date.today()}"
            mime["From"]    = SMTP_USER
            mime["To"]      = ALERT_EMAIL_TO

            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
                server.send_message(mime)

            logger.info("Dead-man email sent to %s", ALERT_EMAIL_TO)
        except Exception:
            logger.error("Dead-man email FAILED:\n%s", traceback.format_exc())

    # ------------------------------------------------------------------
    def _alert_slack(self, body: str) -> None:
        if not SLACK_WEBHOOK_URL:
            logger.debug("Slack alert skipped (SLACK_WEBHOOK_URL not configured).")
            return
        try:
            import json
            import urllib.request

            payload = json.dumps({"text": body}).encode("utf-8")
            req     = urllib.request.Request(
                SLACK_WEBHOOK_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
            logger.info("Slack alert sent (HTTP %d).", status)
        except Exception:
            logger.error("Slack alert FAILED:\n%s", traceback.format_exc())


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class DailyScheduler:
    """
    Runs the pipeline every trading day at a specified time.

    Two modes:
        run_once  : Run immediately (used for manual or cron invocation).
        run_loop  : Blocking loop that sleeps until the next scheduled time.

    The dead-man's switch is checked separately at the deadline time.
    """

    def __init__(
        self,
        schedule_time_et: str  = SCHEDULE_TIME_ET,
        deadline_time_et: str  = DEADLINE_TIME_ET,
        signal_dir:       Path = SIGNAL_DIR,
    ):
        self.schedule_time_et = schedule_time_et
        self.switch           = DeadMansSwitch(signal_dir, deadline_time_et)

    # ------------------------------------------------------------------
    def run_once(self) -> bool:
        """Run the pipeline once and return True on success.

        FIX Bug C3: use SignalGenerator.generate() instead of the stub
        PipelineRunner.run() which uses hardcoded conformal intervals.
        Falls back to PipelineRunner if SignalGenerator is unavailable.
        """
        logger.info("Scheduler: starting signal generation.")
        try:
            # Primary path: full SignalGenerator with real conformal intervals
            from src.pipeline.signal_generator import SignalGenerator
            sg = SignalGenerator()
            signal = sg.generate()

            if signal is not None:
                tradeable = getattr(signal, "tradeable", False)
                if tradeable:
                    logger.info(
                        "Signal generated: %s | high=%.5f | low=%.5f | "
                        "call=%.2f | put=%.2f",
                        getattr(signal, "target_date", "?"),
                        getattr(signal, "predicted_high", 0) or 0,
                        getattr(signal, "predicted_low",  0) or 0,
                        getattr(signal, "ic_short_call",  0) or 0,
                        getattr(signal, "ic_short_put",   0) or 0,
                    )
                else:
                    regime = getattr(signal, "regime", "UNKNOWN")
                    logger.warning("Signal NOT TRADEABLE — regime=%s", regime)
                return True

            logger.warning("SignalGenerator returned None — pipeline may have no data")
            return False

        except ImportError:
            # Graceful fallback: PipelineRunner (less accurate intervals)
            logger.warning(
                "SignalGenerator unavailable; falling back to PipelineRunner."
            )
            try:
                from src.pipeline.runner import PipelineRunner
                runner = PipelineRunner()
                signal = runner.run(mode="live", save_signal=True)
                if signal.tradeable:
                    logger.info(
                        "PipelineRunner signal: %s | high=%.2f | low=%.2f",
                        signal.signal_date,
                        signal.pred_high or 0,
                        signal.pred_low  or 0,
                    )
                else:
                    logger.warning(
                        "PipelineRunner signal NOT TRADEABLE — regime=%s",
                        signal.regime,
                    )
                return True
            except Exception:
                logger.error("PipelineRunner fallback FAILED:\n%s", traceback.format_exc())
                return False

        except Exception:
            logger.error("SignalGenerator run FAILED:\n%s", traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    def run_loop(self, poll_seconds: int = 30) -> None:
        """
        Blocking loop.  Runs the pipeline once per day at schedule_time_et,
        then checks the dead-man's switch at deadline_time_et.

        Intended to be run as a long-lived process (systemd service / Docker).
        """
        import pytz
        from datetime import timedelta

        tz      = pytz.timezone("America/New_York")
        logger.info("Scheduler loop started.  Schedule=%s ET  Deadline=%s ET",
                    self.schedule_time_et, self.switch.deadline_time_et)

        ran_today        = False
        checked_deadman  = False
        last_date        = None

        while True:
            now_et = datetime.now(tz)
            today  = now_et.date()

            # Reset daily flags at midnight
            if today != last_date:
                ran_today       = False
                checked_deadman = False
                last_date       = today

            # Skip weekends
            if now_et.weekday() >= 5:
                time.sleep(300)
                continue

            sched_h, sched_m = map(int, self.schedule_time_et.split(":"))
            dead_h,  dead_m  = map(int,
                self.switch.deadline_time_et.split(":"))

            now_minutes = now_et.hour * 60 + now_et.minute

            # Run pipeline
            if not ran_today and now_minutes >= sched_h * 60 + sched_m:
                ran_today = True
                self.run_once()

            # Dead-man check
            if not checked_deadman and now_minutes >= dead_h * 60 + dead_m:
                checked_deadman = True
                self.switch.check()

            time.sleep(poll_seconds)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="SPX Algo daily scheduler")
    parser.add_argument("--once",      action="store_true",
                        help="Run the pipeline once and exit")
    parser.add_argument("--check",     action="store_true",
                        help="Run dead-man check only and exit")
    parser.add_argument("--loop",      action="store_true",
                        help="Run in blocking loop mode (default)")
    args = parser.parse_args()

    scheduler = DailyScheduler()

    if args.once:
        ok = scheduler.run_once()
        sys.exit(0 if ok else 1)          # sys is now imported above
    elif args.check:
        ok = scheduler.switch.check()
        sys.exit(0 if ok else 1)
    else:
        scheduler.run_loop()


if __name__ == "__main__":
    main()
