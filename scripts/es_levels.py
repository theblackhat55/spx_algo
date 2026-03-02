#!/usr/bin/env python3
"""
scripts/es_levels.py
====================
Generate ES/MES futures trading levels from the SPX algo signal.

Converts predicted high/low percentages, conformal intervals, and
Iron Condor strikes into actionable ES futures levels.

Outputs:
- Key levels (support/resistance)
- Value area (68% probability range)
- Extended range (90% probability range)
- Fade zones (high-probability reversal areas)
- Risk-adjusted position sizing guidance

Usage: python scripts/es_levels.py [--date YYYY-MM-DD] [--json]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("es_levels")

SIGNALS_DIR = Path("output/signals")
INTEL_PATH = Path("data/processed/market_intel.json")

# ES multiplier: 1 ES point = $50, 1 MES point = $5
ES_MULTIPLIER = 50
MES_MULTIPLIER = 5


def load_signal(as_of_date=None):
    """Load the latest or date-specific signal."""
    if as_of_date:
        path = SIGNALS_DIR / f"signal_{as_of_date}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)

    path = SIGNALS_DIR / "latest_signal.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_intel():
    """Load market intel for risk context."""
    if INTEL_PATH.exists():
        with open(INTEL_PATH) as f:
            return json.load(f)
    return None


def round_to_quarter(price):
    """Round to nearest 0.25 (ES tick size)."""
    return round(price * 4) / 4


def generate_levels(signal, intel=None):
    """Generate ES/MES trading levels from signal data."""
    prior_close = signal.get("prior_close", 0)
    signal_date = signal.get("signal_date", date.today().isoformat())
    regime = signal.get("regime", "UNKNOWN")
    direction = signal.get("direction", "NEUTRAL")
    tradeable = signal.get("tradeable", False)
    vix = signal.get("vix_spot", 20)

    # Predicted levels
    pred_high_pct = signal.get("pred_high_pct", 0) or 0
    pred_low_pct = signal.get("pred_low_pct", 0) or 0

    pred_high = round_to_quarter(prior_close * (1 + pred_high_pct))
    pred_low = round_to_quarter(prior_close * (1 + pred_low_pct))
    pred_mid = round_to_quarter((pred_high + pred_low) / 2)

    # Conformal intervals -> value area and extended range
    conf_68_high = round_to_quarter(signal.get("conf_68_high_hi", pred_high))
    conf_68_low = round_to_quarter(signal.get("conf_68_low_lo", pred_low))
    conf_90_high = round_to_quarter(signal.get("conf_90_high_hi", pred_high * 1.002))
    conf_90_low = round_to_quarter(signal.get("conf_90_low_lo", pred_low * 0.998))

    # Iron Condor strikes -> fade zones
    ic_short_call = round_to_quarter(signal.get("ic_short_call", conf_90_high))
    ic_short_put = round_to_quarter(signal.get("ic_short_put", conf_90_low))
    ic_long_call = round_to_quarter(signal.get("ic_long_call", ic_short_call + 10))
    ic_long_put = round_to_quarter(signal.get("ic_long_put", ic_short_put - 10))

    # Risk context
    risk_score = 1
    tail_risk = False
    key_events = []
    if intel:
        risk_score = intel.get("risk_score", 1)
        tail_risk = intel.get("tail_risk_flag", False)
        key_events = intel.get("key_events", [])

    # Position sizing by regime and risk
    if tail_risk or risk_score >= 4:
        size_guidance = "NO TRADE — Tail risk / extreme conditions"
        mes_contracts = 0
        es_contracts = 0
    elif regime == "RED" or risk_score >= 3:
        size_guidance = "MINIMAL — 1 MES only, wide stops"
        mes_contracts = 1
        es_contracts = 0
    elif regime == "YELLOW" or risk_score >= 2:
        size_guidance = "REDUCED — Half normal size"
        mes_contracts = 2
        es_contracts = 0
    else:
        size_guidance = "NORMAL — Full size per your plan"
        mes_contracts = 4
        es_contracts = 1

    # Expected move in points
    expected_range = pred_high - pred_low
    extended_range = conf_90_high - conf_90_low

    # Stop loss suggestions (based on 90% band)
    long_stop = round_to_quarter(conf_90_low - 2)  # 2 pts below 90% band
    short_stop = round_to_quarter(conf_90_high + 2)  # 2 pts above 90% band

    levels = {
        "date": signal_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prior_close": prior_close,
        "regime": regime,
        "direction": direction,
        "vix": round(vix, 2),
        "risk_score": risk_score,
        "tail_risk": tail_risk,
        "tradeable": tradeable and not tail_risk and risk_score < 4,

        # Key levels
        "predicted_high": pred_high,
        "predicted_low": pred_low,
        "predicted_mid": pred_mid,

        # Value area (68% probability)
        "va_high": conf_68_high,
        "va_low": conf_68_low,
        "va_width": round(conf_68_high - conf_68_low, 2),

        # Extended range (90% probability)
        "range_high": conf_90_high,
        "range_low": conf_90_low,
        "range_width": round(conf_90_high - conf_90_low, 2),

        # Fade zones (high probability reversal)
        "fade_short_zone": f"{ic_short_call} - {ic_long_call}",
        "fade_short_entry": ic_short_call,
        "fade_short_stop": round_to_quarter(ic_long_call + 2),

        "fade_long_zone": f"{ic_long_put} - {ic_short_put}",
        "fade_long_entry": ic_short_put,
        "fade_long_stop": round_to_quarter(ic_long_put - 2),

        # Risk management
        "long_stop": long_stop,
        "short_stop": short_stop,
        "expected_range_pts": round(expected_range, 2),
        "extended_range_pts": round(extended_range, 2),

        # Sizing
        "size_guidance": size_guidance,
        "suggested_mes": mes_contracts,
        "suggested_es": es_contracts,
    }

    return levels


def format_whatsapp(levels):
    """Format levels for WhatsApp message."""
    tradeable = levels["tradeable"]
    risk = levels["risk_score"]
    tail = levels["tail_risk"]

    risk_emoji = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "⛔"}.get(risk, "❓")

    lines = [
        f"ES/MES Trading Levels",
        f"Date: {levels['date']}",
        f"Prior Close: {levels['prior_close']:.2f}",
        f"Regime: {levels['regime']} | VIX: {levels['vix']}",
        f"Direction: {levels['direction']}",
        f"Risk: {risk_emoji} {risk}/5" + (" | TAIL RISK" if tail else ""),
        "",
    ]

    if not tradeable:
        lines.append("STATUS: DO NOT TRADE")
        lines.append(f"Reason: {levels['size_guidance']}")
        lines.append("")
        lines.append("Monitor only. Key reference levels:")
        lines.append(f"  Upside wall: {levels['fade_short_entry']:.2f}")
        lines.append(f"  Downside wall: {levels['fade_long_entry']:.2f}")
    else:
        lines.extend([
            f"VALUE AREA (68%): {levels['va_low']:.2f} - {levels['va_high']:.2f} ({levels['va_width']:.0f} pts)",
            f"FULL RANGE (90%): {levels['range_low']:.2f} - {levels['range_high']:.2f} ({levels['range_width']:.0f} pts)",
            "",
            "KEY LEVELS:",
            f"  Pred High: {levels['predicted_high']:.2f}",
            f"  Mid Point: {levels['predicted_mid']:.2f}",
            f"  Pred Low:  {levels['predicted_low']:.2f}",
            "",
            "FADE ZONES (high prob reversal):",
            f"  Short: {levels['fade_short_entry']:.2f} (stop {levels['fade_short_stop']:.2f})",
            f"  Long:  {levels['fade_long_entry']:.2f} (stop {levels['fade_long_stop']:.2f})",
            "",
            f"STOPS: Long SL {levels['long_stop']:.2f} | Short SL {levels['short_stop']:.2f}",
            f"Expected move: {levels['expected_range_pts']:.0f} pts | Extended: {levels['extended_range_pts']:.0f} pts",
            "",
            f"SIZE: {levels['size_guidance']}",
            f"Suggested: {levels['suggested_mes']} MES / {levels['suggested_es']} ES",
        ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Signal date (YYYY-MM-DD)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    signal = load_signal(args.date)
    if signal is None:
        print("ERROR: No signal found. Run daily_cron.sh first.")
        sys.exit(1)

    intel = load_intel()
    levels = generate_levels(signal, intel)

    # Save to file
    out_path = Path("output/signals/es_levels_latest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(levels, f, indent=2)

    if args.json:
        print(json.dumps(levels, indent=2))
    else:
        print(format_whatsapp(levels))


if __name__ == "__main__":
    main()
