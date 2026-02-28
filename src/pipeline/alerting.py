"""
src/pipeline/alerting.py  (patched)
=====================================
Added send_text() method to Alerter so reconciler.py can call it directly.
send_text() broadcasts a plain-text message to all configured channels,
identical to send_failure() but without the "🚨" prefix so informational
summaries don't look like emergencies.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import socket
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class AlertConfig:
    """Alert channel configuration (all fields optional)."""
    discord_webhook_url:  Optional[str] = None
    telegram_bot_token:   Optional[str] = None
    telegram_chat_id:     Optional[str] = None
    smtp_host:            Optional[str] = None
    smtp_port:            int           = 587
    smtp_user:            Optional[str] = None
    smtp_pass:            Optional[str] = None
    alert_email_to:       Optional[str] = None
    alert_email_from:     Optional[str] = None
    timeout_seconds:      int           = 10


def alert_config_from_env() -> AlertConfig:
    """Build AlertConfig from environment variables."""
    return AlertConfig(
        discord_webhook_url  = os.getenv("DISCORD_WEBHOOK_URL"),
        telegram_bot_token   = os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id     = os.getenv("TELEGRAM_CHAT_ID"),
        smtp_host            = os.getenv("SMTP_HOST"),
        smtp_port            = int(os.getenv("SMTP_PORT", "587")),
        smtp_user            = os.getenv("SMTP_USER"),
        smtp_pass            = os.getenv("SMTP_PASS"),
        alert_email_to       = os.getenv("ALERT_EMAIL_TO"),
        alert_email_from     = os.getenv("SMTP_USER"),
    )


# ---------------------------------------------------------------------------
# Regime color helpers
# ---------------------------------------------------------------------------

_REGIME_COLORS = {
    "GREEN":  0x2ECC71,
    "YELLOW": 0xF1C40F,
    "RED":    0xE74C3C,
}

_REGIME_EMOJI = {
    "GREEN":  "🟢",
    "YELLOW": "🟡",
    "RED":    "🔴",
}


def _fmt(v: Any, decimals: int = 2, pct: bool = False) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:+.{decimals}f}%"
    return f"{v:.{decimals}f}"


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

def send_discord_webhook(signal, webhook_url: str, timeout: int = 10) -> bool:
    regime  = getattr(signal, "regime", "YELLOW")
    color   = _REGIME_COLORS.get(regime, 0x95A5A6)
    emoji   = _REGIME_EMOJI.get(regime, "⚪")

    fields = [
        {"name": "Regime",    "value": f"{emoji} **{regime}**",                  "inline": True},
        {"name": "Tradeable", "value": "✅ Yes" if signal.tradeable else "❌ No", "inline": True},
        {"name": "Data Quality", "value": getattr(signal, "data_quality", "—"),  "inline": True},
    ]

    if hasattr(signal, "predicted_high") and signal.predicted_high:
        fields += [
            {"name": "Pred High", "value": _fmt(signal.predicted_high), "inline": True},
            {"name": "Pred Low",  "value": _fmt(signal.predicted_low),  "inline": True},
            {"name": "Range",     "value": _fmt(signal.predicted_range), "inline": True},
        ]

    if hasattr(signal, "conf_68_high_hi") and signal.conf_68_high_hi:
        fields += [
            {"name": "68% Call Strike", "value": _fmt(signal.ic_short_call), "inline": True},
            {"name": "68% Put Strike",  "value": _fmt(signal.ic_short_put),  "inline": True},
            {"name": "Direction",       "value": f"{getattr(signal,'direction','—')} "
                                                  f"({_fmt(getattr(signal,'direction_prob',None))})",
             "inline": True},
        ]

    prior_close = getattr(signal, "prior_close", None)
    desc = (
        f"**Prior Close:** {_fmt(prior_close)}\n"
        f"**Signal Date:** {signal.signal_date}\n"
        f"**Generated:** {getattr(signal, 'generated_at', '—')}"
    )
    model_ver = getattr(signal, "model_versions", {})
    footer = f"data_quality={getattr(signal,'data_quality','—')} | models={json.dumps(model_ver)}"

    embed = {
        "title":       f"📊 SPX Daily Signal — {signal.signal_date}",
        "description": desc,
        "color":       color,
        "fields":      fields,
        "footer":      {"text": footer},
    }

    try:
        resp = requests.post(webhook_url, json={"embeds": [embed]}, timeout=timeout)
        resp.raise_for_status()
        logger.info("Discord webhook sent (status %s)", resp.status_code)
        return True
    except Exception as exc:
        logger.warning("Discord webhook failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_telegram_message(signal, bot_token: str, chat_id: str, timeout: int = 10) -> bool:
    regime = getattr(signal, "regime", "YELLOW")
    emoji  = _REGIME_EMOJI.get(regime, "⚪")

    lines = [
        f"*📊 SPX Daily Signal — {signal.signal_date}*",
        f"{emoji} Regime: *{regime}*",
        f"Tradeable: {'✅' if signal.tradeable else '❌'}",
        f"Prior Close: `{_fmt(getattr(signal, 'prior_close', None))}`",
    ]

    if hasattr(signal, "predicted_high") and signal.predicted_high:
        lines += [
            f"Pred High: `{_fmt(signal.predicted_high)}`",
            f"Pred Low:  `{_fmt(signal.predicted_low)}`",
            f"Range:     `{_fmt(signal.predicted_range)}`",
        ]

    if hasattr(signal, "ic_short_call") and signal.ic_short_call:
        lines += [
            f"Short Call: `{_fmt(signal.ic_short_call)}`  Long Call: `{_fmt(signal.ic_long_call)}`",
            f"Short Put:  `{_fmt(signal.ic_short_put)}`   Long Put:  `{_fmt(signal.ic_long_put)}`",
        ]

    dir_str = getattr(signal, "direction", None)
    if dir_str:
        lines.append(f"Direction: *{dir_str}* ({_fmt(getattr(signal,'direction_prob',None))})")

    lines.append(f"Quality: `{getattr(signal, 'data_quality', '—')}`")

    url  = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": "\n".join(lines), "parse_mode": "Markdown"}

    try:
        resp = requests.post(url, json=data, timeout=timeout)
        resp.raise_for_status()
        logger.info("Telegram message sent (status %s)", resp.status_code)
        return True
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def send_email_alert(signal, smtp_config: AlertConfig) -> bool:
    if not (smtp_config.smtp_host and smtp_config.smtp_user and smtp_config.alert_email_to):
        logger.debug("Email not configured — skipping")
        return False

    regime = getattr(signal, "regime", "YELLOW")
    emoji  = _REGIME_EMOJI.get(regime, "⚪")
    colors = {"GREEN": "#2ecc71", "YELLOW": "#f1c40f", "RED": "#e74c3c"}
    bg     = colors.get(regime, "#95a5a6")

    def row(label, value):
        return f"<tr><td style='padding:4px 8px;font-weight:bold'>{label}</td><td style='padding:4px 8px'>{value}</td></tr>"

    html_rows = [
        row("Signal Date",  signal.signal_date),
        row("Regime",       f"<span style='color:{bg}'><b>{emoji} {regime}</b></span>"),
        row("Tradeable",    "✅ Yes" if signal.tradeable else "❌ No"),
        row("Data Quality", getattr(signal, "data_quality", "—")),
        row("Prior Close",  _fmt(getattr(signal, "prior_close", None))),
    ]

    if hasattr(signal, "predicted_high") and signal.predicted_high:
        html_rows += [
            row("Predicted High",  _fmt(signal.predicted_high)),
            row("Predicted Low",   _fmt(signal.predicted_low)),
            row("Predicted Range", _fmt(signal.predicted_range)),
        ]

    if hasattr(signal, "ic_short_call") and signal.ic_short_call:
        html_rows += [
            row("Short Call / Long Call",
                f"{_fmt(signal.ic_short_call)} / {_fmt(signal.ic_long_call)}"),
            row("Short Put / Long Put",
                f"{_fmt(signal.ic_short_put)} / {_fmt(signal.ic_long_put)}"),
        ]

    html = f"""
    <html><body>
    <h2 style='color:{bg}'>📊 SPX Daily Signal — {signal.signal_date}</h2>
    <table border='0' cellspacing='0' cellpadding='0' style='font-family:monospace;font-size:14px'>
    {''.join(html_rows)}
    </table>
    <hr/>
    <p style='color:#888;font-size:12px'>Generated at: {getattr(signal,'generated_at','—')}</p>
    </body></html>
    """

    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"SPX Signal {signal.signal_date} — {emoji} {regime}"
    msg["From"]    = smtp_config.alert_email_from or smtp_config.smtp_user
    msg["To"]      = smtp_config.alert_email_to
    msg.attach(MIMEText(html, "html"))

    try:
        json_bytes = signal.to_json().encode()
        att = MIMEApplication(json_bytes, Name=f"signal_{signal.signal_date}.json")
        att["Content-Disposition"] = f'attachment; filename="signal_{signal.signal_date}.json"'
        msg.attach(att)
    except Exception:
        pass

    try:
        with smtplib.SMTP(smtp_config.smtp_host, smtp_config.smtp_port,
                          timeout=smtp_config.timeout_seconds) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_config.smtp_user, smtp_config.smtp_pass)
            server.sendmail(msg["From"], msg["To"], msg.as_string())
        logger.info("Email alert sent to %s", smtp_config.alert_email_to)
        return True
    except Exception as exc:
        logger.warning("Email send failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Failure alert (module-level function)
# ---------------------------------------------------------------------------

def send_failure_alert(error_message: str, config: AlertConfig) -> None:
    """Broadcast a failure alert to all configured channels."""
    logger.error("FAILURE ALERT: %s", error_message)
    hostname = socket.gethostname()
    text = f"🚨 *SPX Algo Pipeline FAILURE* on `{hostname}`\n\n{error_message}"

    if config.discord_webhook_url:
        payload = {
            "embeds": [{
                "title":       "🚨 Pipeline Failure",
                "description": error_message,
                "color":       0xE74C3C,
                "footer":      {"text": f"host={hostname}"},
            }]
        }
        try:
            requests.post(config.discord_webhook_url, json=payload,
                          timeout=config.timeout_seconds)
        except Exception as exc:
            logger.warning("Discord failure alert failed: %s", exc)

    if config.telegram_bot_token and config.telegram_chat_id:
        url  = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
        data = {"chat_id": config.telegram_chat_id, "text": text, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=data, timeout=config.timeout_seconds)
        except Exception as exc:
            logger.warning("Telegram failure alert failed: %s", exc)

    if config.smtp_host and config.alert_email_to:
        try:
            msg = MIMEText(error_message)
            msg["Subject"] = f"🚨 SPX Pipeline Failure — {hostname}"
            msg["From"]    = config.alert_email_from or config.smtp_user
            msg["To"]      = config.alert_email_to
            with smtplib.SMTP(config.smtp_host, config.smtp_port,
                              timeout=config.timeout_seconds) as s:
                s.ehlo(); s.starttls()
                s.login(config.smtp_user, config.smtp_pass)
                s.sendmail(msg["From"], msg["To"], msg.as_string())
        except Exception as exc:
            logger.warning("Email failure alert failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level plain-text sender (used by reconciler for summaries)
# ---------------------------------------------------------------------------

def send_text_alert(text: str, config: AlertConfig) -> None:
    """
    Broadcast a plain informational text message to all configured channels.
    Unlike send_failure_alert(), this does NOT add the 🚨 prefix or ERROR log.
    """
    logger.info("Sending text alert: %s", text[:80])
    hostname = socket.gethostname()

    if config.discord_webhook_url:
        payload = {
            "embeds": [{
                "description": text,
                "color":       0x3498DB,   # blue = informational
                "footer":      {"text": f"host={hostname}"},
            }]
        }
        try:
            requests.post(config.discord_webhook_url, json=payload,
                          timeout=config.timeout_seconds)
        except Exception as exc:
            logger.warning("Discord text alert failed: %s", exc)

    if config.telegram_bot_token and config.telegram_chat_id:
        url  = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
        data = {"chat_id": config.telegram_chat_id, "text": text, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=data, timeout=config.timeout_seconds)
        except Exception as exc:
            logger.warning("Telegram text alert failed: %s", exc)

    if config.smtp_host and config.alert_email_to:
        try:
            msg = MIMEText(text)
            msg["Subject"] = "SPX Algo — Daily Update"
            msg["From"]    = config.alert_email_from or config.smtp_user
            msg["To"]      = config.alert_email_to
            with smtplib.SMTP(config.smtp_host, config.smtp_port,
                              timeout=config.timeout_seconds) as s:
                s.ehlo(); s.starttls()
                s.login(config.smtp_user, config.smtp_pass)
                s.sendmail(msg["From"], msg["To"], msg.as_string())
        except Exception as exc:
            logger.warning("Email text alert failed: %s", exc)


# ---------------------------------------------------------------------------
# High-level Alerter class
# ---------------------------------------------------------------------------

class Alerter:
    """Convenience wrapper that broadcasts to all configured channels."""

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or alert_config_from_env()

    def send_signal(self, signal) -> Dict[str, bool]:
        """Send signal to all available channels. Returns channel → success map."""
        results: Dict[str, bool] = {}

        if self.config.discord_webhook_url:
            results["discord"] = send_discord_webhook(
                signal, self.config.discord_webhook_url,
                timeout=self.config.timeout_seconds,
            )

        if self.config.telegram_bot_token and self.config.telegram_chat_id:
            results["telegram"] = send_telegram_message(
                signal,
                self.config.telegram_bot_token,
                self.config.telegram_chat_id,
                timeout=self.config.timeout_seconds,
            )

        if self.config.smtp_host and self.config.alert_email_to:
            results["email"] = send_email_alert(signal, self.config)

        if not results:
            logger.info("No alert channels configured — signal logged only")

        return results

    def send_failure(self, message: str) -> None:
        """Broadcast a failure/error message to all channels."""
        send_failure_alert(message, self.config)

    def send_text(self, text: str) -> None:
        """
        FIX Bug H4: Added send_text() method.
        Broadcast a plain informational text message (not a failure/error)
        to all configured channels.  Called by reconciler.py for daily
        summaries and weekly digests.
        """
        send_text_alert(text, self.config)
