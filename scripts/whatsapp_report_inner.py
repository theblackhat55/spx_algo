import json, sys
from pathlib import Path

sig_path = Path("output/signals/latest_signal.json")
if not sig_path.exists():
    print("No signal file found")
    sys.exit(1)

with open(sig_path) as f:
    sig = json.load(f)

trade_stats = ""
trades_path = Path("output/trades/paper_trade_log.csv")
if trades_path.exists():
    import pandas as pd
    trades = pd.read_csv(trades_path)
    if len(trades) > 0:
        completed = trades.dropna(subset=["condor_result"]) if "condor_result" in trades.columns else pd.DataFrame()
        if len(completed) > 0:
            wins = int((completed["condor_result"] == "WIN").sum())
            total = len(completed)
            win_rate = wins / total * 100
            total_pnl = float(completed["condor_pnl"].sum()) if "condor_pnl" in completed.columns else 0.0
            avg_pnl = total_pnl / total
            trade_stats = (
                f"\nPaper Trade Stats ({total} trades)\n"
                f"  Win Rate: {win_rate:.1f}% ({wins}/{total})\n"
                f"  Total PnL: ${total_pnl:,.2f}\n"
                f"  Avg PnL: ${avg_pnl:,.2f}/trade"
            )

dp = sig.get("direction_prob", 0)  # FullSignal field is direction_prob, not direction_probability
report = (
    f"SPX Iron-Condor Signal\n"
    f"━━━━━━━━━━━━━━━━━━━━━━\n"
    f"Date: {sig.get('signal_date', 'N/A')}\n"
    f"Prior Close: {sig.get('prior_close', 'N/A')}\n"
    f"\n"
    f"Predictions\n"
    f"  High: {sig.get('predicted_high', 'N/A')}\n"
    f"  Low: {sig.get('predicted_low', 'N/A')}\n"
    f"\n"
    f"Direction: {sig.get('direction', 'N/A')} ({dp:.1%})\n"
    f"Regime: {sig.get('regime', 'N/A')}\n"
    f"VIX: {sig.get('vix_spot', 'N/A')}\n"
    f"Tradeable: {sig.get('tradeable', 'N/A')}\n"
    f"\n"
    f"Iron Condor Strikes\n"
    f"  Short Call: {sig.get('ic_short_call', 'N/A')}\n"
    f"  Short Put:  {sig.get('ic_short_put', 'N/A')}\n"
    f"  Long Call:  {sig.get('ic_long_call', 'N/A')}\n"
    f"  Long Put:   {sig.get('ic_long_put', 'N/A')}\n"
    f"\n"
    f"90% Confidence\n"
    f"  High: {sig.get('conf_90_high_lo', 'N/A')} - {sig.get('conf_90_high_hi', 'N/A')}\n"
    f"  Low:  {sig.get('conf_90_low_lo', 'N/A')} - {sig.get('conf_90_low_hi', 'N/A')}\n"
    f"{trade_stats}\n"
    f"\n"
    f"Generated: {sig.get('generated_at', 'N/A')}"
)
print(report)
