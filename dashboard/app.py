import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="SPX Iron-Condor Algo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SIGNALS_DIR = ROOT / "output" / "signals"
TRADES_FILE = ROOT / "output" / "trades" / "paper_trade_log.csv"
REPORTS_DIR = ROOT / "output" / "reports"


@st.cache_data(ttl=300)
def load_spx():
    df = pd.read_parquet(RAW_DIR / "spx_daily.parquet")
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=300)
def load_vix():
    df = pd.read_parquet(RAW_DIR / "vix_daily.parquet")
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=300)
def load_latest_signal():
    path = SIGNALS_DIR / "latest_signal.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=300)
def load_all_signals():
    signals = []
    for p in sorted(SIGNALS_DIR.glob("signal_*.json")):
        with open(p) as f:
            signals.append(json.load(f))
    return signals


@st.cache_data(ttl=300)
def load_paper_log():
    if TRADES_FILE.exists():
        df = pd.read_csv(TRADES_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_replay_results():
    """Load replay CSV and normalize column names."""
    for name in ["replay_jan_feb_2026_v2.csv", "replay_jan_feb_2026.csv"]:
        path = REPORTS_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"])
            # Normalize column names to what the dashboard expects
            rename_map = {
                "h_err": "h_err_pct",
                "l_err": "l_err_pct",
                "dir_ok": "dir_correct",
                "net_pnl": "net_pnl_dollars",
            }
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
            return df
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_backtest_results():
    for name in ["backtest_10pt_wings_v2.csv", "backtest_10pt_wings.csv"]:
        path = REPORTS_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
    return pd.DataFrame()


def load_signal_for_date(date_str):
    """Load signal JSON for a specific date."""
    path = SIGNALS_DIR / f"signal_{date_str}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("📊 SPX Algo Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📈 Backtest", "🎯 Replay (Jan-Feb 2026)", "📋 Paper Trade Log", "🔮 Latest Signal"],
)


# ══════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("SPX Iron-Condor Algo — Dashboard")

    spx = load_spx()
    vix = load_vix()
    sig = load_latest_signal()
    replay = load_replay_results()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("SPX Close", f"{spx['Close'].iloc[-1]:,.2f}",
                f"{spx['Close'].iloc[-1] - spx['Close'].iloc[-2]:+.2f}")
    col2.metric("VIX", f"{vix['Close'].iloc[-1]:.2f}")

    if sig:
        col3.metric("Regime", sig.get("regime", "N/A"))
        col4.metric("Direction", sig.get("direction", "N/A"),
                     f"{sig.get('direction_prob', 0)*100:.1f}%")
        col5.metric("Tradeable", "✅ Yes" if sig.get("tradeable") else "❌ No")

    if not replay.empty and "net_pnl_dollars" in replay.columns:
        st.markdown("---")
        st.subheader("Jan-Feb 2026 Replay Summary")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        wins = len(replay[replay["condor"] == "WIN"])
        total = len(replay)
        m1.metric("Win Rate", f"{wins/total*100:.1f}%")
        m2.metric("Total P&L", f"${replay['net_pnl_dollars'].sum():,.0f}")
        win_df = replay[replay["condor"] == "WIN"]
        m3.metric("Avg Win", f"${win_df['net_pnl_dollars'].mean():,.0f}" if len(win_df) > 0 else "N/A")
        equity = replay["net_pnl_dollars"].cumsum()
        m4.metric("Max DD", f"${(equity - equity.cummax()).min():,.0f}")
        if "h_err_pct" in replay.columns:
            m5.metric("MAE High", f"{replay['h_err_pct'].mean():.3f}%")
            m6.metric("MAE Low", f"{replay['l_err_pct'].mean():.3f}%")

    st.markdown("---")
    st.subheader("SPX Last 60 Trading Days")
    recent = spx.tail(60)
    fig = go.Figure(data=[go.Candlestick(
        x=recent.index, open=recent["Open"], high=recent["High"],
        low=recent["Low"], close=recent["Close"], name="SPX"
    )])
    fig.update_layout(height=450, xaxis_rangeslider_visible=False,
                      margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
elif page == "📈 Backtest":
    st.title("Backtest Dashboard — Predictions vs Actuals")

    replay = load_replay_results()
    spx = load_spx()

    if not replay.empty:
        st.subheader("Prediction vs Actual (Jan 2 – Feb 26, 2026)")

        # We need to enrich replay data with predictions from signal files
        # or regenerate. For now, load signals and merge.
        all_sigs = load_all_signals()
        sig_df = pd.DataFrame(all_sigs) if all_sigs else pd.DataFrame()

        # Build enriched dataframe by merging replay with SPX actuals
        dates = replay["date"]
        spx_aligned = spx.reindex(dates)

        # Add actual OHLC from SPX if not in replay
        if "actual_high" not in replay.columns:
            replay["actual_high"] = spx_aligned["High"].values
            replay["actual_low"] = spx_aligned["Low"].values
            replay["actual_close"] = spx_aligned["Close"].values

        # Add prior close
        if "prior_close" not in replay.columns:
            prior_closes = []
            for d in dates:
                loc = spx.index.get_loc(d)
                prior_closes.append(float(spx["Close"].iloc[loc - 1]) if loc > 0 else np.nan)
            replay["prior_close"] = prior_closes

        # Compute predicted high/low from errors if not present
        if "pred_high" not in replay.columns and "h_err_pct" in replay.columns:
            # pred = actual ± error (approximate reconstruction)
            # This is directionally approximate; actual values stored in signals
            replay["pred_high"] = replay["actual_high"] + (replay["h_err_pct"] / 100 * replay["prior_close"])
            replay["pred_low"] = replay["actual_low"] + (replay["l_err_pct"] / 100 * replay["prior_close"])

        # ── Main Chart ───────────────────────────────────────────────
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=(
                "SPX: Predicted vs Actual High/Low",
                "Prediction Error (%)",
                "Equity Curve ($)",
            ),
            row_heights=[0.45, 0.25, 0.30],
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=dates, open=replay["prior_close"], high=replay["actual_high"],
            low=replay["actual_low"], close=replay["actual_close"],
            name="Actual OHLC",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ), row=1, col=1)

        # Predicted high/low
        if "pred_high" in replay.columns:
            fig.add_trace(go.Scatter(
                x=dates, y=replay["pred_high"], mode="lines+markers",
                name="Predicted High", line=dict(color="#2196F3", width=2, dash="dot"),
                marker=dict(size=5),
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=dates, y=replay["pred_low"], mode="lines+markers",
                name="Predicted Low", line=dict(color="#FF9800", width=2, dash="dot"),
                marker=dict(size=5),
            ), row=1, col=1)

        # Error bars
        if "h_err_pct" in replay.columns:
            colors_h = ["#ef5350" if e > 0.5 else "#26a69a" for e in replay["h_err_pct"]]
            colors_l = ["#ef5350" if e > 0.5 else "#FF9800" for e in replay["l_err_pct"]]

            fig.add_trace(go.Bar(
                x=dates, y=replay["h_err_pct"], name="High Error %",
                marker_color=colors_h, opacity=0.7,
            ), row=2, col=1)

            fig.add_trace(go.Bar(
                x=dates, y=-replay["l_err_pct"], name="Low Error %",
                marker_color=colors_l, opacity=0.7,
            ), row=2, col=1)

            fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
            fig.add_hline(y=-0.5, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)

        # Equity curve
        if "net_pnl_dollars" in replay.columns:
            equity = replay["net_pnl_dollars"].cumsum()

            fig.add_trace(go.Scatter(
                x=dates, y=equity, mode="lines", name="Cumulative P&L",
                line=dict(color="#2196F3", width=2),
                fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
            ), row=3, col=1)

            loss_mask = replay["condor"] == "LOSS"
            if loss_mask.any():
                fig.add_trace(go.Scatter(
                    x=dates[loss_mask], y=equity[loss_mask],
                    mode="markers", name="Loss Days",
                    marker=dict(color="red", size=10, symbol="x"),
                ), row=3, col=1)

        fig.update_layout(
            height=900, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=20, t=60, b=30),
            xaxis_rangeslider_visible=False,
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Error %", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative $", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # ── Day-by-Day Table ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("Day-by-Day Comparison")

        table_cols = ["date"]
        display_names = {"date": "Date"}

        for col, name in [
            ("prior_close", "Prior Close"), ("pred_high", "Pred High"),
            ("actual_high", "Actual High"), ("h_err_pct", "High Err %"),
            ("pred_low", "Pred Low"), ("actual_low", "Actual Low"),
            ("l_err_pct", "Low Err %"), ("actual_close", "Actual Close"),
            ("dir_correct", "Dir Correct"), ("regime", "Regime"),
            ("condor", "Condor"), ("net_pnl_dollars", "P&L ($)"),
        ]:
            if col in replay.columns:
                table_cols.append(col)
                display_names[col] = name

        display_df = replay[table_cols].copy()
        display_df.rename(columns=display_names, inplace=True)

        # Format numeric columns
        for col in ["Prior Close", "Pred High", "Actual High", "Pred Low", "Actual Low", "Actual Close"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
        if "High Err %" in display_df.columns:
            display_df["High Err %"] = display_df["High Err %"].map(lambda x: f"{x:.3f}%")
        if "Low Err %" in display_df.columns:
            display_df["Low Err %"] = display_df["Low Err %"].map(lambda x: f"{x:.3f}%")
        if "P&L ($)" in display_df.columns:
            display_df["P&L ($)"] = display_df["P&L ($)"].map(lambda x: f"${x:+,.0f}")
        if "Dir Correct" in display_df.columns:
            display_df["Dir Correct"] = display_df["Dir Correct"].map(lambda x: "✓" if x else "✗")
        if "Date" in display_df.columns:
            display_df["Date"] = pd.to_datetime(display_df["Date"]).dt.strftime("%Y-%m-%d")

        st.dataframe(display_df, use_container_width=True, height=600)

        # ── Coverage Heatmap ─────────────────────────────────────────
        cov_cols = ["cov68h", "cov68l", "cov90h", "cov90l"]
        if all(c in replay.columns for c in cov_cols):
            st.markdown("---")
            st.subheader("Conformal Coverage Heatmap")

            cov_data = replay[["date"] + cov_cols].copy()
            cov_data["date"] = cov_data["date"].dt.strftime("%Y-%m-%d")

            cov_matrix = cov_data[cov_cols].astype(int).T
            cov_matrix.columns = cov_data["date"]
            cov_matrix.index = ["68% High", "68% Low", "90% High", "90% Low"]

            fig_hm = go.Figure(data=go.Heatmap(
                z=cov_matrix.values,
                x=cov_matrix.columns,
                y=cov_matrix.index,
                colorscale=[[0, "#ef5350"], [1, "#26a69a"]],
                showscale=False,
            ))
            fig_hm.update_layout(height=200, margin=dict(l=100, r=20, t=30, b=30))
            st.plotly_chart(fig_hm, use_container_width=True)

        # ── Statistics ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Performance Statistics")

        sc1, sc2, sc3, sc4 = st.columns(4)

        with sc1:
            st.markdown("**Prediction Accuracy**")
            if "h_err_pct" in replay.columns:
                st.write(f"Mean High Error: {replay['h_err_pct'].mean():.4f}%")
                st.write(f"Mean Low Error: {replay['l_err_pct'].mean():.4f}%")
                st.write(f"Combined MAE: {(replay['h_err_pct'].mean()+replay['l_err_pct'].mean())/2:.4f}%")

        with sc2:
            st.markdown("**Direction**")
            if "dir_correct" in replay.columns:
                da = replay["dir_correct"].mean() * 100
                st.write(f"Accuracy: {replay['dir_correct'].sum()}/{len(replay)} ({da:.1f}%)")

        with sc3:
            st.markdown("**Iron Condor**")
            wins = len(replay[replay["condor"] == "WIN"])
            losses_df = replay[replay["condor"] == "LOSS"]
            st.write(f"Win Rate: {wins}/{len(replay)} ({wins/len(replay)*100:.1f}%)")
            if "net_pnl_dollars" in replay.columns:
                st.write(f"Total P&L: ${replay['net_pnl_dollars'].sum():,.2f}")
                win_pnl = replay[replay['condor']=='WIN']['net_pnl_dollars']
                st.write(f"Avg Win: ${win_pnl.mean():,.2f}" if len(win_pnl) > 0 else "Avg Win: N/A")
                if len(losses_df) > 0:
                    st.write(f"Avg Loss: ${losses_df['net_pnl_dollars'].mean():,.2f}")
                equity = replay["net_pnl_dollars"].cumsum()
                st.write(f"Max DD: ${(equity - equity.cummax()).min():,.2f}")

        with sc4:
            st.markdown("**Coverage**")
            for col, label in [("cov68h", "68% High"), ("cov68l", "68% Low"),
                               ("cov90h", "90% High"), ("cov90l", "90% Low")]:
                if col in replay.columns:
                    st.write(f"{label}: {replay[col].mean()*100:.1f}%")

    else:
        st.warning("No replay data found. Run the replay backtest first.")


# ══════════════════════════════════════════════════════════════════════
elif page == "🎯 Replay (Jan-Feb 2026)":
    st.title("Replay Detail — Jan-Feb 2026")

    replay = load_replay_results()
    spx = load_spx()

    if not replay.empty:
        selected_date = st.selectbox(
            "Select trading day",
            replay["date"].dt.strftime("%Y-%m-%d").tolist(),
            index=len(replay) - 1,
        )

        row = replay[replay["date"] == pd.Timestamp(selected_date)].iloc[0]

        # Get actual OHLC from SPX
        td = pd.Timestamp(selected_date)
        spx_row = spx.loc[td] if td in spx.index else None

        # Day summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Regime", row.get("regime", "N/A"))

        dir_correct = row.get("dir_correct", None)
        c2.metric("Direction", row.get("direction_pred", "N/A") if "direction_pred" in row.index else "N/A",
                   "✓ Correct" if dir_correct else "✗ Wrong" if dir_correct is not None else "")

        condor = row.get("condor", "N/A")
        pnl = row.get("net_pnl_dollars", 0)
        c3.metric("Condor Result", condor, f"${pnl:+,.0f}" if pd.notna(pnl) else "")

        vix_val = row.get("vix", None)
        c4.metric("VIX", f"{vix_val:.2f}" if pd.notna(vix_val) and vix_val else "N/A")

        st.markdown("---")

        # Errors
        e1, e2, e3 = st.columns(3)
        if "h_err_pct" in row.index:
            e1.metric("High Error", f"{row['h_err_pct']:.3f}%")
        if "l_err_pct" in row.index:
            e2.metric("Low Error", f"{row['l_err_pct']:.3f}%")
        if "h_err_pct" in row.index and "l_err_pct" in row.index:
            e3.metric("Combined MAE", f"{(row['h_err_pct'] + row['l_err_pct'])/2:.3f}%")

        # Coverage flags
        st.markdown("---")
        st.subheader("Conformal Coverage")
        cv1, cv2, cv3, cv4 = st.columns(4)
        for col, label, container in [
            ("cov68h", "68% High", cv1), ("cov68l", "68% Low", cv2),
            ("cov90h", "90% High", cv3), ("cov90l", "90% Low", cv4),
        ]:
            if col in row.index:
                container.metric(label, "✅ Covered" if row[col] else "❌ Missed")

        # Price comparison chart
        if spx_row is not None:
            st.markdown("---")
            st.subheader("Price Levels")

            prior_loc = spx.index.get_loc(td)
            prior_close = float(spx["Close"].iloc[prior_loc - 1]) if prior_loc > 0 else np.nan
            actual_high = float(spx_row["High"])
            actual_low = float(spx_row["Low"])
            actual_close = float(spx_row["Close"])

            # Reconstruct predicted from errors (approximate)
            if "h_err_pct" in row.index and pd.notna(prior_close):
                pred_high = actual_high + (row["h_err_pct"] / 100 * prior_close)
                pred_low = actual_low + (row["l_err_pct"] / 100 * prior_close)
            else:
                pred_high = pred_low = np.nan

            fig_ic = go.Figure()

            # Actual range
            fig_ic.add_trace(go.Bar(
                x=["Day"], y=[actual_high - actual_low],
                base=[actual_low], name="Actual Range",
                marker_color="rgba(38,166,154,0.3)", width=0.4,
            ))

            fig_ic.add_hline(y=prior_close, line_color="gray", line_width=2,
                             annotation_text=f"Prior Close: {prior_close:,.2f}")
            fig_ic.add_hline(y=actual_high, line_color="#26a69a", line_dash="solid",
                             annotation_text=f"Actual High: {actual_high:,.2f}")
            fig_ic.add_hline(y=actual_low, line_color="#ef5350", line_dash="solid",
                             annotation_text=f"Actual Low: {actual_low:,.2f}")

            if pd.notna(pred_high):
                fig_ic.add_hline(y=pred_high, line_color="#2196F3", line_dash="dot",
                                 annotation_text=f"Pred High: {pred_high:,.2f}")
                fig_ic.add_hline(y=pred_low, line_color="#FF9800", line_dash="dot",
                                 annotation_text=f"Pred Low: {pred_low:,.2f}")

            fig_ic.update_layout(height=450, yaxis_title="SPX Price",
                                 margin=dict(l=50, r=180, t=30, b=30),
                                 showlegend=False)
            st.plotly_chart(fig_ic, use_container_width=True)

        # Raw data
        with st.expander("Raw row data"):
            st.json({k: (v if not isinstance(v, (np.integer, np.floating)) else float(v))
                     for k, v in row.to_dict().items()})

    else:
        st.warning("No replay data found.")


# ══════════════════════════════════════════════════════════════════════
elif page == "📋 Paper Trade Log":
    st.title("Paper Trade Log")

    paper = load_paper_log()

    if not paper.empty:
        st.subheader(f"{len(paper)} entries")

        completed = paper[paper["actual_close"].notna()] if "actual_close" in paper.columns else pd.DataFrame()
        pending = paper[paper["actual_close"].isna()] if "actual_close" in paper.columns else paper

        if not pending.empty:
            st.info(f"📌 {len(pending)} pending signal(s) awaiting market data")
            display_cols = [c for c in ["date", "predicted_high", "predicted_low",
                                         "predicted_direction", "regime", "prior_close",
                                         "call_strike", "put_strike"] if c in pending.columns]
            st.dataframe(pending[display_cols], use_container_width=True)

        if not completed.empty:
            st.markdown("---")
            st.subheader("Completed Trades")

            if "condor_pnl" in completed.columns:
                pnl_series = pd.to_numeric(completed["condor_pnl"], errors="coerce")
                equity = pnl_series.cumsum()
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=completed["date"], y=equity, mode="lines+markers",
                    fill="tozeroy", name="Equity",
                ))
                fig_eq.update_layout(height=300, title="Paper Trade Equity Curve")
                st.plotly_chart(fig_eq, use_container_width=True)

            st.dataframe(completed, use_container_width=True, height=400)
    else:
        st.info("No paper trades yet. The first signal was logged for March 2, 2026.")


# ══════════════════════════════════════════════════════════════════════
elif page == "🔮 Latest Signal":
    st.title("Latest Signal")

    sig = load_latest_signal()

    if sig:
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Signal Date", sig.get("signal_date", "N/A"))
        h2.metric("Regime", sig.get("regime", "N/A"))
        h3.metric("Direction", sig.get("direction", "N/A"),
                   f"{sig.get('direction_prob', 0)*100:.1f}%")
        h4.metric("VIX", f"{sig.get('vix_spot', 0):.2f}")
        h5.metric("Tradeable", "✅" if sig.get("tradeable") else "❌")

        st.markdown("---")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.markdown("**Price Predictions**")
            st.write(f"Prior Close: {sig.get('prior_close', 0):,.2f}")
            st.write(f"Predicted High: {sig.get('predicted_high', 0):,.2f} ({sig.get('pred_high_pct', 0)*100:+.4f}%)")
            st.write(f"Predicted Low: {sig.get('predicted_low', 0):,.2f} ({sig.get('pred_low_pct', 0)*100:+.4f}%)")
            st.write(f"Predicted Range: {sig.get('predicted_range', 0):,.2f} pts")

        with p2:
            st.markdown("**Iron Condor Strikes**")
            st.write(f"Short Call: {sig.get('ic_short_call', 0):,.2f}")
            st.write(f"Long Call: {sig.get('ic_long_call', 0):,.2f}")
            st.write(f"Short Put: {sig.get('ic_short_put', 0):,.2f}")
            st.write(f"Long Put: {sig.get('ic_long_put', 0):,.2f}")
            wing = sig.get("ic_long_call", 0) - sig.get("ic_short_call", 0)
            st.write(f"Wing Width: {wing:.0f} pts (${wing*100:,.0f} risk)")

        with p3:
            st.markdown("**Conformal Intervals**")
            st.write(f"68% High: [{sig.get('conf_68_high_lo', 0):,.2f}, {sig.get('conf_68_high_hi', 0):,.2f}]")
            st.write(f"68% Low: [{sig.get('conf_68_low_lo', 0):,.2f}, {sig.get('conf_68_low_hi', 0):,.2f}]")
            st.write(f"90% High: [{sig.get('conf_90_high_lo', 0):,.2f}, {sig.get('conf_90_high_hi', 0):,.2f}]")
            st.write(f"90% Low: [{sig.get('conf_90_low_lo', 0):,.2f}, {sig.get('conf_90_low_hi', 0):,.2f}]")

        # Strike map
        st.markdown("---")
        st.subheader("Strike Map")

        pc = sig.get("prior_close", 0)
        fig = go.Figure()

        fig.add_hrect(y0=sig.get("conf_90_low_lo", 0), y1=sig.get("conf_90_high_hi", 0),
                      fillcolor="rgba(33,150,243,0.05)", line_width=0)
        fig.add_hrect(y0=sig.get("conf_68_low_lo", 0), y1=sig.get("conf_68_high_hi", 0),
                      fillcolor="rgba(33,150,243,0.1)", line_width=0)

        fig.add_hline(y=pc, line_color="gray", line_width=2,
                      annotation_text=f"Prior Close: {pc:,.2f}")
        fig.add_hline(y=sig.get("predicted_high", 0), line_color="#2196F3", line_dash="dot",
                      annotation_text=f"Pred High: {sig.get('predicted_high', 0):,.2f}")
        fig.add_hline(y=sig.get("predicted_low", 0), line_color="#FF9800", line_dash="dot",
                      annotation_text=f"Pred Low: {sig.get('predicted_low', 0):,.2f}")
        fig.add_hline(y=sig.get("ic_short_call", 0), line_color="#e91e63", line_dash="dash",
                      annotation_text=f"Short Call: {sig.get('ic_short_call', 0):,.2f}")
        fig.add_hline(y=sig.get("ic_short_put", 0), line_color="#9c27b0", line_dash="dash",
                      annotation_text=f"Short Put: {sig.get('ic_short_put', 0):,.2f}")

        fig.update_layout(height=500, yaxis_title="SPX Price",
                          margin=dict(l=50, r=200, t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Signal metadata"):
            st.write(f"Generated at: {sig.get('generated_at', 'N/A')}")
            st.write(f"Model versions: {sig.get('model_versions', {})}")
            st.write(f"Features used: {sig.get('n_features_used', 0)}")
            st.write(f"Data quality: {sig.get('data_quality', 'N/A')}")
            if sig.get("notes"):
                for note in sig["notes"]:
                    st.write(f"  • {note}")

        with st.expander("Raw JSON"):
            st.json(sig)
    else:
        st.warning("No signal file found.")


# Footer
st.sidebar.markdown("---")
st.sidebar.caption("SPX Algo Dashboard v1.0")
st.sidebar.caption(f"Data through: {load_spx().index[-1].strftime('%Y-%m-%d')}")
