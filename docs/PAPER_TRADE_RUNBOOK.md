# Paper Trade Runbook
## SPX Iron-Condor Algo — Phase 6 Operations Guide

---

### Prerequisites

| Requirement | Details |
|-------------|---------|
| IBKR account | Any funded account (paper account is free) |
| TWS or IB Gateway | Download from [ibkr.com/tws](https://www.interactivebrokers.com/en/index.php?f=16042) |
| SPX options data | CBOE SPX Level 1 (~$1.50/month on paper) |
| Python 3.11 | `python --version` |
| Conda / venv | `pip install -r requirements.txt` |

---

### Step 1 — Install TWS or IB Gateway

1. Download **TWS** (full UI) or **IB Gateway** (headless) from IBKR.
2. Log in with your **paper trading** credentials (separate login from live).
3. Navigate to **Global Configuration → API → Settings**:
   - ✅ Enable ActiveX and Socket Clients
   - Socket port: **7497** (paper TWS) or **4002** (paper IB Gateway)
   - ✅ Allow connections from localhost only
   - Uncheck **Read-Only API**

---

### Step 2 — Create / Access Paper Trading Account

1. Log in at [interactivebrokers.com](https://www.interactivebrokers.com).
2. Go to **Account Management → Paper Trading Account**.
3. IBKR provides $1,000,000 virtual capital by default.
4. Enable **Options Trading** on the paper account if not already active.

---

### Step 3 — Subscribe to SPX Options Data (Paper)

1. In **Account Management → Market Data Subscriptions**.
2. Add **CBOE SPX Options** (~$1.50/month).
3. Without this subscription, SPX option quotes will be delayed or unavailable.

---

### Step 4 — Configure Environment

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Fill in:
```
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
FRED_API_KEY=your_key_from_fred.stlouisfed.org
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...   # optional
TELEGRAM_BOT_TOKEN=...                                     # optional
ALERT_EMAIL_TO=you@example.com                             # optional
EXECUTION_MODE=PAPER                                       # NEVER change to LIVE until ready
```

---

### Step 5 — Run the Smoke Test

```bash
bash scripts/smoke_test.sh --skip-docker
```

Expected output:
```
✅  Test suite passed
✅  Conformal coverage gate PASSED
✅  Today's signal generation OK
✅  Drift detector: HEALTHY
✅  Signal JSON valid
✅  SMOKE TEST COMPLETE — All gates passed.
```

If any step fails, **do not proceed** until resolved (see `docs/TROUBLESHOOTING.md`).

---

### Step 6 — Download Historical Data

```bash
python -m src.data.fetcher
```

This downloads SPX, VIX, and FRED macro data to `data/raw/`.
Minimum required: **3 years** of daily data for walk-forward validation.

---

### Step 7 — Start the Scheduler

```bash
python -m src.pipeline.scheduler
```

The scheduler will:
- **10:00 AM ET**: Run daily reconciliation (yesterday's signal vs actuals).
- **4:05 PM ET**: Generate today's signal from end-of-day data.
- **4:30 PM ET**: Dead-man's switch alert if signal file is missing.

Keep this process running in a terminal multiplexer (`screen`, `tmux`) or as a service:

```bash
# systemd example
sudo systemctl start spx-algo-scheduler
```

---

### Step 8 — Monitor Daily

Check your configured alert channel (Discord / Telegram / email) for:

| Time (ET) | Message |
|-----------|---------|
| ~10:05 AM | Reconciliation: yesterday's errors + drift status |
| ~4:10 PM  | Today's signal: regime, predicted high/low, condor strikes |
| 4:30 PM   | Dead-man's switch (only if signal is missing) |
| Friday ~4:15 PM | Weekly digest: 5-day MAE, win rate, P&L, drift history |

---

### Step 9 — Review Performance (After 20 Days)

```bash
python -m src.execution.paper_logger --summary 20
```

**Go/No-Go Criteria for Continuing:**

| Metric | Threshold | Notes |
|--------|-----------|-------|
| MAE (high/low) | < 0.5% | Rolling 20-day |
| Directional accuracy | > 58% | Same-day |
| Condor win rate | > 65% | Active trades only |
| Conformal 68% coverage | 60–76% | Rolling 20-day |
| Drift status | No DEGRADED > 2 days | Consecutive |
| Total P&L | Positive or small loss | Context-dependent |

If any criterion fails: diagnose via `docs/TROUBLESHOOTING.md` before continuing.

---

### Step 10 — Go-Live Decision (After 60 Days)

All criteria above must be met on **rolling 63-day window** before committing real capital.

Additional requirements:
- [ ] Reviewed max drawdown profile and acceptable to you personally.
- [ ] IBKR risk settings configured (stop-loss orders, position alerts).
- [ ] Execution mode change from `PAPER` → `LIVE` reviewed and tested with a **single 1-contract condor**.
- [ ] Emergency shutdown procedure tested (kill scheduler, close all open positions manually).

---

### Emergency Shutdown

```bash
# Stop scheduler immediately
pkill -f "src.pipeline.scheduler"

# Close all open SPX positions manually via TWS
# TWS → Portfolio → SPX positions → Close All
```

---

### Docker Deployment (Optional)

```bash
# Build image
make build

# Run scheduler (production mode)
docker compose up -d scheduler

# View logs
docker compose logs -f scheduler

# One-shot signal generation
docker compose run --rm algo python -m src.pipeline.signal_generator
```
