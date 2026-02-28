# Troubleshooting Guide
## SPX Iron-Condor Algo — Common Issues & Fixes

---

### 1. yFinance Rate Limit

**Symptom:** `Exception: Too Many Requests 429` or `YFRateLimitError`.

**Cause:** yFinance is throttled by Yahoo Finance (~2,000 requests/day per IP).

**Fix:**
```python
# In src/data/live_fetcher.py, RETRY_DELAYS are already (5, 15, 45) seconds.
# If persistent, add a longer delay:
import time
time.sleep(60)
```
Or use the FRED/Parquet fallback: `python -m src.data.fetcher --source fred`.

**Prevention:** Do not call `yf.download()` in a tight loop. The live fetcher already uses exponential back-off.

---

### 2. IBKR "Ambiguous Contract" Error

**Symptom:** `Contract is ambiguous` or `No security found for SPX`.

**Cause:** SPX options require explicit exchange and multiplier specification.

**Fix:** In `src/execution/broker.py`, ensure the contract is built as:
```python
from ib_async import Option, Contract
contract = Option(
    symbol      = "SPX",
    lastTradeDateOrContractMonth = expiry_yyyymmdd,
    strike      = strike,
    right       = "C",   # or "P"
    exchange    = "SMART",
    currency    = "USD",
    multiplier  = "100",
)
```

**Also check:** TWS → Contract Search → verify SPX is listed under CBOE.

---

### 3. FRED API Key Expired / Invalid

**Symptom:** `fredapi.api.ValueError: Bad Request. The key is not valid` or HTTP 400.

**Fix:**
1. Go to [research.stlouisfed.org/useraccount/apikey](https://research.stlouisfed.org/useraccount/apikey).
2. Generate a new API key (free, instant).
3. Update `.env`: `FRED_API_KEY=your_new_key`.
4. Restart the scheduler.

---

### 4. HMM Convergence Warning

**Symptom:**
```
ConvergenceWarning: Model is not converging. ...
```
or `LinAlgError: 2-th leading minor ... not positive definite`.

**Cause:** HMM with `covariance_type='full'` requires sufficient data and non-degenerate features.

**Fix:** Already handled in `src/calibration/regime.py` with covariance-type fallback (full → diag → spherical). If still failing:
```python
# Increase n_iter in RegimeDetector
detector = RegimeDetector(hmm_n_iter=500)
```

**Also try:** Normalize observation features before fitting:
```python
from sklearn.preprocessing import StandardScaler
obs = StandardScaler().fit_transform(obs)
```

---

### 5. Docker Timezone Wrong

**Symptom:** Signal generated at wrong time; dead-man's switch fires incorrectly.

**Verify:**
```bash
docker exec spx_scheduler python -c "from datetime import datetime; import pytz; print(datetime.now(pytz.timezone('America/New_York')))"
```

**Fix in Dockerfile** (already included, but verify):
```dockerfile
ENV TZ=America/New_York
RUN apt-get install -y tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone
```

Also ensure `docker-compose.yml` passes `TZ`:
```yaml
environment:
  - TZ=America/New_York
```

---

### 6. ConformalPredictor Returns Constant Intervals

**Symptom:** All intervals are the same width; `detect_stub_intervals()` returns `True`.

**Cause:** Model is returning a constant prediction (underfitted), so all residuals are the same.

**Fix:**
1. Verify the model has been fitted on enough data: `MIN_TRAIN_ROWS = 252`.
2. Check features are not all zeros: `X_tr.describe()`.
3. Use `RidgeRegressionModel(alpha=0.01)` for a weaker regularization.

---

### 7. Signal File Not Created

**Symptom:** Dead-man's switch fires; `output/signals/signal_YYYYMMDD.json` missing.

**Diagnostic:**
```bash
python -m src.pipeline.signal_generator --date $(date +%Y-%m-%d) 2>&1 | tail -20
```

**Common causes:**
- SPX data not downloaded: run `python -m src.data.fetcher`.
- Feature build failure: check `data/processed/features.parquet` exists.
- Model not fitted: first run takes longer as it fits from scratch.

---

### 8. pytest Coverage Below 80%

**Symptom:** CI fails with `FAIL Required test coverage of 80% not reached`.

**Fix:**
```bash
python -m pytest tests/ --cov=src --cov-report=term-missing | grep "TOTAL"
```

Identify uncovered modules and add targeted tests, or add `# pragma: no cover` to
infrastructure-only code (e.g., `__main__` blocks, logging setup).

---

### 9. Reconciler: No Signal File for Date

**Symptom:** `log_outcome: no signal row found for 2024-01-15`.

**Cause:** Signal was never generated for that date (weekend, holiday, or system was down).

**Fix:**
```bash
# Regenerate historical signal
python -m src.pipeline.signal_generator --date 2024-01-15
# Then re-run reconciliation
python -c "from src.pipeline.reconciler import Reconciler; Reconciler().reconcile('2024-01-15')"
```

---

### 10. Optuna Import Error

**Symptom:** `ImportError: No module named 'optuna'`.

**Fix:**
```bash
pip install "optuna>=3.5.0"
```

Optuna is required for `src/validation/hyperparam_sweep.py` but is not needed for core signal generation.

---

### 11. Port Already in Use (IBKR Client ID Conflict)

**Symptom:** `ConnectionRefusedError` or `Already connected`.

**Fix:** Each IBKR API connection requires a unique `client_id`. If multiple processes connect simultaneously:
```python
# Use different client IDs per process
broker = IBKRBroker(client_id=2)  # scheduler
broker = IBKRBroker(client_id=3)  # manual override
```

---

### 12. Drift Detector Shows DEGRADED Immediately

**Symptom:** First run of `check_drift()` returns DEGRADED.

**Cause:** `drift_log.csv` has stale data from a prior misconfigured run.

**Fix:**
```bash
rm output/monitoring/drift_log.csv
rm output/monitoring/retrain_requested.flag
```

The detector starts fresh with an empty window and returns HEALTHY.
