# SPX Algo – Project Handoff

## Project objective
Upgrade `spx_algo` from a strategy-specific signal system into a reusable **next-day SPX OHLC hybrid forecast engine** that can support multiple downstream uses such as:
- iron condors
- ES/MES planning
- discretionary market prep
- dashboards / reporting

---

## Current status
Core deployment path is **complete**.

The repo now has:
- OHLC component target engineering
- event features
- options summary features
- ES daily features
- parallel OHLC model training
- benchmark vs naive / rolling baselines
- hybrid forecast selection
- production hybrid forecast JSON
- daily workflow integration
- reconciliation framework
- scorecard framework

---

## Best current architecture
Current best component source selection:

- `target_gap_ret` → `rolling_baseline`
- `target_high_from_open` → `model`
- `target_low_from_open` → `model`
- `target_close_from_open` → `model`

Reason:
- pure ML underperformed simple baselines overall
- hybrid component selection improved absolute OHLC accuracy
- gap/open remains weakest modeled component

---

## Latest production artifact
Primary forecast artifact:

- `output/forecasts/latest_hybrid_ohlc_forecast.json`

Archive directory:

- `output/forecasts/archive/`

This artifact is now integrated into the daily workflow.

---

## Major completed phases

### Phase 1
Added OHLC component target builder:
- `src/targets/ohlc_targets.py`

### Phase 2
Added explicit event features:
- `src/features/events.py`

### Phase 3
Added summarized options-implied daily features:
- options summary ingestion in feature builder

### Phase 4
Added OHLC model path:
- `src/models/ohlc_forecaster.py`
- `scripts/train_ohlc_models.py`

### Phase 4.5
Added latest OHLC forecast JSON generation:
- `src/pipeline/forecast_generator.py`

### Phase 5 / 5.5
Added benchmark and rolling-baseline comparison:
- `src/evaluation/ohlc_benchmark.py`
- `scripts/evaluate_ohlc_vs_baseline.py`

### Phase 6A
Added hybrid forecast selection:
- `src/evaluation/ohlc_hybrid.py`
- `scripts/evaluate_ohlc_hybrid.py`

### Phase 6B
Added ES daily features:
- `src/features/es_features.py`
- `scripts/fetch_es_daily.py`

### Phase 6C / 6D
Built ES overnight feature pipeline and recent-window experiment:
- technically working
- not suitable for full historical training due to limited Yahoo intraday coverage

### Phase 7A
Added production hybrid forecast generator:
- `src/pipeline/hybrid_forecast_generator.py`
- `scripts/generate_hybrid_ohlc_forecast.py`

### Phase 7B
Integrated hybrid forecast into daily workflow:
- `scripts/run_hybrid_forecast_step.py`
- `scripts/daily_orchestrator.py`

### Phase 8A
Added reconciliation framework:
- `src/evaluation/forecast_reconciliation.py`
- `scripts/reconcile_hybrid_forecast.py`

### Phase 8B
Added rolling scorecard framework:
- `src/evaluation/forecast_scorecard.py`
- `scripts/build_forecast_scorecard.py`

---

## Current findings
1. **Pure ML OHLC forecasting is not best overall**
   - naive / rolling baselines beat pure ML on several components

2. **Hybrid forecasting is the best current approach**
   - baseline for gap
   - ML for high / low / close components

3. **ES daily features help direction more than point accuracy**
   - close direction improved
   - point forecast generally worsened

4. **ES overnight features are not yet production-usable historically**
   - Yahoo intraday coverage is too short
   - recent-window experiment was too small to conclude much

---
