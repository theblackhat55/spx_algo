"""
config/feature_registry.py
===========================
Central registry of every feature in the spx_algo system.

Each entry contains:
  category          : str   — one of: proximity | volatility | technical |
                               calendar | options | lagged_target
  module            : str   — dotted import path of the generating module
  requires_open     : bool  — True if the feature CANNOT be computed before
                              today's open is known (post-open only)
  lookback_window   : int   — minimum rows of history required before the
                              feature value is valid (warmup rows)
  description       : str   — human-readable explanation

Rules enforced by tests:
  • Every feature generated in code must appear here.
  • Every feature listed here must appear in the built feature matrix.
"""

FEATURE_REGISTRY: dict[str, dict] = {

    # ── PROXIMITY FEATURES (Task 4) ──────────────────────────────────────────
    "prev_high_pct_1": {
        "category": "proximity", "module": "src.features.proximity",
        "requires_open": False, "lookback_window": 2,
        "description": "Lag-1 percentage distance from prior close to prior high",
    },
    "prev_low_pct_1": {
        "category": "proximity", "module": "src.features.proximity",
        "requires_open": False, "lookback_window": 2,
        "description": "Lag-1 percentage distance from prior close to prior low",
    },
    "prev_high_pct_2": {
        "category": "proximity", "module": "src.features.proximity",
        "requires_open": False, "lookback_window": 3,
        "description": "Lag-2 percentage distance from prior close to prior high",
    },
    "prev_low_pct_2": {
        "category": "proximity", "module": "src.features.proximity",
        "requires_open": False, "lookback_window": 3,
        "description": "Lag-2 percentage distance from prior close to prior low",
    },
    "prev_high_pct_3": {
        "category": "proximity", "module": "src.features.proximity",
        "requires_open": False, "lookback_window": 4,
        "description": "Lag-3 percentage distance from prior close to prior high",
    },
    "prev_low_pct_3": {
        "category": "proximity", "module": "src.features.proximity",
        "requires_open": False, "lookback_window": 4,
        "description": "Lag-3 percentage distance from prior close to prior low",
    },
    "prev_high_pct_5": {
        "category": "proximity", "module": "src.features.proximity",
        "requires_open": False, "lookback_window": 6,
        "description": "Lag-5 percentage distance from prior close to prior high",
    },
    "prev_low_pct_5": {
        "category": "proximity", "module": "src.features.proximity",
        "requires_open": False, "lookback_window": 6,
        "description": "Lag-5 percentage distance from prior close to prior low",
    },
    "open_gap_pct": {
        "category": "proximity", "module": "src.features.proximity",
        "requires_open": True, "lookback_window": 2,
        "description": "Overnight gap: (Open_t - Close_{t-1}) / Close_{t-1}",
    },
    "prev_range_pct": {
        "category": "proximity", "module": "src.features.proximity",
        "requires_open": False, "lookback_window": 2,
        "description": "Prior day range as % of close: (High-Low) / Close",
    },
    "rolling_avg_range_5": {
        "category": "proximity", "module": "src.features.proximity",
        "requires_open": False, "lookback_window": 6,
        "description": "5-day rolling average of (High-Low) / Close",
    },
    "rolling_avg_range_20": {
        "category": "proximity", "module": "src.features.proximity",
        "requires_open": False, "lookback_window": 21,
        "description": "20-day rolling average of (High-Low) / Close",
    },

    # ── VOLATILITY FEATURES (Task 5) ─────────────────────────────────────────
    "atr_5": {
        "category": "volatility", "module": "src.features.volatility",
        "requires_open": False, "lookback_window": 6,
        "description": "Average True Range over 5 days",
    },
    "atr_10": {
        "category": "volatility", "module": "src.features.volatility",
        "requires_open": False, "lookback_window": 11,
        "description": "Average True Range over 10 days",
    },
    "atr_20": {
        "category": "volatility", "module": "src.features.volatility",
        "requires_open": False, "lookback_window": 21,
        "description": "Average True Range over 20 days",
    },
    "atr_60": {
        "category": "volatility", "module": "src.features.volatility",
        "requires_open": False, "lookback_window": 61,
        "description": "Average True Range over 60 days",
    },
    "atr_ratio_5_60": {
        "category": "volatility", "module": "src.features.volatility",
        "requires_open": False, "lookback_window": 61,
        "description": "ATR_5 / ATR_60 — short-term vs long-term volatility expansion",
    },
    "parkinson_vol_20": {
        "category": "volatility", "module": "src.features.volatility",
        "requires_open": False, "lookback_window": 21,
        "description": "Parkinson range-based volatility estimator (20-day)",
    },
    "garman_klass_vol_20": {
        "category": "volatility", "module": "src.features.volatility",
        "requires_open": False, "lookback_window": 21,
        "description": "Garman-Klass open-high-low-close volatility estimator (20-day)",
    },
    "vix_normalized": {
        "category": "volatility", "module": "src.features.volatility",
        "requires_open": False, "lookback_window": 253,
        "description": "VIX / 252-day rolling mean of VIX",
    },
    "vix_zscore": {
        "category": "volatility", "module": "src.features.volatility",
        "requires_open": False, "lookback_window": 253,
        "description": "VIX Z-score: (VIX - rolling_mean) / rolling_std over 252 days",
    },
    "vix_percentile_252": {
        "category": "volatility", "module": "src.features.volatility",
        "requires_open": False, "lookback_window": 253,
        "description": "VIX percentile rank over the past 252 days",
    },
    "vix_roc_5": {
        "category": "volatility", "module": "src.features.volatility",
        "requires_open": False, "lookback_window": 6,
        "description": "VIX 5-day rate of change",
    },
    "garch_cond_var": {
        "category": "volatility", "module": "src.features.volatility",
        "requires_open": False, "lookback_window": 252,
        "description": "GARCH(1,1) one-step-ahead conditional variance forecast",
    },

    # ── TECHNICAL FEATURES (Task 6) ──────────────────────────────────────────
    "ema_dist_9": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 10,
        "description": "(Close - EMA_9) / EMA_9 — distance from 9-day EMA",
    },
    "ema_dist_21": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 22,
        "description": "(Close - EMA_21) / EMA_21 — distance from 21-day EMA",
    },
    "ema_dist_50": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 51,
        "description": "(Close - EMA_50) / EMA_50 — distance from 50-day EMA",
    },
    "ema_dist_100": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 101,
        "description": "(Close - EMA_100) / EMA_100 — distance from 100-day EMA",
    },
    "ema_dist_200": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 201,
        "description": "(Close - EMA_200) / EMA_200 — distance from 200-day EMA",
    },
    "trend_short": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 22,
        "description": "Binary: EMA_9 > EMA_21",
    },
    "trend_medium": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 51,
        "description": "Binary: EMA_21 > EMA_50",
    },
    "rsi_7": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 8,
        "description": "RSI with 7-day period",
    },
    "rsi_14": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 15,
        "description": "RSI with 14-day period",
    },
    "rsi_28": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 29,
        "description": "RSI with 28-day period",
    },
    "roc_5": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 6,
        "description": "Rate of Change over 5 days",
    },
    "roc_10": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 11,
        "description": "Rate of Change over 10 days",
    },
    "roc_20": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 21,
        "description": "Rate of Change over 20 days",
    },
    "macd_histogram": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 35,
        "description": "MACD histogram (12, 26, 9)",
    },
    "bb_position": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 21,
        "description": "Bollinger Band position: (Close - Lower) / (Upper - Lower)",
    },
    "bb_width": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 21,
        "description": "Bollinger Band width: (Upper - Lower) / Middle",
    },
    "volume_ratio_5": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 6,
        "description": "Volume / 5-day rolling mean volume",
    },
    "volume_ratio_10": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 11,
        "description": "Volume / 10-day rolling mean volume",
    },
    "volume_ratio_20": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 21,
        "description": "Volume / 20-day rolling mean volume",
    },
    "obv_roc_10": {
        "category": "technical", "module": "src.features.technical",
        "requires_open": False, "lookback_window": 11,
        "description": "On-Balance Volume 10-day rate of change",
    },

    # ── CALENDAR FEATURES (Task 7) ────────────────────────────────────────────
    "is_monday": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Binary: day is Monday",
    },
    "is_tuesday": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Binary: day is Tuesday",
    },
    "is_wednesday": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Binary: day is Wednesday",
    },
    "is_thursday": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Binary: day is Thursday",
    },
    "is_friday": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Binary: day is Friday",
    },
    "month_sin": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Cyclic month encoding: sin(2π × month / 12)",
    },
    "month_cos": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Cyclic month encoding: cos(2π × month / 12)",
    },
    "week_sin": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Cyclic week-of-year encoding: sin(2π × week / 52)",
    },
    "week_cos": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Cyclic week-of-year encoding: cos(2π × week / 52)",
    },
    "is_fomc_day": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Binary: FOMC announcement day",
    },
    "is_day_before_fomc": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Binary: trading day before an FOMC announcement",
    },
    "is_day_after_fomc": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Binary: trading day after an FOMC announcement",
    },
    "is_monthly_opex": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Binary: monthly options expiration Friday",
    },
    "is_quarterly_opex": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Binary: quarterly triple-witching options expiration",
    },
    "is_opex_week": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Binary: week containing options expiration",
    },
    "days_to_next_fomc": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Trading days until the next FOMC announcement",
    },
    "days_to_next_opex": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Trading days until the next options expiration Friday",
    },
    "is_month_end": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Binary: last trading day of the calendar month",
    },
    "is_quarter_end": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Binary: last trading day of the calendar quarter",
    },
    "is_first_trading_day": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Binary: first trading day of the calendar month",
    },
    "trading_days_remaining_month": {
        "category": "calendar", "module": "src.features.calendar_features",
        "requires_open": False, "lookback_window": 1,
        "description": "Estimated trading days remaining in the current month",
    },

    # ── OPTIONS FEATURES (Task 8) ─────────────────────────────────────────────
    "iv_rank_252": {
        "category": "options", "module": "src.features.options_features",
        "requires_open": False, "lookback_window": 253,
        "description": "IV Rank: (VIX - 252d min) / (252d max - 252d min)",
    },
    "iv_percentile_252": {
        "category": "options", "module": "src.features.options_features",
        "requires_open": False, "lookback_window": 253,
        "description": "IV Percentile: % of last 252 VIX values below current VIX",
    },
    "vix_term_structure_proxy": {
        "category": "options", "module": "src.features.options_features",
        "requires_open": False, "lookback_window": 6,
        "description": "VIX 5-day change — rising VIX indicates backwardation",
    },

    # ── LAGGED TARGET FEATURES (Task 9) ──────────────────────────────────────
    "prev_high_exceedance": {
        "category": "lagged_target", "module": "src.features.lagged_targets",
        "requires_open": False, "lookback_window": 2,
        "description": "(High_{t-1} - Close_{t-1}) / Close_{t-1}",
    },
    "prev_low_exceedance": {
        "category": "lagged_target", "module": "src.features.lagged_targets",
        "requires_open": False, "lookback_window": 2,
        "description": "(Close_{t-1} - Low_{t-1}) / Close_{t-1}",
    },
    "range_change": {
        "category": "lagged_target", "module": "src.features.lagged_targets",
        "requires_open": False, "lookback_window": 3,
        "description": "(Range_{t-1} - Range_{t-2}) / Range_{t-2}",
    },
    "high_direction_streak": {
        "category": "lagged_target", "module": "src.features.lagged_targets",
        "requires_open": False, "lookback_window": 5,
        "description": "Consecutive days where High_t > High_{t-1} (negative = declining)",
    },
    "low_direction_streak": {
        "category": "lagged_target", "module": "src.features.lagged_targets",
        "requires_open": False, "lookback_window": 5,
        "description": "Consecutive days where Low_t > Low_{t-1} (negative = declining)",
    },
    "inside_day_flag": {
        "category": "lagged_target", "module": "src.features.lagged_targets",
        "requires_open": False, "lookback_window": 3,
        "description": "1 if yesterday was an inside day (High < 2 prior High, Low > 2 prior Low)",
    },
    "inside_day_streak": {
        "category": "lagged_target", "module": "src.features.lagged_targets",
        "requires_open": False, "lookback_window": 5,
        "description": "Count of consecutive inside days (predicts range expansion)",
    },
    "outside_day_flag": {
        "category": "lagged_target", "module": "src.features.lagged_targets",
        "requires_open": False, "lookback_window": 3,
        "description": "1 if yesterday was an outside day (High > prior High, Low < prior Low)",
    },
}
