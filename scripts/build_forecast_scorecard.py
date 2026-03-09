#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from src.evaluation.forecast_scorecard import (
    build_scorecard,
    load_forecast_history,
    save_scorecard,
)


HISTORY_CSV = Path("output/reports/forecast_history.csv")
SCORECARD_JSON = Path("output/reports/forecast_scorecard.json")


def main() -> None:
    history = load_forecast_history(HISTORY_CSV)
    scorecard = build_scorecard(history)
    save_scorecard(scorecard, SCORECARD_JSON)

    print("Saved forecast scorecard to:", SCORECARD_JSON)
    print("Latest forecast date:", scorecard["latest_forecast_date"])
    print("\nAll-history summary:", scorecard["all_history"])
    print("Rolling-5 summary:", scorecard["rolling_5"])
    print("Rolling-20 summary:", scorecard["rolling_20"])
    print("Source counts:", scorecard["source_counts"])


if __name__ == "__main__":
    main()
