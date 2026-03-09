#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
LATEST_FORECAST = REPO_ROOT / "output" / "forecasts" / "latest_gap_augmented_range_skew_forecast.json"
ARCHIVE_DIR = REPO_ROOT / "output" / "forecasts" / "archive"
GENERATOR = REPO_ROOT / "scripts" / "generate_gap_augmented_range_skew_forecast.py"


def main() -> None:
    if not GENERATOR.exists():
        raise SystemExit(f"Generator script not found: {GENERATOR}")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(GENERATOR)]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)

    if result.stdout.strip():
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr.strip():
            print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)

    if not LATEST_FORECAST.exists():
        raise SystemExit(f"Latest forecast file not found after generation: {LATEST_FORECAST}")

    payload = json.loads(LATEST_FORECAST.read_text())
    forecast_date = payload.get("forecast_for_date")
    if not forecast_date:
        raise SystemExit("forecast_for_date missing from latest range+skew forecast JSON")

    archive_path = ARCHIVE_DIR / f"{forecast_date}_gap_augmented_range_skew_forecast.json"
    shutil.copy2(LATEST_FORECAST, archive_path)

    print("=== GAP AUGMENTED RANGE+SKEW STEP CHECK ===")
    print("latest_forecast:", LATEST_FORECAST)
    print("archive_forecast:", archive_path)
    print("forecast_for_date:", payload.get("forecast_for_date"))
    print("generated_from_feature_date:", payload.get("generated_from_feature_date"))
    print("source_selection:", payload.get("source_selection"))
    print("range_skew_overlay:", payload.get("range_skew_overlay"))
    print("predicted_ohlc:", payload.get("predicted_ohlc"))


if __name__ == "__main__":
    main()
