#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_HYBRID_ARCHIVE_DIR = Path("output/forecasts/archive")
DEFAULT_RANGE_SKEW_ARCHIVE_DIR = Path("output/forecasts/archive")
DEFAULT_COMPARE_SCRIPT = Path("scripts/compare_daily_hybrid_vs_range_skew_actuals.py")


def main():
    ap = argparse.ArgumentParser(
        description="Run daily hybrid vs range+skew comparison using archived forecast files"
    )
    ap.add_argument("--date", required=True, help="Forecast date YYYY-MM-DD")
    ap.add_argument(
        "--hybrid-archive-dir",
        default=str(DEFAULT_HYBRID_ARCHIVE_DIR),
        help="Directory containing archived hybrid forecasts",
    )
    ap.add_argument(
        "--range-skew-archive-dir",
        default=str(DEFAULT_RANGE_SKEW_ARCHIVE_DIR),
        help="Directory containing archived range+skew forecasts",
    )
    ap.add_argument(
        "--compare-script",
        default=str(DEFAULT_COMPARE_SCRIPT),
        help="Path to compare_daily_hybrid_vs_range_skew_actuals.py",
    )
    ap.add_argument(
        "--outdir",
        default="output/reports/daily_forecast_comparison",
        help="Output directory for comparison JSON and scorecard CSV",
    )
    ap.add_argument(
        "--allow-date-override",
        action="store_true",
        help="Allow running compare script with explicit --date override",
    )
    args = ap.parse_args()

    hybrid_dir = Path(args.hybrid_archive_dir)
    rs_dir = Path(args.range_skew_archive_dir)
    compare_script = Path(args.compare_script)

    hybrid_candidates = [
        hybrid_dir / f"{args.date}_gap_augmented_hybrid_ohlc_forecast.json",
        hybrid_dir / f"{args.date}_hybrid_ohlc_forecast.json",
        hybrid_dir / f"{args.date}_gap_augmented_hybrid_forecast.json",
    ]
    rs_candidates = [
        rs_dir / f"{args.date}_gap_augmented_range_skew_forecast.json",
        rs_dir / f"{args.date}_range_skew_forecast.json",
    ]

    hybrid_path = next((p for p in hybrid_candidates if p.exists()), None)
    rs_path = next((p for p in rs_candidates if p.exists()), None)

    if hybrid_path is None:
        print("ERROR: Could not find archived hybrid forecast for date:", args.date, file=sys.stderr)
        print("Checked:", *[str(p) for p in hybrid_candidates], sep="\n  - ", file=sys.stderr)
        sys.exit(1)

    if rs_path is None:
        print("ERROR: Could not find archived range+skew forecast for date:", args.date, file=sys.stderr)
        print("Checked:", *[str(p) for p in rs_candidates], sep="\n  - ", file=sys.stderr)
        sys.exit(1)

    if not compare_script.exists():
        print(f"ERROR: Compare script not found: {compare_script}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        str(compare_script),
        "--hybrid-json", str(hybrid_path),
        "--range-skew-json", str(rs_path),
        "--outdir", args.outdir,
    ]

    if args.allow_date_override:
        cmd.extend(["--date", args.date])

    print("Running archived daily comparison:")
    print("  hybrid forecast:     ", hybrid_path)
    print("  range+skew forecast: ", rs_path)
    print("  compare script:      ", compare_script)
    print("  outdir:              ", args.outdir)
    print("  date:                ", args.date)
    print()

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
