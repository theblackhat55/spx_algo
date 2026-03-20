#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/spx_algo")
LOG_PATH = ROOT / "output" / "monitoring" / "action_log.jsonl"

def main() -> None:
    p = argparse.ArgumentParser(description="Append an ops action log entry.")
    p.add_argument("--action", required=True)
    p.add_argument("--status", required=True)
    p.add_argument("--reason", default="")
    p.add_argument("--details-json", default="")
    args = p.parse_args()

    details = {}
    if args.details_json:
        try:
            details = json.loads(args.details_json)
        except Exception as exc:
            details = {"details_json_parse_error": str(exc), "raw": args.details_json}

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": args.action,
        "status": args.status,
        "reason": args.reason,
        "details": details,
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

    print("Logged action to:", LOG_PATH)
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
