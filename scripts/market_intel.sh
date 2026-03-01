#!/bin/bash
set -euo pipefail
cd /root/spx_algo
source .venv/bin/activate

python3.11 -c "
import json, datetime
from pathlib import Path

intel = {
    'date': (datetime.date.today() + datetime.timedelta(days=1)).isoformat(),
    'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'risk_score': 0,
    'key_events': [],
    'sentiment_summary': '',
    'tail_risk_flag': False,
    'reasoning': ''
}

try:
    with open('output/signals/latest_signal.json') as f:
        sig = json.load(f)
    intel['vix_current'] = sig.get('vix_spot', 0)
    intel['prior_close'] = sig.get('prior_close', 0)
    intel['regime'] = sig.get('regime', 'UNKNOWN')
except Exception:
    intel['vix_current'] = 0
    intel['prior_close'] = 0
    intel['regime'] = 'UNKNOWN'

outpath = Path('data/processed/market_intel.json')
outpath.parent.mkdir(parents=True, exist_ok=True)
outpath.write_text(json.dumps(intel, indent=2))
print(json.dumps(intel, indent=2))
"
