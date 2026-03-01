#!/bin/bash
set -euo pipefail
cd /root/spx_algo
source .venv/bin/activate
python3.11 /root/spx_algo/scripts/whatsapp_report_inner.py
