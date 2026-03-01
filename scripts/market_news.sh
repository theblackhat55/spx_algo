#!/bin/bash
FINNHUB_KEY="d6ia069r01ql9cifhvagd6ia069r01ql9cifhvb0"

echo "=== MARKET NEWS (General) ==="
curl -s "https://finnhub.io/api/v1/news?category=general&token=${FINNHUB_KEY}" | python3 -c "
import json,sys
articles=json.load(sys.stdin)
for a in articles[:7]:
    print(f\"- [{a.get('source','')}] {a.get('headline','')}\")" 2>/dev/null

sleep 1

echo ""
echo "=== SPX/SPY NEWS ==="
curl -s "https://finnhub.io/api/v1/company-news?symbol=SPY&from=$(date -d '-2 days' +%Y-%m-%d)&to=$(date +%Y-%m-%d)&token=${FINNHUB_KEY}" | python3 -c "
import json,sys
articles=json.load(sys.stdin)
for a in articles[:5]:
    print(f\"- [{a.get('source','')}] {a.get('headline','')}\")" 2>/dev/null

sleep 1

echo ""
echo "=== MARKET SENTIMENT (SPY) ==="
curl -s "https://finnhub.io/api/v1/news-sentiment?symbol=SPY&token=${FINNHUB_KEY}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
s=d.get('sentiment',{})
buzz=d.get('buzz',{})
print(f\"Sentiment: {s.get('bearishPercent',0):.0%} bearish / {s.get('bullishPercent',0):.0%} bullish\")
print(f\"Buzz: {buzz.get('articlesInLastWeek',0)} articles this week, {buzz.get('buzz',0):.2f}x normal volume\")" 2>/dev/null
