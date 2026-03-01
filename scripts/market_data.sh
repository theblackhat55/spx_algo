#!/bin/bash
AV_KEY="79OVVCSG7GV0CQQE"
BASE="https://www.alphavantage.co/query"

echo "=== SPX QUOTE ==="
curl -s "${BASE}?function=GLOBAL_QUOTE&symbol=SPY&apikey=${AV_KEY}" | python3 -c "
import json,sys
d=json.load(sys.stdin).get('Global Quote',{})
if d: print(f\"SPY: {d.get('05. price','')} | Change: {d.get('10. change percent','')} | Date: {d.get('07. latest trading day','')}\")
else: print('No data')" 2>/dev/null

sleep 2

echo ""
echo "=== VIX ==="
curl -s "${BASE}?function=GLOBAL_QUOTE&symbol=VIX&apikey=${AV_KEY}" | python3 -c "
import json,sys
d=json.load(sys.stdin).get('Global Quote',{})
if d: print(f\"VIX: {d.get('05. price','')} | Change: {d.get('10. change percent','')}\")
else: print('No data')" 2>/dev/null

sleep 2

echo ""
echo "=== OIL (USO) ==="
curl -s "${BASE}?function=GLOBAL_QUOTE&symbol=USO&apikey=${AV_KEY}" | python3 -c "
import json,sys
d=json.load(sys.stdin).get('Global Quote',{})
if d: print(f\"USO: {d.get('05. price','')} | Change: {d.get('10. change percent','')}\")
else: print('No data')" 2>/dev/null

sleep 2

echo ""
echo "=== NEWS SENTIMENT ==="
curl -s "${BASE}?function=NEWS_SENTIMENT&tickers=SPY,VIX&limit=5&apikey=${AV_KEY}" | python3 -c "
import json,sys
data=json.load(sys.stdin)
for item in data.get('feed',[])[:5]:
    print(f\"- {item.get('title','')} | {item.get('overall_sentiment_label','')} ({item.get('overall_sentiment_score','')})\")
if not data.get('feed'): print('No news data (may require premium)')" 2>/dev/null

sleep 2

echo ""
echo "=== TOP MOVERS ==="
curl -s "${BASE}?function=TOP_GAINERS_LOSERS&apikey=${AV_KEY}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
if 'top_losers' in d:
    print('Top Losers:')
    for x in d.get('top_losers',[])[:3]:
        print(f\"  {x.get('ticker')}: {x.get('change_percentage')}\")
    print('Top Gainers:')
    for x in d.get('top_gainers',[])[:3]:
        print(f\"  {x.get('ticker')}: {x.get('change_percentage')}\")
else: print('No data (may require premium)')" 2>/dev/null
