#!/bin/sh
# Downloads three frames from inside every titled video (YouTube's hq1/hq2/hq3) and screens them with
# screen.swift, so the opening only plays clips of real people with no drawings or on-screen text.
set -e
cd "$(dirname "$0")"
mkdir -p ../data/frames
../.venv/bin/python -c "import json,re;s=open('../site/data.js').read();d=json.loads(s[s.index('{'):s.rindex('}')+1]);print('\n'.join(c['id'] for c in d['all']))" |
  while read id; do for f in hq1 hq2 hq3; do [ -s "../data/frames/${id}_$f.jpg" ] || echo "https://i.ytimg.com/vi/$id/$f.jpg ../data/frames/${id}_$f.jpg"; done; done |
  xargs -P 16 -n 2 sh -c 'curl -s -f -o "$1" "$0" || true'
swift screen.swift ../data/frames ../data/screen.json
