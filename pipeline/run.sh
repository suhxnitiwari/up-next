#!/bin/sh
# Rebuild everything from the Takeout export.
set -e
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=../.venv/bin/python
$PY parse.py
$PY enrich.py
$PY classify.py
$PY analyze.py
./frames.sh      # screens frames from inside each video for the opening
$PY analyze.py   # again, now with the screening
