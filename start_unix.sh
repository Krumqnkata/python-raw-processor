#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
fi
if [ ! -f .venv/raw-studio-deps-v1 ]; then
    .venv/bin/python -m pip install -r requirements.txt
    touch .venv/raw-studio-deps-v1
fi
exec .venv/bin/python raw_processor.py
