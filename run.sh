#!/usr/bin/env bash
# Starts the demo at http://127.0.0.1:7860
cd "$(dirname "$0")"
source .venv/bin/activate
python app.py
