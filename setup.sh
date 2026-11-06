#!/usr/bin/env bash
# One-time setup: creates a virtual environment in .venv and installs the requirements.
# The embedding model (130 MB) downloads on first start; the open LLM (about 8 GB) on the first question.
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.11 || command -v python3.12 || command -v python3)
echo "Using $($PY --version)"
$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo
echo "Setup complete. Start the demo with:  ./run.sh"
