#!/usr/bin/env bash
# Reproduce every reported metric with one command.
# Assumes deps are installed (pip install -r requirements.txt).
set -euo pipefail
cd "$(dirname "$0")"

# Prefer the local venv if present.
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="${PYTHON:-python}"; fi
# Models download once on first run, then run offline. To force a fully offline
# run after prefetching, invoke with HF_HUB_OFFLINE=1 ./run_eval.sh
export TOKENIZERS_PARALLELISM=false

echo "########## Corpus analysis (chunking decision) ##########"
$PY analyze_corpus.py

echo
echo "########## BASELINE ##########"
$PY eval.py --system baseline

echo
echo "########## IMPROVED (MiniLM — baseline's embedder) ##########"
$PY eval.py --system improved --model all-MiniLM-L6-v2

echo
echo "########## IMPROVED (e5-small-v2) ##########"
$PY eval.py --system improved --model e5-small-v2 || echo "(skipped: e5-small-v2 unavailable offline)"

echo
echo "########## IMPROVED (gte-small) ##########"
$PY eval.py --system improved --model gte-small || echo "(skipped: gte-small unavailable offline)"
