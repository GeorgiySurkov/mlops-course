#!/usr/bin/env bash
# Stage 2: run two experiments with different hyperparameters, each enqueued to the
# agent on the students queue. Add more by calling train.py with other flags.
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">> experiment 1: baseline (unigrams)"
uv run python train.py --name baseline --max-features 10000 --ngram-min 1 --ngram-max 1 --C 1.0 --max-iter 300

echo ">> experiment 2: bigrams, more features, stronger regularization"
uv run python train.py --name bigrams --max-features 20000 --ngram-min 1 --ngram-max 2 --C 0.5 --max-iter 500

echo ">> both experiments enqueued — watch them run on the agent in the ClearML UI"
