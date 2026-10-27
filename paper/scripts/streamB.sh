#!/usr/bin/env bash
# Stream B: Method 4 width sweep, Method 5A, exact optima. One CPU thread.
set -x
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/../.."
B="python -m rubiks.bench"
python paper/scripts/optimal_square.py
for W in 16 64 256 1024 4096; do
  $B run --solver m4_cayleypy --puzzle square --N 3-8 --dataset random --time 600 \
    --kw "{\"width\": $W}" --out results/paper/m4_square_W$W.jsonl
done
$B run --solver m5a_tokens --puzzle square --N 4-10 --dataset random --time 300 \
  --kw '{"width": 512, "max_width": 2048}' --out results/paper/m5a_square.jsonl
$B run --solver m5a_tokens --puzzle square --N 12,14 --dataset random --limit 5 --time 900 \
  --kw '{"width": 512, "max_width": 2048}' --out results/paper/m5a_square_large.jsonl
