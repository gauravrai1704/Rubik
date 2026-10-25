#!/usr/bin/env bash
# Stream A: Method 5B Square sweep and 5B ablations. One CPU thread.
set -x
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/../.."
B="python -m rubiks.bench"
$B run --solver m5b_stages --puzzle square --N 3-20 --dataset random --time 900 \
  --kw '{"width": 128, "max_width": 2048}' --out results/paper/m5b_square.jsonl
python paper/scripts/run_ablations.py square
python paper/scripts/run_ablations.py cube
