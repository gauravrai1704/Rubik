#!/usr/bin/env bash
# E4: solution length vs beam width for the learned methods.
source "$(dirname "$0")/common.sh"
for N in 3 5 10 20; do bench make --puzzle square --N $N --depth random --count 20; done
for W in 64 256 1024 4096 16384; do
  for N in 3 5 10 20; do
    bench run --solver m5b_stages --puzzle square --N $N --dataset random --device $DEVICE --time $TIME \
      --kw "{\"width\": $W, \"max_width\": $W}" --out results/e4/m5b_W${W}_N${N}.jsonl
  done
done
# Method 4 (per-N CayleyPy models; Squares 3-8 only)
for W in 64 256 1024 4096 16384; do
  for N in 3 5; do
    bench run --solver m4_cayleypy --puzzle square --N $N --dataset random --device $DEVICE --time $TIME \
      --kw "{\"width\": $W}" --out results/e4/m4_W${W}_N${N}.jsonl
  done
done
bench summary results/e4/*.jsonl
