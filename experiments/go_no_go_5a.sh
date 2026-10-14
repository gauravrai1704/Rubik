#!/usr/bin/env bash
# Section 8.6 go / no-go: train 5A on Squares N=4..8, test on N=10,14,20.
# Pass = 100% of N=10 solved, in fewer moves than Method 2.
source "$(dirname "$0")/common.sh"
python -m rubiks.solvers.m5a_tokens.train --puzzle square --train-N 4-8 --val-N 10,14,20 --steps ${STEPS:-20000} \
  --device $DEVICE --out checkpoints/m5a/square.pt
bench make --puzzle square --N 10,14,20 --depth random
bench run --solver m5a_tokens --puzzle square --N 10,14,20 --dataset random --device $DEVICE --time $TIME \
  --out results/go_no_go/m5a.jsonl
bench run --solver m5b_stages --puzzle square --N 10,14,20 --dataset random --device $DEVICE --time $TIME \
  --out results/go_no_go/m5b.jsonl
bench run --solver m2_reduction --puzzle square --N 10,14,20 --dataset random --time $TIME \
  --out results/go_no_go/m2.jsonl || echo "(Method 2 not implemented yet)"
bench summary results/go_no_go/*.jsonl
