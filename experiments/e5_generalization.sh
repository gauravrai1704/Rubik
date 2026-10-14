#!/usr/bin/env bash
# E5: one 5A model trained on N <= 7, evaluated on N = 3..20.
source "$(dirname "$0")/common.sh"
python -m rubiks.solvers.m5a_tokens.train --puzzle cube --train-N 3-7 --val-N 8,10,14 --steps ${STEPS:-50000} \
  --move-head --device $DEVICE --out checkpoints/m5a/cube.pt
bench make --puzzle cube --N 3-20 --depth random
bench run --solver m5a_tokens --puzzle cube --N 3-20 --dataset random --device $DEVICE --time $TIME \
  --out results/e5/m5a_cube.jsonl
bench summary results/e5/*.jsonl
