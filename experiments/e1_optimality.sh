#!/usr/bin/env bash
# E1: optimality gap on 3x3x3 and small Squares. Method 1 fills optimal_len; others are compared.
source "$(dirname "$0")/common.sh"
bench make --puzzle cube --N 3 --depth random
bench make --puzzle square --N 3-6 --depth random
for S in m1_idastar m5b_stages m5a_tokens m4_cayleypy m2_reduction m3_demaine; do
  bench run --solver $S --puzzle square --N 3-6 --dataset random --time $TIME --device $DEVICE \
    --out results/e1/${S}_square.jsonl || echo "($S not available)"
done
bench summary results/e1/*.jsonl
