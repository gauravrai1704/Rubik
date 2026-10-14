#!/usr/bin/env bash
# E2: solution length per method for N = 3..20, both puzzles, random states.
source "$(dirname "$0")/common.sh"
SOLVERS=${SOLVERS:-"m5b_stages m5a_tokens m2_reduction m3_demaine"}
for P in square cube; do
  bench make --puzzle $P --N 3-20 --depth random
  for S in $SOLVERS; do
    bench run --solver $S --puzzle $P --N ${NS:-3-20} --dataset random --time $TIME --device $DEVICE \
      --out results/e2/${S}_${P}.jsonl || echo "($S on $P failed or not implemented)"
  done
done
bench summary results/e2/*.jsonl
