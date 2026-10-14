#!/usr/bin/env bash
# E6: same solver and width, beam stored on device vs host vs disk. Compare time and peak memory.
source "$(dirname "$0")/common.sh"
bench make --puzzle square --N 20 --depth random --count 10
for T in device host disk; do
  bench run --solver m5b_stages --puzzle square --N 20 --dataset random --device $DEVICE --time $TIME \
    --kw "{\"width\": ${W:-8192}, \"max_width\": ${W:-8192}, \"tier\": \"$T\"}" --out results/e6/tier_${T}.jsonl
done
bench summary results/e6/*.jsonl
