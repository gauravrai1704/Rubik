#!/usr/bin/env bash
# Stream C: Methods 1-3 and 5C on the paper's instances (4-core CPU).
set -x
cd "$(dirname "$0")/../.."
B="python -m rubiks.bench"
R=paper/results
for S in m2_reduction m3_demaine m5c_schedule; do
  $B run --solver $S --puzzle square --N 3-20 --dataset random --time 900 --out $R/${S}_square.jsonl
  $B run --solver $S --puzzle cube --N 3-8,10,15,20 --dataset random --time 1200 --out $R/${S}_cube.jsonl
done
$B run --solver m1_idastar --puzzle square --N 3-6 --dataset random --time 120 --out $R/m1_square.jsonl
$B make --puzzle cube --N 3 --depth 12 --count 100
$B run --solver m1_idastar --puzzle cube --N 3 --dataset d12 --time 120 --out $R/m1_cube3_d12.jsonl
$B run --solver m1_idastar --puzzle cube --N 3 --dataset random --limit 2 --time 1800 --out $R/m1_cube3_random.jsonl
mkdir -p $R/ablations
$B run --solver m5c_schedule --puzzle cube --N 6-8,10,15 --dataset random --limit 2 --time 1800 \
  --kw '{"lookahead": 0}' --out $R/ablations/cube5c_no_lookahead.jsonl
$B run --solver m5c_schedule --puzzle cube --N 6-8,10 --dataset random --limit 2 --time 1800 \
  --kw '{"lookahead": 1, "share_setups": false}' --out $R/ablations/cube5c_no_setup_sharing.jsonl
$B run --solver m5c_schedule --puzzle cube --N 20 --dataset random --limit 1 --time 3000 \
  --kw '{"lookahead": 1}' --out $R/ablations/cube5c_lookahead_n20.jsonl
