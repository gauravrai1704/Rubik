#!/usr/bin/env bash
# Methods 1-3 (Sections 4.5, 5.4, 6.5): optimal lengths, reduction vs Demaine on the same instances.
source "$(dirname "$0")/common.sh"
SQ_COUNT=${SQ_COUNT:-5}; CUBE_COUNT=${CUBE_COUNT:-2}
python -m rubiks.solvers.m1_idastar.label --gods 3
bench make --puzzle square --N 3-20 --depth random --count $SQ_COUNT
bench make --puzzle cube --N 3-10,12,15,20 --depth random --count $CUBE_COUNT
bench make --puzzle cube --N 3 --depth 12 --count ${M1_COUNT:-100}
# Method 1: Squares until IDA* stops finishing, 3x3x3 at scramble depth 12 and on random states
bench run --solver m1_idastar --puzzle square --N 3-6 --dataset random --time ${M1_TIME:-120} --out results/m1/square.jsonl
bench run --solver m1_idastar --puzzle cube --N 3 --dataset d12 --time ${M1_TIME:-120} --out results/m1/cube3_d12.jsonl
bench run --solver m1_idastar --puzzle cube --N 3 --dataset random --time ${M1_CUBE_TIME:-1800} --out results/m1/cube3_random.jsonl
# Methods 2 and 3: every N, both puzzles
for S in m2_reduction m3_demaine; do
  bench run --solver $S --puzzle square --N 3-20 --dataset random --time $TIME --out results/m23/${S}_square.jsonl
  bench run --solver $S --puzzle cube --N 3-10,12,15,20 --dataset random --time $TIME --out results/m23/${S}_cube.jsonl
done
bench summary results/m1/*.jsonl results/m23/*.jsonl
