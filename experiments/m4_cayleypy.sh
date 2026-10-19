#!/usr/bin/env bash
# Method 4 (Section 7.3-7.4): train the paper's 4M ResMLPs for cubes 3-5 and per-N Square models,
# then length vs beam width W. Run on the GPU machine: DEVICE=cuda EPOCHS=8192 bash experiments/m4_cayleypy.sh
# Training time and peak GPU memory land in checkpoints/m4/<puzzle><N>*.json.
source "$(dirname "$0")/common.sh"
EPOCHS=${EPOCHS:-8192}          # x 1M examples = the paper's 8B
SQ_EPOCHS=${SQ_EPOCHS:-1024}
WIDTHS=${WIDTHS:-"1024 4096 16384 65536 262144"}
for N in 3 4 5; do
  [ -f checkpoints/m4/cube$N.large.pt ] || python -m rubiks.solvers.m4_cayleypy.train --puzzle cube --N $N \
    --preset paper$N --epochs $EPOCHS --log-every 64 --device $DEVICE --out checkpoints/m4/cube$N.large.pt
done
for N in 3 4 5 6 7 8; do
  [ -f checkpoints/m4/square$N.large.pt ] || python -m rubiks.solvers.m4_cayleypy.train --puzzle square --N $N \
    --preset 1M --epochs $SQ_EPOCHS --log-every 64 --device $DEVICE --out checkpoints/m4/square$N.large.pt
done
bench make --puzzle cube --N 3-5 --depth random --count ${COUNT:-20}
bench make --puzzle square --N 3-8 --depth random
for W in $WIDTHS; do   # length-vs-W curve (Section 7.4); add --kw tier "host" for widths past GPU memory
  bench run --solver m4_cayleypy --puzzle cube --N 3-4 --dataset random --device $DEVICE --time $TIME \
    --kw "{\"width\": $W}" --out results/m4/cube_W$W.jsonl
done
bench run --solver m4_cayleypy --puzzle cube --N 5 --dataset random --device $DEVICE --time $TIME \
  --kw '{"width": 262144}' --out results/m4/cube5.jsonl
bench run --solver m4_cayleypy --puzzle square --N 3-8 --dataset random --device $DEVICE --time $TIME \
  --kw '{"width": 4096, "max_width": 65536}' --out results/m4/square.jsonl
bench summary results/m4/*.jsonl
