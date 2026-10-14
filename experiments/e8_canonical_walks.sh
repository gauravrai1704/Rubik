#!/usr/bin/env bash
# E8: 5A trained on plain vs canonical (commuting-move-free) random walks.
source "$(dirname "$0")/common.sh"
for MODE in plain canonical; do
  FLAG=$([ $MODE = canonical ] && echo --canonical-walks || true)
  python -m rubiks.solvers.m5a_tokens.train --puzzle square --train-N 4-8 --val-N 10 --steps ${STEPS:-20000} \
    $FLAG --device $DEVICE --out checkpoints/m5a/square_$MODE.pt
done
