#!/usr/bin/env bash
# E3: length / (N²/ln N) per N. Reuses E2's results; the summary prints the norm_htm column.
source "$(dirname "$0")/common.sh"
[ -d results/e2 ] || bash "$(dirname "$0")/e2_lengths.sh"
bench summary results/e2/*.jsonl
