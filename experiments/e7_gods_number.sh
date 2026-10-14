#!/usr/bin/env bash
# E7: counting lower bound (Demaine 2011, Thm 2/4) and the longest solution our best solver needed.
source "$(dirname "$0")/common.sh"
python - <<'PY'
import math
for N in range(4, 21):
    k = (N // 2 - 1) ** 2
    sq = (k * math.log(6)) / math.log(2 * N) - 1
    cu = (k * math.log(math.factorial(24) / math.factorial(4) ** 6)) / math.log(6 * N) - 1
    print(f"N={N:2d}  square lower bound >= {max(0, math.ceil(sq)):4d}   cube lower bound >= {max(0, math.ceil(cu)):5d}")
PY
echo "upper-bound evidence: max HTM per N in results/e2 (see 'summary' htm_max column)"
