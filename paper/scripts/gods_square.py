"""God's number of the Rubik's Square for N = 3, 4, 5 by full BFS from the 4 solved colourings
(Method 1, rubiks.solvers.m1_idastar.square.gods_number). N = 5 takes ~15 min and ~6 GB.

    python paper/scripts/gods_square.py      # writes paper/results/gods_square.json
"""
import json
from pathlib import Path

from rubiks.solvers.m1_idastar.square import gods_number

out = {str(N): gods_number(N, "colour") for N in (3, 4, 5)}
Path("paper/results/gods_square.json").write_text(json.dumps(out, indent=1))
print(out)
