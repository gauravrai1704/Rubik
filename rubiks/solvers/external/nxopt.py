"""Optimal quarter-turn solutions of the 3x3x3 with Tomas Rokicki's ``nxopt`` (github.com/rokicki/cube20src,
GPL-3), used for the odd-N "3x3x3 stage" of KMCoders' pipeline (corners, middle edges, fixed centers), which
KMCoders solved with an external optimal solver in the competition.

Setup (once): clone cube20src, ``ctangle nxopt.w; ctangle cubepos.w`` and build the quarter-turn binary,
e.g. ``g++ -O3 -march=native -std=c++11 -DPS=16 -DQUARTER -DTHREADS -DEO1 -DEP3 -o nxopt13q nxopt.cpp
cubepos.cpp -lpthread``; set ``NXOPT`` to the binary. Its pruning table (1.42 GB for EO1/EP3) is built on
the first run and saved in the working directory (``NXOPT_DIR``, default: the binary's folder).

nxopt reads positions from stdin as a scramble (face letter + number of clockwise quarter turns) and prints
an optimal solution for each; we pass the inverse of any known solution (e.g. Kociemba's) as the scramble.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from ...engine import cube
from ...engine.moves import parse_notation, to_notation
from ...engine.puzzle import Move

_P3 = cube(3)


def available() -> bool:
    return bool(os.environ.get("NXOPT")) and Path(os.environ["NXOPT"]).exists()


def _to_nxopt(moves3: list[Move]) -> str:
    """Engine 3x3x3 face turns -> nxopt scramble tokens (U1 = U clockwise, U3 = U')."""
    out = []
    for tok in to_notation(moves3, _P3).split():
        face, suf = tok[0], tok[1:]
        out.append(face + {"": "1", "2": "2", "'": "3"}[suf])
    return " ".join(out)


def _from_nxopt(text: str) -> list[Move]:
    toks = re.findall(r"([URFDLB])([123'2]?)", text)
    note = " ".join(f + {"1": "", "": "", "2": "2", "3": "'", "'": "'"}[s] for f, s in toks)
    return parse_notation(note, _P3) if note else []


def optimal_from_solution(moves3: list[Move], timeout: float = 3600.0) -> list[Move]:
    """An optimal QTM solution of the 3x3x3 position that ``moves3`` (face turns) solves."""
    inv = [m.inverse() for m in reversed(moves3)]
    exe = Path(os.environ["NXOPT"])
    cwd = Path(os.environ.get("NXOPT_DIR", exe.parent))
    r = subprocess.run([str(exe), "-t", str(os.cpu_count() or 1), "-"], input=_to_nxopt(inv) + "\n",
                       cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return parse_output(r.stdout)


def parse_output(stdout: str) -> list[Move]:
    """nxopt prints ``Solved <seq> in <length> probes ...`` and then the moves (``U1 R3 F2``) on the next line."""
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"Solved \d+ in (\d+)", line)
        if m:
            length = int(m.group(1))
            if length == 0:
                return []
            mv = _from_nxopt(lines[i + 1] if i + 1 < len(lines) else "")
            if sum(2 if x.turn == 2 else 1 for x in mv) != length:
                raise RuntimeError(f"nxopt reported length {length} but printed {lines[i + 1]!r}")
            return mv
    raise RuntimeError("no solution in nxopt output: " + stdout[-400:])
