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


class _Server:
    """One long-lived nxopt process: its pruning table is loaded once and positions are sent one per line.
    Thread-safe (calls are serialised; nxopt itself is single-threaded per position)."""

    def __init__(self):
        import threading

        exe = Path(os.environ["NXOPT"])
        cwd = Path(os.environ.get("NXOPT_DIR", exe.parent))
        self.p = subprocess.Popen([str(exe), "-t", "1", "-"], cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._buf = b""  # our own line buffer: select() cannot see lines already read into Python's buffer
        self.lock = threading.Lock()
        import time as _t

        # nxopt buffers its output until it has solved a position, so send a one-move warm-up position and wait
        # (minutes from a cold disk) for its answer before any per-position time limit applies
        self.p.stdin.write("U1\n")
        self.p.stdin.flush()
        deadline = _t.time() + 1800
        while True:
            line = self._readline(deadline)
            if not line:
                raise RuntimeError("nxopt exited while loading its table")
            if line.startswith("Solved"):
                self._readline(deadline)  # its moves line
                break

    def _readline(self, deadline: float) -> str:
        import select
        import time as _t

        fd = self.p.stdout.fileno()
        while b"\n" not in self._buf:
            left = deadline - _t.time()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                self.p.kill()  # a hard position: give up; the next call starts a fresh process
                raise TimeoutError("nxopt exceeded its time limit")
            chunk = os.read(fd, 65536)
            if not chunk:  # process exited
                line, self._buf = self._buf, b""
                return line.decode()
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode() + "\n"

    def solve(self, scramble: str, timeout: float = 1200.0) -> list[Move]:
        import time as _t

        with self.lock:
            deadline = _t.time() + timeout
            self.p.stdin.write(scramble + "\n")
            self.p.stdin.flush()
            buf = []
            while True:
                line = self._readline(deadline)
                if not line:
                    raise RuntimeError("nxopt exited: " + "".join(buf[-5:]))
                buf.append(line)
                if line.startswith("Solved") or line.startswith("Did not solve"):
                    m = re.match(r"Solved \d+ in (\d+)", line)
                    if m and int(m.group(1)) > 0:
                        buf.append(self._readline(deadline))
                    return parse_output("".join(buf[-2:]))


_server: _Server | None = None
_server_lock = __import__("threading").Lock()


def optimal_from_solution(moves3: list[Move], timeout: float = 1200.0) -> list[Move]:
    """An optimal QTM solution of the 3x3x3 position that ``moves3`` (face turns) solves."""
    global _server
    inv = [m.inverse() for m in reversed(moves3)]
    if not inv:
        return []
    with _server_lock:  # one process for all threads (each loads the 1.42 GB table)
        if _server is None or _server.p.poll() is not None:
            _server = _Server()
        srv = _server
    key = _to_nxopt(inv)
    cache = _cache()
    if key in cache:
        return parse_notation(cache[key], _P3) if cache[key] else []
    moves = srv.solve(key, timeout)
    _remember(key, to_notation(moves, _P3) if moves else "")
    return moves


def _cache_path() -> Path:
    exe = Path(os.environ["NXOPT"])
    return Path(os.environ.get("NXOPT_DIR", exe.parent)) / "solutions.tsv"


def _cache() -> dict[str, str]:
    """Optimal solutions found earlier (scramble -> moves), so a rerun never searches the same position twice."""
    p = _cache_path()
    if not p.exists():
        return {}
    return dict(l.split("\t", 1) for l in p.read_text().split("\n") if "\t" in l)


def _remember(key: str, sol: str) -> None:
    with _server_lock, _cache_path().open("a") as fh:
        fh.write(f"{key}\t{sol}\n")


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
