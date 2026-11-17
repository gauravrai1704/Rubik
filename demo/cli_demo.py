"""Terminal version of the demo (fallback if the browser misbehaves). Same three parts as the web page.

    python demo/cli_demo.py                 # Square 6, cube 6, Santa 4x4x4 puzzle 162
    python demo/cli_demo.py --square 8 --cube 8 --santa 9
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rubiks.engine import make, metrics, random_state
from rubiks.engine.puzzle import cube
from rubiks.solvers import Budget, get_solver, load_all

HERE = Path(__file__).resolve().parent
COL = "WYGBOR"  # colour letters for the Square's top face


def show_square_top(P, s) -> None:
    """Print the Square's top face as letters (one colour per letter)."""
    top = [i for i, n in enumerate(P.sticker_normal) if tuple(n) == (0, 0, 1)]
    pos = P.sticker_pos
    N = P.N
    grid = [["?"] * N for _ in range(N)]
    for i in top:
        x, y = (pos[i][0] + N - 1) // 2, (N - 1 - pos[i][1]) // 2
        grid[y][x] = COL[int(s[i]) % 6]
    for row in grid:
        print("   " + " ".join(row))


def run(kind: str, N: int, names: list[tuple[str, str, dict]], seed: int = 1) -> None:
    P = make(kind, N)
    s = random_state(P, np.random.default_rng(seed))
    print(f"\n== {kind} {N}x{N}x{1 if kind == 'square' else N}, random state (seed {seed}) ==")
    if kind == "square":
        print("   top face:")
        show_square_top(P, s)
    print(f"   {'method':<11}{'HTM':>7}{'QTM':>7}{'time':>9}   verified")
    for shown, reg, kw in names:
        t0 = time.time()
        r = get_solver(reg, P, Budget(wall_time_s=120), **kw).solve(s)
        ok = P.is_solved(P.apply(s, r.moves))  # re-applied by the engine, like the harness
        m = metrics(r.moves, P)
        print(f"   {shown:<11}{m['htm']:>7}{m['qtm']:>7}{time.time() - t0:>8.1f}s   {'yes' if ok else 'NO'}")


def santa(pick: int) -> None:
    data = json.loads((HERE / "data" / "santa.json").read_text())
    p = data["puzzles"][pick]
    P = cube(p["N"])
    s = np.asarray(p["state"], dtype=P.solved.dtype)
    print(f"\n== Santa 2023, {p['N']}x{p['N']}x{p['N']} puzzle {p['pid']} (exact goal, quarter turns) ==")
    lens = sorted(c["qtm"] for c in p["copies"])
    print(f"   {len(lens)} symmetric copies solved by KMCoders' program, lengths {lens[0]}..{lens[-1]}:")
    lo = lens[0]
    for v in sorted(set(lens)):
        mark = "  <- SymPort keeps this" if v == lo else ""
        print(f"   {v:>4} {'*' * lens.count(v)}{mark}")
    for name, key in (("SymPort", "symport"), ("KMC-published", "published")):
        end = P.apply(s, p[key]["ids"])
        print(f"   {name:<14}{p[key]['qtm']:>5} quarter turns   exact goal: {'yes' if np.array_equal(end, P.solved) else 'NO'}")
    print(f"   {'KMC one run':<14}{p['one_run']:>5} quarter turns")
    print("\n   Table 2 (mean quarter turns per Santa size):")
    print(f"   {'N':>3}{'n':>5}{'SymPort':>10}{'published':>11}{'one run':>9}")
    for t in data["table"]:
        w = " <" if t["symport"] < t["published"] else ""
        print(f"   {t['N']:>3}{t['n']:>5}{t['symport']:>10.1f}{t['published']:>11.1f}{t['one_run']:>9.1f}{w}")
    print("   (< : SymPort shorter on average; it wins for N = 4..7 and is second at 8 and 9)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--square", type=int, default=6)
    ap.add_argument("--cube", type=int, default=6)
    ap.add_argument("--santa", type=int, default=0, help="index into demo/data/santa.json (0 = 4x4x4 puzzle 162)")
    a = ap.parse_args()
    load_all()
    sq = [("Reduction", "m2_reduction", {}), ("Demaine", "m3_demaine", {}), ("NetGain", "m5c_schedule", {})]
    if a.square <= 8:
        sq.append(("CayleyPy", "m4_cayleypy", {"width": 1024}))
    inners = ([("m4_cayleypy", {"width": 1024}), ("m5c_schedule", {})] if a.square <= 8
              else [("m5c_schedule", {}), ("m3_demaine", {})])
    sq.append(("SymPort", "m7_portfolio", {"copies": 8, "workers": 2, "inners": inners}))
    run("square", a.square, sq)
    run("cube", a.cube, [("Reduction", "m2_reduction", {}), ("Demaine", "m3_demaine", {}), ("NetGain", "m5c_schedule", {})])
    santa(a.santa)


if __name__ == "__main__":
    main()
