"""Method 1: Square BFS God's number, IDA* optimality against BFS, 3×3×3 cubie tables and IDA*."""
import numpy as np
import pytest

from rubiks.engine import cube, random_state, square
from rubiks.solvers import Budget, get_solver
from rubiks.solvers.cluster_tools import orient
from rubiks.solvers.m1_idastar.square import gods_number, ida_star


def test_square3_gods_number():
    g = gods_number(3, goal="position")
    assert g["states"] == 768 == sum(g["distribution"]) and g["gods_number"] == 8
    c = gods_number(3)
    assert c["states"] == 768 and c["distribution"][0] == 4 and c["gods_number"] <= 8


def test_square_ida_is_optimal_against_bfs(rng):
    P = square(3)
    # BFS distances from solved (moves are involutions, so distance is symmetric)
    dist = {P.solved.astype(np.int8).tobytes(): 0}
    frontier = [P.solved.astype(np.int8)]
    d = 0
    while frontier:
        d += 1
        nxt = []
        for st in frontier:
            for p in P.move_perms:
                k = st[p]
                if k.tobytes() not in dist:
                    dist[k.tobytes()] = d
                    nxt.append(k)
        frontier = nxt
    for _ in range(10):
        s = random_state(P, rng).astype(np.int8)
        ids, _ = ida_star(3, s, time_limit=30)
        assert len(ids) == dist[s.tobytes()]
        assert P.is_solved(P.apply(s, ids))


def test_square_any_orientation_is_colour_optimal(rng):
    """Multi-source BFS from the 4 solved colourings gives the true colour-solving distance."""
    from rubiks.solvers.m1_idastar.square import ida_star_any_orientation

    P = square(3)
    starts = [P.solved[R].astype(np.int8) for R in P.rotations]
    dist = {s.tobytes(): 0 for s in starts}
    frontier, d = starts, 0
    while frontier:
        d += 1
        nxt = []
        for st in frontier:
            for p in P.move_perms:
                k = st[p]
                if k.tobytes() not in dist:
                    dist[k.tobytes()] = d
                    nxt.append(k)
        frontier = nxt
    for _ in range(15):
        s = random_state(P, rng).astype(np.int8)
        ids, _ = ida_star_any_orientation(3, s, time_limit=30)
        assert len(ids) == dist[s.tobytes()] and P.is_solved(P.apply(s, ids))


def test_solver_square(rng):
    P = square(4)
    S = get_solver("m1_idastar", P, Budget(wall_time_s=60))
    s = random_state(P, rng)
    r = S.solve(s)
    assert r.info["optimal"] and P.is_solved(P.apply(s, r.moves))


def test_out_of_scope():
    with pytest.raises(ValueError, match="3×3×3"):
        get_solver("m1_idastar", cube(4))


def test_cubie_tables_match_engine(rng):
    from rubiks.solvers.m1_idastar.cube3 import Cubies, build_tables, ori_index, rank_perm

    cb = Cubies("htm")
    P = cb.P
    cpm, com, etab, _ = build_tables(cb)
    for _ in range(50):
        seq = [cb.moves[i] for i in rng.integers(0, len(cb.moves), 15)]
        cp, co, ep, eo = cb.from_colours(P.apply(P.solved, seq))
        ci = oi = 0
        codes = np.arange(12) * 2
        for m in seq:
            k = cb.moves.index(m)
            ci, oi, codes = cpm[k, ci], com[k, oi], etab[k, codes]
        assert ci == rank_perm(cp) and oi == ori_index(co, 3)
        want = np.zeros(12, dtype=np.int64)
        want[ep] = np.arange(12) * 2 + eo
        assert np.array_equal(codes, want)


@pytest.mark.slow
@pytest.mark.parametrize("metric", ["htm", "qtm"])
def test_cube3_optimal_short_scrambles(metric, rng):
    """Builds (or loads) the 170 MB pattern databases on first run."""
    from rubiks.solvers.m1_idastar.cube3 import databases

    cb = databases(metric)[0]
    P = cube(3)
    for depth in (6, 9):
        last, seq = len(cb.moves), []
        for _ in range(depth):
            m = int(rng.choice(np.nonzero(cb.allowed[last])[0]))
            seq.append(P.moves[cb.moves[m]])
            last = m
        s, _ = orient(P, P.apply(P.solved, seq))
        r = get_solver("m1_idastar", P, Budget(wall_time_s=300), metric=metric).solve(s)
        assert r.info["optimal"] and P.is_solved(P.apply(s, r.moves))
        assert len(r.moves) <= depth
