"""Method 6: insertion algebra, annealer exactness, optimal corners, solving (random and Santa states)."""
import numpy as np
import pytest

from rubiks.engine import clusters, cube, random_state
from rubiks.solvers import get_solver
from rubiks.solvers.m6_kmcoders.anneal import Annealer
from rubiks.solvers.m6_kmcoders.sequence import InsertionSequence


def test_insertion_matches_replay(rng):
    P = cube(5)
    s0 = random_state(P, rng)
    seq = InsertionSequence(P, s0, [int(x) for x in rng.integers(0, len(P.moves), 20)])
    for _ in range(30):
        t = int(rng.integers(0, seq.T + 1))
        B = [int(x) for x in rng.integers(0, len(P.moves), 3)]
        PB = np.arange(P.num_stickers)
        for m in B:
            PB = PB[P.move_perms[m]]
        predicted = seq.final()[seq.inv[t][PB[seq.suf[t]]]]
        seq.insert(t, B)
        assert np.array_equal(predicted, seq.final())
        assert np.array_equal(seq.final(), P.apply(s0, seq.moves))


def test_annealer_two_time_edits_are_exact(rng):
    P = cube(4)
    s0 = random_state(P, rng)
    A = Annealer(P, s0, [int(x) for x in rng.integers(0, len(P.moves), 12)], P.solved, seed=3)
    for _ in range(200):
        e = A.propose2()
        if not e:
            continue
        d, nf = A.delta2(e)
        mv = [int(x) for x in A.moves]
        for t, k, new in reversed(e):
            mv[t:t + k] = new
        assert np.array_equal(P.apply(s0, mv), nf)
        if rng.random() < 0.5:
            A._apply_edits(e, nf, d)


def test_optimal_corners_qtm(rng):
    from rubiks.solvers.m6_kmcoders.corners import corner_solution_qtm

    P = cube(4)
    corner = next(c for c in clusters(P) if c.kind == "corner")
    s = random_state(P, rng)
    m = corner_solution_qtm(P, s)
    assert len(m) <= 14  # 2x2x2 God's number in QTM
    assert np.array_equal(P.apply(s, m)[corner.stickers], P.solved[corner.stickers])


@pytest.mark.parametrize("N", [4, 5])
def test_solves_random_cubes(N, rng):
    P = cube(N)
    s = random_state(P, rng)
    r = get_solver("m6_kmcoders", P, anneal_iters=20000).solve(s)
    assert P.is_solved(P.apply(s, r.moves))


def test_exact_goal_on_santa():
    santa = pytest.importorskip("rubiks.bench.santa2023")
    try:
        ins = santa.instances(4)[0]
    except Exception as e:
        pytest.skip(f"Santa 2023 data unavailable: {e}")
    P = cube(4)
    r = get_solver("m6_kmcoders", P, exact_orientation=True, anneal_iters=20000).solve(ins.state)
    assert P.is_identity(P.apply(ins.state, r.moves))
