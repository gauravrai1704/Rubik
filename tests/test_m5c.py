"""Method 5C: net-gain block choice, verified bulk steps, solving, never worse than Method 3 on the Square."""
import numpy as np
import pytest

from rubiks.engine import cube, metrics, random_state, square
from rubiks.solvers import get_solver
from rubiks.solvers.m5c_schedule.scheduler import _alternate


def test_alternate_takes_a_block_with_one_bad_cell():
    """A block whose cells mostly help is taken whole, though one cell costs a little."""
    G = np.zeros((6, 6))
    G[np.ix_([1, 2, 3], [4, 5])] = 2.0
    G[3, 5] = -1.0  # one harmful cell inside the block
    valid = np.ones((6, 6), dtype=bool)
    X, Y = _alternate(G, valid, lam=0.25, base=4, seeds=[1])
    assert sorted(X) == [1, 2, 3] and sorted(Y) == [4, 5]


def test_alternate_respects_invalid_cells_and_disjointness():
    G = np.ones((5, 5)) * 3
    valid = np.ones((5, 5), dtype=bool)
    valid[2, 3] = False
    X, Y = _alternate(G, valid, lam=0.25, base=4, seeds=[0])
    assert not set(X) & set(Y)
    assert valid[np.ix_(X, Y)].all()


@pytest.mark.parametrize("N", [4, 5, 6])
def test_cube_random(N, rng):
    P = cube(N)
    S = get_solver("m5c_schedule", P, lookahead=0)
    s = random_state(P, rng)
    r = S.solve(s)
    assert P.is_solved(P.apply(s, r.moves))
    assert r.info["schedule"]["centers"]["rounds"] > 0


@pytest.mark.parametrize("N", [3, 6, 9, 12])
def test_square_never_longer_than_method3(N, rng):
    P = square(N)
    m5c, m3 = get_solver("m5c_schedule", P), get_solver("m3_demaine", P)
    for _ in range(2):
        s = random_state(P, rng)
        a, b = m5c.solve(s), m3.solve(s)
        assert P.is_solved(P.apply(s, a.moves))
        assert metrics(a.moves, P)["htm"] <= metrics(b.moves, P)["htm"]


@pytest.mark.slow
@pytest.mark.parametrize("N", [8, 10])
def test_cube_shorter_than_method3(N, rng):
    P = cube(N)
    s = random_state(P, rng)
    a = get_solver("m5c_schedule", P).solve(s)
    b = get_solver("m3_demaine", P).solve(s)
    assert P.is_solved(P.apply(s, a.moves))
    assert metrics(a.moves, P)["htm"] < metrics(b.moves, P)["htm"]
