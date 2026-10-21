"""Method 3: exact block covers, Lemma 4 bulk flips, verified cube bulk commutators, solving."""
import numpy as np
import pytest

from rubiks.engine import Move, clusters, cube, metrics, random_state, square
from rubiks.solvers import get_solver
from rubiks.solvers.m3_demaine.batching import TypeBank, blocks, center_templates


def test_blocks_cover_exactly(rng):
    for _ in range(20):
        S = {(int(c), int(r)) for c, r in rng.integers(0, 9, size=(30, 2))}
        bl = blocks(S)
        cells = [(x, y) for X, Y in bl for x in X for y in Y]
        assert sorted(cells) == sorted(S) and len(cells) == len(set(cells))


def test_lemma4_bulk_equals_single_commutators():
    """C_X R_Y C_X R_Y changes exactly the clusters in X × Y, each by its own C R C R."""
    P = square(10)
    idx = P.move_index
    X, Y = [1, 3], [2, 4]
    C = [idx[Move(0, x, 2)] for x in X]
    R = [idx[Move(1, y, 2)] for y in Y]
    bulk = P.apply(P.solved, [P.moves[m] for m in C + R + C + R])
    single = P.solved
    for x in X:
        for y in Y:
            single = P.apply(single, [Move(0, x, 2), Move(1, y, 2), Move(0, x, 2), Move(1, y, 2)])
    assert np.array_equal(bulk, single)


def test_cube_bulk_commutator_hits_several_orbits():
    """One center type with layer sets I, J: each orbit (i, j) gets its own pure 3-cycle."""
    P = cube(8)
    cl = [c for c in clusters(P) if c.kind.startswith("center_") and c.kind != "center_fixed"]
    bank = TypeBank(P, cl, center_templates(P)[:4], setup_depth=0)
    t = bank.ops[cl[0].index][0].type_id
    seq = bank.sequence(t, ({1, 2}, {5, 6}))
    singles = P.solved
    for i in (1, 2):
        for j in (5, 6):
            singles = P.apply(singles, [P.moves[m] for m in bank.sequence(t, ({i}, {j}))])
    bulk = P.apply(P.solved, [P.moves[m] for m in seq])
    assert np.array_equal(bulk, singles)
    assert len(seq) < 4 * len(bank.sequence(t, ({1}, {5})))


@pytest.mark.parametrize("N", [3, 4, 5, 6, 8, 11, 16])
def test_square_random_and_not_longer_than_m2(N, rng):
    P = square(N)
    m3, m2 = get_solver("m3_demaine", P), get_solver("m2_reduction", P)
    for _ in range(2):
        s = random_state(P, rng)
        r3, r2 = m3.solve(s), m2.solve(s)
        assert P.is_solved(P.apply(s, r3.moves))
        assert metrics(r3.moves, P)["htm"] <= metrics(r2.moves, P)["htm"]


@pytest.mark.parametrize("N", [3, 4, 5, 6])
def test_cube_random(N, rng):
    P = cube(N)
    s = random_state(P, rng)
    r = get_solver("m3_demaine", P).solve(s)
    assert P.is_solved(P.apply(s, r.moves))


@pytest.mark.slow
@pytest.mark.parametrize("N", [7, 9])
def test_cube_random_larger(N, rng):
    P = cube(N)
    s = random_state(P, rng)
    r = get_solver("m3_demaine", P).solve(s)
    assert P.is_solved(P.apply(s, r.moves))
