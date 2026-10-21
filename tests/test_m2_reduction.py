"""Method 2: orientation, piece identification, 3-cycle libraries, every stage and parity case."""
import numpy as np
import pytest

from rubiks.engine import Move, clusters, cube, random_state, square
from rubiks.solvers import Budget, get_solver
from rubiks.solvers.cluster_tools import CycleLibrary, greedy_solve, orient, parity, piece_permutation
from rubiks.solvers.m2_reduction.solver import wing_parity
from rubiks.solvers.three_stage import corner_parity_odd, solve_3x3_stage


@pytest.mark.parametrize("N", [3, 4, 5])
def test_orient_keeps_reachability_and_fixes_centers(N, rng):
    P = cube(N)
    s, inv = orient(P, random_state(P, rng))
    for c in clusters(P):
        if c.kind == "center_fixed":
            assert np.array_equal(s[c.stickers], P.solved[c.stickers])
    r = get_solver("m2_reduction", P).solve(s)  # still solvable after relabelling
    assert P.is_solved(P.apply(s, r.moves))


def test_piece_permutation_tracks_a_known_move():
    P = cube(4)
    wing = next(c for c in clusters(P) if c.kind == "wing")
    assert parity(piece_permutation(P, wing, P.solved)) == 0
    s = P.apply(P.solved, [Move(0, 1, 1)])  # inner slice quarter turn: one 4-cycle of wings
    assert parity(piece_permutation(P, wing, s)) == 1
    s = P.apply(P.solved, [Move(1, 3, 1)])  # outer face quarter turn: two 4-cycles
    assert parity(piece_permutation(P, wing, s)) == 0


@pytest.mark.parametrize("kind", ["wing", "center_x"])
def test_library_cycles_are_pure_and_complete(kind):
    P = cube(4)
    c = next(c for c in clusters(P) if c.kind == kind)
    lib = CycleLibrary(P, c)
    assert lib.complete and len(lib.cycles) == 24 * 23 * 22 // 3
    for seq in lib.seqs[:: max(1, len(lib.seqs) // 40)]:
        s = P.apply(P.solved, [P.moves[m] for m in seq])
        moved = np.nonzero(s != P.solved)[0]
        assert set(moved.tolist()) <= set(c.stickers.tolist())


def test_greedy_solves_one_center_orbit(rng):
    P = cube(5)
    s, _ = orient(P, random_state(P, rng))
    c = next(c for c in clusters(P) if c.kind == "center_x")
    ids, x = greedy_solve(CycleLibrary(P, c), s[c.stickers])
    after = P.apply(s, [P.moves[m] for m in ids])
    assert np.array_equal(after[c.stickers], c.solved)
    others = np.setdiff1d(np.arange(P.num_stickers), c.stickers)
    assert np.array_equal(after[others], s[others])


@pytest.mark.parametrize("N", [4, 5])
def test_wing_parity_case(N):
    P = cube(N)
    s = P.apply(P.solved, [Move(0, 1, 1)])  # odd wing orbit (OLL-parity source)
    assert len(wing_parity(P, s)) == 1
    r = get_solver("m2_reduction", P).solve(s)
    assert P.is_solved(P.apply(s, r.moves)) and r.info["stages"][0]["moves"] == 1


def test_pll_parity_case_even_n():
    P = cube(4)
    s = P.apply(P.solved, [Move(1, 3, 1)])  # odd corner permutation on an even cube
    assert corner_parity_odd(P, s)
    mv = solve_3x3_stage(P, s)
    t = P.apply(s, mv)
    corner = next(c for c in clusters(P) if c.kind == "corner")
    assert np.array_equal(t[corner.stickers], P.solved[corner.stickers])


@pytest.mark.parametrize("N", [3, 4, 5, 6])
def test_cube_random(N, rng):
    P = cube(N)
    S = get_solver("m2_reduction", P, Budget(wall_time_s=300))
    s = random_state(P, rng)
    r = S.solve(s)
    assert P.is_solved(P.apply(s, r.moves))
    assert [x["stage"] for x in r.info["stages"]][-1] == ("three" if N == 3 else "wings")


@pytest.mark.parametrize("N", [3, 4, 5, 6, 7, 9, 12])
def test_square_random(N, rng):
    P = square(N)
    S = get_solver("m2_reduction", P)
    for _ in range(3):
        s = random_state(P, rng)
        r = S.solve(s)
        assert P.is_solved(P.apply(s, r.moves))


@pytest.mark.slow
@pytest.mark.parametrize("N", [7, 8, 10])
def test_cube_random_larger(N, rng):
    P = cube(N)
    s = random_state(P, rng)
    r = get_solver("m2_reduction", P).solve(s)
    assert P.is_solved(P.apply(s, r.moves))
