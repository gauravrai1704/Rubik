"""Methods 1-4 on the shared interface: every method registered, every method solves a 4×4 Square."""
import pytest

from rubiks.engine import random_state, square
from rubiks.solvers import REGISTRY, Budget, get_solver, load_all

METHODS = ["m1_idastar", "m2_reduction", "m3_demaine", "m4_cayleypy"]


def test_all_registered():
    load_all()
    for m in METHODS + ["m5a_tokens", "m5b_stages"]:
        assert m in REGISTRY


@pytest.mark.parametrize("name", METHODS)
def test_solves_small_square(name, rng):
    P = square(4)
    solver = get_solver(name, P, Budget(wall_time_s=60))
    s = random_state(P, rng)
    r = solver.solve(s)
    assert P.is_solved(P.apply(s, r.moves))
