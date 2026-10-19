"""Methods 1-4 (teammates). Each owner removes the xfail for their method when it works."""
import pytest

from rubiks.engine import random_state, square
from rubiks.solvers import REGISTRY, Budget, get_solver, load_all

METHODS = ["m1_idastar", "m2_reduction", "m3_demaine", "m4_cayleypy"]
DONE = {"m4_cayleypy"}


def test_all_registered():
    load_all()
    for m in METHODS + ["m5a_tokens", "m5b_stages"]:
        assert m in REGISTRY


@pytest.mark.parametrize("name", [m if m in DONE else
                                  pytest.param(m, marks=pytest.mark.xfail(raises=NotImplementedError, strict=True))
                                  for m in METHODS])
def test_solves_small_square(name, rng):
    P = square(4)
    solver = get_solver(name, P, Budget(wall_time_s=60))
    s = random_state(P, rng)
    r = solver.solve(s)
    assert P.is_solved(P.apply(s, r.moves))
