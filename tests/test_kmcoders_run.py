"""KMCoders' own solver through our harness (skipped unless KMCODERS_DIR points at a built checkout), and
the rotation conjugation used by the symmetry ensemble."""
import os
from pathlib import Path

import numpy as np
import pytest

from rubiks.engine import cube, random_state
from rubiks.solvers import Budget, get_solver, load_all
from rubiks.solvers.symmetry import map_back, rotated

load_all()
KDIR = os.environ.get("KMCODERS_DIR", "")
built = KDIR and (Path(KDIR) / "solution/target/release/solve_cube").exists()


@pytest.mark.parametrize("N", [4, 5])
def test_rotated_copy_maps_back(N, rng):
    P = cube(N)
    s = random_state(P, rng)
    S = get_solver("m2_reduction", P, Budget(wall_time_s=60), exact_orientation=True)
    for r in (0, 7, 23, 24, 31, 47):
        s2 = rotated(P, s, r)
        m = S.solve(s2).moves
        assert np.array_equal(P.apply(s2, m), P.solved)
        assert np.array_equal(P.apply(s, map_back(P, m, r)), P.solved)


def test_input_matches_their_files():
    from rubiks.bench import santa2023
    from rubiks.solvers.external.kmcoders_run import santa_input

    if not built:
        pytest.skip("KMCODERS_DIR not set")
    for N in (4, 5):
        ins = santa2023.instances(N)[0]
        pid = int(ins.id.split("-")[-1])
        theirs = (Path(KDIR) / f"solution/all/in/{pid:04d}.txt").read_text()
        assert santa_input(cube(N), ins.state, pid) == theirs


@pytest.mark.skipif(not built, reason="KMCODERS_DIR not set")
def test_solves_random_4x4x4(rng):
    P = cube(4)
    s = random_state(P, rng)
    r = get_solver("kmcoders", P, Budget(wall_time_s=900)).solve(s)
    assert np.array_equal(P.apply(s, r.moves), P.solved)
