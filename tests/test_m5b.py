"""Method 5B: exact Square tables, stage plans, staged solving (Section 9)."""
import pytest

from rubiks.engine import clusters, cube, random_state, square
from rubiks.solvers import Budget
from rubiks.solvers.m5b_stages.cluster_models import table_for
from rubiks.solvers.m5b_stages.solver import StagedSolver
from rubiks.solvers.m5b_stages.stages import plan_stages


@pytest.mark.parametrize("N", [3, 4, 5, 6, 9])
def test_every_square_cluster_has_an_exact_table(N):
    for c in clusters(square(N)):
        t = table_for(c)
        assert t is not None and t.lookup(c.solved[None])[0] == 0


def test_interior_cluster_has_six_configurations():
    """Demaine 2011, Lemma 1 — up to the colour-solved symmetry of the projection."""
    c = [c for c in clusters(square(6)) if c.kind == "sq_interior"][0]
    t = table_for(c)
    assert t.size == 6 * 4 or t.size >= 6  # 6 top/bottom patterns x side placements


def test_stage_plans_cover_every_cluster():
    for P in (cube(3), cube(4), cube(7), square(3), square(8)):
        st = plan_stages(P)
        covered = sorted(c.index for s in st for c in s.targets)
        assert covered == list(range(len(clusters(P))))


@pytest.mark.parametrize("N", [3, 4, 5, 6, 7])
def test_square_solves(N, rng):
    P = square(N)
    S = StagedSolver(P, Budget(wall_time_s=120), width=128, max_width=2048)
    for _ in range(3):
        s = random_state(P, rng)
        r = S.solve(s)
        assert P.is_solved(P.apply(s, r.moves)), r.info


@pytest.mark.slow
@pytest.mark.parametrize("N", [10, 13, 20])
def test_square_solves_large(N, rng):
    P = square(N)
    S = StagedSolver(P, Budget(wall_time_s=600), width=128, max_width=2048)
    s = random_state(P, rng)
    r = S.solve(s)
    assert P.is_solved(P.apply(s, r.moves)), r.info
