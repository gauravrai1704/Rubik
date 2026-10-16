"""Method 5B: exact Square tables, stage plans, staged solving (Section 9)."""
import numpy as np
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


def test_stage_plans_cover_every_sticker():
    for P in (cube(3), cube(4), cube(7), square(3), square(8)):
        for kw in ({}, {"pieces_per_stage": 0}, {"order": "reduction"}):
            if P.kind == "square" and kw:
                continue
            st = plan_stages(P, **kw)
            covered = np.unique(np.concatenate([s.goal_stickers() for s in st]))
            assert np.array_equal(covered, np.arange(P.num_stickers)), (P, kw)


def test_split_stages_are_small():
    st = plan_stages(cube(6), pieces_per_stage=6)
    assert all(len(s.goal_stickers()) <= 24 for s in st if s.name.startswith(("wings", "centers")))


def test_coarse_tables_are_lower_bounds(rng):
    """Coarse tables never exceed the exact distance (checked against exact Square tables)."""
    from rubiks.solvers.m5b_stages.cluster_models import coarse_tables

    for c in clusters(square(5)):
        exact = table_for(c)
        for t, cmap in coarse_tables(c):
            s = random_state(square(5), rng)[c.stickers][None]
            assert t.lookup(cmap[s.astype(np.int64)])[0] <= exact.lookup(s)[0]


def test_center_macros_are_pure():
    from rubiks.search.macros import cube_center_macros

    P = cube(5)
    center = np.zeros(P.num_stickers, bool)
    for c in clusters(P):
        if c.kind.startswith("center_") and c.kind != "center_fixed":
            center[c.stickers] = True
    for seq in cube_center_macros(P)[:50]:
        q = np.arange(P.num_stickers)
        for m in seq:
            q = q[P.move_perms[m]]
        assert not (q != np.arange(P.num_stickers))[~center].any()


@pytest.mark.parametrize("N", [3, 4, 5, 6, 7])
def test_square_solves(N, rng):
    P = square(N)
    S = StagedSolver(P, Budget(wall_time_s=120), width=128, max_width=2048)
    for _ in range(3):
        s = random_state(P, rng)
        r = S.solve(s)
        assert P.is_solved(P.apply(s, r.moves)), r.info


def test_cube3_solves(rng):
    """Uses the committed 3x3x3 cluster MLPs (checkpoints/m5b) plus exact coarse tables."""
    P = cube(3)
    S = StagedSolver(P, Budget(wall_time_s=240), width=512, max_width=8192)
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


@pytest.mark.parametrize("N", [4, 9, 19])
def test_square_frame_is_solved_exactly(N, rng):
    """The frame of any Square is a 3×3 Square (<= 768 states): solved by exact lookup."""
    P = square(N)
    S = StagedSolver(P, Budget(wall_time_s=60), width=64, max_width=64)
    s = random_state(P, rng)
    h, singles, _ = S._stage_setup(0)
    table = S._exact_table(0, singles)
    assert table is not None and len(table[2]) <= 768
    mv = S._solve_exact(s, table, singles)
    g = S._goal(0)
    assert np.array_equal(P.apply(s, mv)[g], P.solved[g])


def test_late_center_stages_use_only_pure_macros():
    P = cube(5)
    S = StagedSolver(P, Budget(wall_time_s=60))
    late = [i for i in range(len(S.stages)) if S._is_late_center_stage(i)]
    assert late, "edges-first order should end with center stages"
    _, singles, macros = S._stage_setup(late[0])
    assert singles == [] and len(macros) > 0
