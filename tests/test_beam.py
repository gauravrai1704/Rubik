"""Shared beam search: packing, tiers, macros, Q-style scoring (Section 9.3)."""
import numpy as np
import pytest
import torch

from rubiks.engine import cube, scramble, square
from rubiks.search.beam import BeamConfig, BeamSearch, pack3, subset_goal, unpack3
from rubiks.search.macros import square_macros


def test_pack_roundtrip():
    for S in (1, 7, 8, 54, 2400):
        x = torch.randint(0, 6, (9, S), dtype=torch.uint8)
        assert torch.equal(unpack3(pack3(x), S), x)


def misplaced(P):
    solved = torch.as_tensor(P.solved)
    return lambda st: (st != solved).sum(1).float()


@pytest.mark.parametrize("tier", ["device", "host", "disk"])
def test_tiers_solve_short_scramble(tier, rng):
    P = cube(3)
    s = P.apply(P.solved, scramble(P, 4, rng))
    r = BeamSearch(P, misplaced(P), BeamConfig(width=3000, max_steps=8, tier=tier)).solve(s)
    assert r.found and P.is_solved(P.apply(s, r.moves))


def test_macros_expand_to_moves(rng):
    P = square(6)
    s = P.apply(P.solved, scramble(P, 6, rng))
    r = BeamSearch(P, misplaced(P), BeamConfig(width=2000, max_steps=10), actions=square_macros(P)).solve(s)
    assert r.found and P.is_solved(P.apply(s, r.moves))


def test_subset_goal():
    P = square(4)
    g = subset_goal(P, np.arange(5))
    st = torch.as_tensor(P.solved)[None]
    assert bool(g(st)[0])


def test_q_scoring_path(rng):
    P = cube(3)
    solved = torch.as_tensor(P.solved)
    perms = torch.as_tensor(P.move_perms)

    class Q:
        def __call__(self, st):
            return (st != solved).sum(1).float()

        def score_children(self, par):
            kids = par[:, perms]
            return (kids != solved).sum(-1).float()

    s = P.apply(P.solved, scramble(P, 3, rng))
    bs = BeamSearch(P, Q(), BeamConfig(width=2000, max_steps=6))
    assert bs.use_q
    r = bs.solve(s)
    assert r.found and P.is_solved(P.apply(s, r.moves))
