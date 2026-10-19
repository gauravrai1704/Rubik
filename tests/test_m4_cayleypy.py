"""Method 4: move sets, non-backtracking walks, ResMLP shape, save-load, solver plumbing."""
import numpy as np
import pytest
import torch

from rubiks.engine import cube, metrics, random_state, scramble, square
from rubiks.solvers import Budget, get_solver
from rubiks.solvers.m4_cayleypy.model import (PRESETS, ResMLP, ResMLPConfig, WalkGenerator, load, move_set, num_params,
                                              successor_mask)
from rubiks.solvers.m4_cayleypy.train import train


def test_paper_move_sets():
    P3 = cube(3)
    s3 = move_set(P3)
    assert len(s3) == 12 and all(P3.moves[i].turn in (1, 3) and P3.moves[i].layer in (0, 2) for i in s3)
    assert len(move_set(cube(4))) == 24 and len(move_set(cube(5))) == 30  # every slice, quarter turns
    assert move_set(square(5)) == list(range(len(square(5).moves)))
    assert len(move_set(P3, "all")) == len(P3.moves)


def test_successor_mask_is_qtm_non_backtracking():
    P = cube(3)
    sub = move_set(P)
    mv = [P.moves[i] for i in sub]
    A = successor_mask(P, sub)
    for i, a in enumerate(mv):
        assert A[i, mv.index(a)]  # R R (= R2 in QTM) allowed
        assert not A[i, mv.index(a.inverse())]  # R R' never
    assert A[len(mv)].all()


def test_paper_4m_presets():
    for name, N in (("paper3", 3), ("paper4", 4), ("paper5", 5)):
        n1, n2, nr = PRESETS[name]
        m = ResMLP(ResMLPConfig(6 * cube(N).num_stickers, n1, n2, nr))
        assert 3.5e6 < num_params(m) < 4.7e6


def test_walks_never_backtrack():
    P = cube(3)
    sub = move_set(P)
    g = WalkGenerator(P, sub, kmax=10)
    X, K = g.walks(64, torch.Generator().manual_seed(0))
    assert X.shape == (640, P.num_stickers) and K.min() == 1 and K.max() == 10
    # step 2 is never back at solved: the inverse of step 1 is excluded
    solved = torch.as_tensor(P.solved)
    assert not (X[64:128] == solved).all(1).any()


def test_train_save_load_and_solve(tmp_path, rng):
    P = square(3)
    path = tmp_path / "square3.pt"
    model, meta = train("square", 3, preset="tiny", epochs=3, epoch_size=4000, out=path, verbose=False)
    assert meta["train_time_s"] >= 0 and "peak_gpu_mb" in meta and len(meta["log"]) == 3
    m2, meta2 = load(path)
    x = torch.nn.functional.one_hot(torch.as_tensor(P.solved).long(), 6).flatten()[None].float()
    assert torch.allclose(m2(x), model.cpu()(x)) and meta2["N"] == 3
    S = get_solver("m4_cayleypy", P, Budget(wall_time_s=60), checkpoint=[str(path), str(path)], width=256)
    s = P.apply(P.solved, scramble(P, 6, rng))
    r = S.solve(s)
    assert P.is_solved(P.apply(s, r.moves)) and len(r.info["runs"]) == 2


def test_wrong_size_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "s3.pt"
    train("square", 3, preset="tiny", epochs=1, epoch_size=500, out=path, verbose=False)
    with pytest.raises(ValueError, match="trained on square3"):
        get_solver("m4_cayleypy", square(4), checkpoint=str(path))


def test_cube3_qtm_search_solves_short_scramble(tmp_path, rng):
    """Even a barely trained net finds depth-3 QTM solutions with a wide beam; R2 = two quarter turns."""
    P = cube(3)
    path = tmp_path / "cube3.pt"
    train("cube", 3, preset="tiny", epochs=2, epoch_size=4000, out=path, verbose=False)
    S = get_solver("m4_cayleypy", P, Budget(wall_time_s=60), checkpoint=str(path), width=4096)
    sub = [P.moves[i] for i in move_set(P)]
    seq = [sub[int(i)] for i in rng.integers(0, len(sub), 3)]
    s = P.apply(P.solved, seq)
    r = S.solve(s)
    assert P.is_solved(P.apply(s, r.moves))
    assert all(m.layer in (0, 2) for m in r.moves)


@pytest.mark.parametrize("N", [3, 4, 5, 6, 7, 8])
def test_committed_square_checkpoints(N):
    P = square(N)
    S = get_solver("m4_cayleypy", P, Budget(wall_time_s=120), width=1024, max_width=8192)
    s = random_state(P, np.random.default_rng(N))
    r = S.solve(s)
    assert P.is_solved(P.apply(s, r.moves)), r.info
    assert metrics(r.moves, P)["htm"] == len(r.moves)
