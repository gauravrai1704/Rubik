"""Method 5A: tokenizer invariance, model forward / move head, save-load, staged mode."""
import numpy as np
import pytest
import torch

from rubiks.engine import cube, random_state, square
from rubiks.solvers.m5a_tokens.model import ClusterTokenModel, ModelConfig, Tokenizer, load, save, type_sizes_for


def test_type_keys_shared_across_sizes():
    keys = [set(Tokenizer(square(N)).type_keys) for N in (5, 9, 15)]
    assert keys[0] == keys[1] == keys[2]
    ck = [set(Tokenizer(cube(N)).type_keys) for N in (7, 9)]
    assert ck[0] == ck[1]


def test_forward_any_size_and_move_head(tmp_path):
    model = ClusterTokenModel(ModelConfig(d_model=32, layers=1, heads=2, ff=64, move_head=True),
                              type_sizes_for([square(5)]))
    for N in (5, 8, 13):
        P = square(N)
        tok = Tokenizer(P)
        x = torch.as_tensor(np.stack([random_state(P, np.random.default_rng(i), 50) for i in range(4)]))
        v, costs = model(x, tok)
        assert v.shape == (4,) and costs.shape == (4, tok.num_tokens) and (costs >= 0).all()
        assert model.move_values(x, tok).shape == (4, len(P.moves))
    path = tmp_path / "m.pt"
    save(model, path, {"x": 1})
    m2, meta = load(path)
    tok = Tokenizer(square(7))
    x = torch.as_tensor(square(7).solved)[None]
    assert torch.allclose(m2(x, tok)[0], model.eval()(x, tok)[0]) and meta == {"x": 1}


def test_unsupported_type_is_explained():
    model = ClusterTokenModel(ModelConfig(d_model=16, layers=1, heads=2, ff=32), type_sizes_for([cube(3)]))
    with pytest.raises(ValueError, match="train it on a size"):
        model.check_supported(Tokenizer(cube(6)))


def test_staged_mode_with_untrained_model_runs(tmp_path, rng):
    """Plumbing check: 5A's per-token costs drive 5B's stages (Section 8.5)."""
    from rubiks.solvers.m5a_tokens.solver import ClusterTokenSolver

    model = ClusterTokenModel(ModelConfig(d_model=16, layers=1, heads=2, ff=32), type_sizes_for([square(5)]))
    path = tmp_path / "sq.pt"
    save(model, path, {})
    P = square(4)
    S = ClusterTokenSolver(P, checkpoint=str(path), mode="staged", width=256, max_width=1024)
    s = random_state(P, rng)
    r = S.solve(s)
    assert "stages" in r.info
