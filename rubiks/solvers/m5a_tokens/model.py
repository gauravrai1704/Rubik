"""Method 5A model: one network for every N (technical basis, Section 8).

The puzzle is read as a *set of cluster tokens* (cubie clusters, see engine.orbits):

* content: one-hot colours of the cluster's stickers in canonical, N-invariant order,
  projected by a linear map shared by every cluster of the same type (kind, size);
* type: learned embedding of the cluster kind (corner, wing, center_x, sq_interior, ...);
* position: the cluster's depth descriptor (|in-face coordinates| / (N-1)) and N / 20.

A transformer encoder (no positional order: the tokens are a set) mixes them.

* **Additive value head**: each token predicts its own cost; costs are summed and a small
  global correction added. Distance grows with the number of clusters, so a sum extends
  naturally to larger N. Per-token costs also serve as per-cluster heuristics inside
  Method 5B's stages (Section 8.5).
* **Move head (Q* style, optional)**: one token per move (axis, layer/(N-1), turn)
  cross-attends to the cluster tokens and predicts the distance after that move, so all
  moves are scored in one forward pass (Agostinelli et al., Q* search).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ...engine.orbits import ALL_KINDS, clusters
from ...engine.puzzle import PuzzleSpec


def distance_scale(N: int) -> float:
    """N²/ln N, the growth rate of God's number (Demaine et al. 2011); used to normalise."""
    return max(1.0, N * N / math.log(max(N, 2)))


class Tokenizer:
    """Precomputed sticker indices and features of a puzzle's cluster tokens."""

    def __init__(self, puzzle: PuzzleSpec, device: str = "cpu"):
        self.puzzle = puzzle
        dev = torch.device(device)
        cl = clusters(puzzle)
        self.clusters = cl
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(cl):
            groups.setdefault(f"{c.kind}:{c.size}", []).append(i)
        self.type_keys = list(groups)
        self.group_idx = [torch.as_tensor(np.stack([cl[i].stickers for i in groups[k]]), device=dev)
                          for k in self.type_keys]
        self.group_tokens = [groups[k] for k in self.type_keys]  # cluster indices, in token order
        order = [i for k in self.type_keys for i in groups[k]]
        self.token_cluster = order  # token t -> cluster index
        self.cluster_token = {ci: t for t, ci in enumerate(order)}
        self.kind_id = torch.as_tensor([ALL_KINDS.index(cl[i].kind) for i in order], device=dev)
        n = puzzle.N / 20.0
        self.geo = torch.as_tensor([[cl[i].depth[0], cl[i].depth[1], n] for i in order], dtype=torch.float32,
                                   device=dev)
        mv = puzzle.moves
        d = np.array(puzzle.dims)
        self.move_axis = torch.as_tensor([m.axis for m in mv], device=dev)
        self.move_turn = torch.as_tensor([m.turn for m in mv], device=dev)
        self.move_layer = torch.as_tensor([m.layer / max(d[m.axis] - 1, 1) for m in mv], dtype=torch.float32,
                                          device=dev)
        self.scale = distance_scale(puzzle.N)
        self.num_tokens = len(order)


@dataclass
class ModelConfig:
    d_model: int = 192
    layers: int = 4
    heads: int = 6
    ff: int = 512
    dropout: float = 0.0
    colors: int = 6
    move_head: bool = False


class ClusterTokenModel(nn.Module):
    def __init__(self, cfg: ModelConfig, type_sizes: dict[str, int]):
        """``type_sizes``: every cluster type the model may see, ``"kind:size" -> size``."""
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.type_sizes = dict(type_sizes)
        self.proj = nn.ModuleDict({k.replace(":", "_"): nn.Linear(K * cfg.colors, d) for k, K in type_sizes.items()})
        self.kind_emb = nn.Embedding(len(ALL_KINDS), d)
        self.geo = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, cfg.heads, cfg.ff, cfg.dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.token_cost = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.global_corr = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        if cfg.move_head:
            self.axis_emb = nn.Embedding(3, d)
            self.turn_emb = nn.Embedding(4, d)
            self.layer_mlp = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
            self.move_attn = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
            self.move_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def check_supported(self, tok: Tokenizer) -> None:
        missing = [k for k in tok.type_keys if k not in self.type_sizes]
        if missing:
            raise ValueError(f"model was not built for cluster types {missing}; train it on a size that has them "
                             f"(cube N>=7 covers every cube type, square N>=5 every square type)")

    def encode(self, states: torch.Tensor, tok: Tokenizer) -> torch.Tensor:
        parts = []
        for key, idx in zip(tok.type_keys, tok.group_idx):
            x = states[:, idx]  # (B, n_t, K)
            x = nn.functional.one_hot(x.long(), self.cfg.colors).float().flatten(2)
            parts.append(self.proj[key.replace(":", "_")](x))
        h = torch.cat(parts, 1)
        h = h + self.kind_emb(tok.kind_id)[None] + self.geo(tok.geo)[None]
        return self.norm(self.encoder(h))

    def token_costs(self, h: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(self.token_cost(h).squeeze(-1))  # (B, T) >= 0

    def forward(self, states: torch.Tensor, tok: Tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (predicted distance (B,), per-token costs (B, T))."""
        h = self.encode(states, tok)
        costs = self.token_costs(h)
        corr = self.global_corr(h.mean(1)).squeeze(-1)
        value = costs.sum(1) + corr * tok.scale * 0.1
        return value.clamp_min(0.0), costs

    def move_values(self, states: torch.Tensor, tok: Tokenizer, h: torch.Tensor | None = None) -> torch.Tensor:
        """Predicted distance after each move, (B, M), in one pass."""
        if not self.cfg.move_head:
            raise RuntimeError("model built without a move head")
        if h is None:
            h = self.encode(states, tok)
        layer_feat = torch.stack([tok.move_layer, torch.full_like(tok.move_layer, tok.puzzle.N / 20.0)], -1)
        q = self.axis_emb(tok.move_axis) + self.turn_emb(tok.move_turn) + self.layer_mlp(layer_feat)
        q = q[None].expand(h.shape[0], -1, -1)
        a, _ = self.move_attn(q, h, h)
        return nn.functional.softplus(self.move_out(a + q).squeeze(-1)) * tok.scale


def type_sizes_for(puzzles: list[PuzzleSpec]) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in puzzles:
        for c in clusters(p):
            out[f"{c.kind}:{c.size}"] = c.size
    return out


def save(model: ClusterTokenModel, path, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "cfg": model.cfg.__dict__, "type_sizes": model.type_sizes,
                "meta": meta}, path)


def load(path, device: str = "cpu") -> tuple[ClusterTokenModel, dict]:
    ck = torch.load(path, map_location=device, weights_only=True)  # never unpickle code from a checkpoint
    model = ClusterTokenModel(ModelConfig(**ck["cfg"]), ck["type_sizes"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck.get("meta", {})
