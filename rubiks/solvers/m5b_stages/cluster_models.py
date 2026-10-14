"""Per-cluster heuristics for Method 5B (technical basis, Section 9.2).

Looking only at one cluster's stickers gives a small *projected* puzzle. Two kinds of
heuristic estimate its distance to solved:

* :class:`TableModel` — exact BFS distances, used whenever the projected puzzle has at most
  ``TABLE_CAP`` states (every Rubik's Square cluster, 3×3×3 fixed centers, ...).
* :class:`ClusterMLP` — a small residual MLP trained on random walks of the projected
  puzzle (corners, middle edges, wings, cube centers).

Both are keyed by the cluster's *signature*: clusters of the same type at different
layers or different N are the same projected puzzle, so one model covers every N.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ...engine.orbits import Orbit
from ...engine.projection import ProjectionTable, cached_table, can_encode
from ...engine.puzzle import PuzzleSpec
from ...engine.random_states import random_walks

TABLE_CAP = 1_500_000
CKPT_DIR = Path(__file__).resolve().parents[3] / "checkpoints" / "m5b"

# Random-walk length used to train each cluster type (diffusion distance horizon).
DEFAULT_KMAX = {
    "corner": 30, "midge": 30, "wing": 40, "center_x": 30, "center_plus": 30,
    "center_oblique": 30, "center_fixed": 10,
}


class TableModel:
    """Exact distance lookup. Unreachable codes get ``max_dist + 1`` (never happens for
    states produced by moves)."""

    def __init__(self, table: ProjectionTable, device: str = "cpu"):
        self.table = table
        self.device = torch.device(device)
        self.codes = torch.as_tensor(table.codes, device=self.device)
        self.dist = torch.as_tensor(table.dist.astype(np.float32), device=self.device)
        w = [1]
        for _ in range(table.K - 1):
            w.append(w[-1] * table.base)
        self.weights = torch.as_tensor(np.array(w[::-1], dtype=np.int64), device=self.device)
        self.miss = float(table.max_dist + 1)
        self.kind = "table"

    def to(self, device):
        return TableModel(self.table, device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        c = (x.long() * self.weights).sum(-1)
        i = torch.searchsorted(self.codes, c).clamp_(max=len(self.codes) - 1)
        hit = self.codes[i] == c
        return torch.where(hit, self.dist[i], torch.full_like(self.dist[i], self.miss))


class ClusterMLP(nn.Module):
    """One-hot cluster colours -> estimated distance (ResMLP, as in CayleyPy, smaller)."""

    def __init__(self, K: int, colors: int = 6, hidden: int = 512, blocks: int = 2):
        super().__init__()
        self.K, self.colors = K, colors
        self.inp = nn.Sequential(nn.Linear(K * colors, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden))
            for _ in range(blocks)
        )
        self.out = nn.Linear(hidden, 1)
        self.kind = "mlp"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.one_hot(x.long(), self.colors).float().flatten(1)
        h = self.inp(h)
        for b in self.blocks:
            h = torch.relu(h + b(h))
        return self.out(h).squeeze(-1).clamp_min(0.0)


# Projected puzzles known to be far larger than TABLE_CAP: skip the BFS attempt.
LARGE_KINDS = {"corner", "midge", "wing", "center_x", "center_plus", "center_oblique"}


def table_for(cluster: Orbit) -> ProjectionTable | None:
    if cluster.kind in LARGE_KINDS or not can_encode(cluster.size, 6):
        return None
    return cached_table((cluster.signature,), cluster.gen_perms, cluster.solved.astype(np.int64), cap=TABLE_CAP)


def _local_axis_layer(puzzle: PuzzleSpec, cluster: Orbit) -> torch.Tensor:
    mv = [puzzle.moves[int(m)] for m in cluster.gen_moves]
    return torch.tensor([[m.axis, m.layer] for m in mv], dtype=torch.long)


def train_cluster_mlp(
    puzzle: PuzzleSpec,
    cluster: Orbit,
    steps: int = 3000,
    batch: int = 4096,
    walks: int = 512,
    kmax: int | None = None,
    lr: float = 1e-3,
    device: str = "cpu",
    hidden: int = 512,
    blocks: int = 2,
    seed: int = 0,
    log_every: int = 250,
) -> tuple[ClusterMLP, dict]:
    """Train on fresh non-backtracking random walks of the projected puzzle every step."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    kmax = kmax or DEFAULT_KMAX.get(cluster.kind, 30)
    perms = torch.as_tensor(cluster.gen_perms, dtype=torch.long, device=dev)
    solved = torch.as_tensor(cluster.solved.astype(np.int64), device=dev)
    al = _local_axis_layer(puzzle, cluster).to(dev)
    model = ClusterMLP(cluster.size, 6, hidden, blocks).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator(device=dev).manual_seed(seed)
    log = []
    t0 = time.time()
    for it in range(1, steps + 1):
        xs, ys = [], []
        for st, k in random_walks(perms, solved, walks, kmax, gen, al):
            xs.append(st)
            ys.append(torch.full((walks,), float(k), device=dev))
        X = torch.cat(xs)
        Y = torch.cat(ys)
        # include the solved state so the model learns h(solved) = 0
        X = torch.cat([X, solved[None].expand(64, -1)])
        Y = torch.cat([Y, torch.zeros(64, device=dev)])
        idx = torch.randint(0, len(X), (batch,), device=dev, generator=gen)
        pred = model(X[idx])
        loss = nn.functional.mse_loss(pred, Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % log_every == 0 or it == steps:
            log.append({"step": it, "loss": loss.item(), "time_s": round(time.time() - t0, 1)})
            print(f"  [{cluster.kind} K={cluster.size}] step {it}/{steps} loss {loss.item():.3f}")
    model.eval()
    meta = {"kind": cluster.kind, "K": cluster.size, "kmax": kmax, "hidden": hidden, "blocks": blocks,
            "steps": steps, "trained_on": repr(puzzle), "signature": cluster.signature, "log": log}
    return model, meta


def save_mlp(model: ClusterMLP, meta: dict, root: Path = CKPT_DIR) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{meta['kind']}_{meta['signature']}.pt"
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    (root / f"{meta['kind']}_{meta['signature']}.json").write_text(json.dumps(meta, indent=1))
    return path


def load_mlp(signature: str, kind: str, device: str = "cpu", root: Path = CKPT_DIR) -> ClusterMLP | None:
    path = root / f"{kind}_{signature}.pt"
    if not path.exists():
        return None
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ck["meta"]
    model = ClusterMLP(m["K"], 6, m["hidden"], m["blocks"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


class ModelBank:
    """Gives every cluster of a puzzle its heuristic, shared by signature."""

    def __init__(self, device: str = "cpu", root: Path = CKPT_DIR, allow_missing: bool = False):
        self.device, self.root, self.allow_missing = device, root, allow_missing
        self._cache: dict[str, object] = {}

    def get(self, cluster: Orbit):
        sig = cluster.signature
        if sig in self._cache:
            return self._cache[sig]
        t = table_for(cluster)
        if t is not None:
            model = TableModel(t, self.device)
        else:
            model = load_mlp(sig, cluster.kind, self.device, self.root)
            if model is None:
                if not self.allow_missing:
                    raise FileNotFoundError(
                        f"no trained model for {cluster.kind} cluster (signature {sig}). Train it with:\n"
                        f"  python -m rubiks.solvers.m5b_stages.train --puzzle cube --N <N>")
                model = None
        self._cache[sig] = model
        return model

    def needs_training(self, cluster: Orbit) -> bool:
        return table_for(cluster) is None and load_mlp(cluster.signature, cluster.kind, "cpu", self.root) is None
