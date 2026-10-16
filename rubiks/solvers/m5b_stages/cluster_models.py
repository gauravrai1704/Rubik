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


class CoarseTableModel:
    """Exact distances of a *coarsened* projection: colours mapped to a few classes (e.g.
    "U or D" vs "anything else"). Admissible lower bound; tiny tables even when the full
    projection is far too large (orientation of corners / middle edges, center layouts)."""

    def __init__(self, table: ProjectionTable, color_map: np.ndarray, device: str = "cpu"):
        self.inner = TableModel(table, device)
        self.color_map = torch.as_tensor(color_map, device=self.inner.device)
        self.kind = "coarse"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(self.color_map[x.long()])


class MaxModel:
    """max over several heuristics (each a lower bound or an estimate)."""

    def __init__(self, models):
        self.models = [m for m in models if m is not None]
        self.kind = "+".join(m.kind for m in self.models)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = self.models[0](x).float()
        for m in self.models[1:]:
            out = torch.maximum(out, m(x).float())
        return out


# opposite-colour pairs: U/D, R/L, F/B
COARSE_PAIRS = ((0, 3), (1, 4), (2, 5))
COARSE_CAP = 1_500_000
COARSE_SKIP = {"wing"}  # 48 stickers: every coarse projection is far above the cap
TABLE_DIR = CKPT_DIR / "tables"
_COARSE_CACHE: dict[tuple, list] = {}


def _disk_table(name: str, build):
    """Load a ProjectionTable from checkpoints/m5b/tables/<name>.npz, or build and save it."""
    path = TABLE_DIR / f"{name}.npz"
    if path.exists():
        z = np.load(path)
        return ProjectionTable(codes=z["codes"], dist=z["dist"], base=int(z["base"]), K=int(z["K"]))
    t = build()
    if t is not None:
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, codes=t.codes, dist=t.dist, base=t.base, K=t.K)
    return t


# kinds where the finer 3-class projection (a -> 1, b -> 2, rest -> 0) is small enough:
# corners: C(8,4) * 3^7 = 153,090 states (which corners are U / D corners, plus twist)
THREE_CLASS_KINDS = {"corner"}


def coarse_tables(cluster: Orbit) -> list[tuple[ProjectionTable, np.ndarray]]:
    """For each opposite-colour pair, BFS the projection with that pair -> 1, others -> 0
    (or, for ``THREE_CLASS_KINDS``, a -> 1, b -> 2, others -> 0)."""
    key = (cluster.signature,)
    if key in _COARSE_CACHE:
        return _COARSE_CACHE[key]
    out = []
    if cluster.kind in COARSE_SKIP:
        _COARSE_CACHE[key] = out
        return out
    for a, b in COARSE_PAIRS:
        three = cluster.kind in THREE_CLASS_KINDS
        cmap = np.zeros(6, dtype=np.int64)
        cmap[a], cmap[b] = 1, (2 if three else 1)
        solved = cmap[cluster.solved.astype(np.int64)]
        if solved.min() == solved.max():
            continue
        tag = "c3" if three else "coarse"
        t = _disk_table(f"{cluster.kind}_{cluster.signature}_{tag}{a}{b}",
                        lambda: cached_table((cluster.signature, tag, a, b), cluster.gen_perms, solved,
                                             base=3 if three else 2, cap=COARSE_CAP))
        if t is not None:
            out.append((t, cmap))
    _COARSE_CACHE[key] = out
    return out


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

    def __init__(self, device: str = "cpu", root: Path = CKPT_DIR, allow_missing: bool = False,
                 coarse: bool = True, tables_only: tuple[str, ...] = ()):
        """``coarse``: combine each MLP with exact coarse tables by max (ablation: off).
        ``tables_only``: cluster kinds that use their coarse tables alone, without the MLP."""
        self.device, self.root, self.allow_missing, self.coarse = device, root, allow_missing, coarse
        self.tables_only = set(tables_only)
        self._cache: dict[str, object] = {}

    def get(self, cluster: Orbit):
        sig = cluster.signature
        if sig in self._cache:
            return self._cache[sig]
        t = table_for(cluster)
        if t is not None:
            model = TableModel(t, self.device)
        else:
            mlp = None if cluster.kind in self.tables_only else load_mlp(sig, cluster.kind, self.device, self.root)
            if mlp is None and not self.allow_missing and cluster.kind not in self.tables_only:
                raise FileNotFoundError(
                    f"no trained model for {cluster.kind} cluster (signature {sig}). Train it with:\n"
                    f"  python -m rubiks.solvers.m5b_stages.train --puzzle cube --N <N>")
            coarse = [CoarseTableModel(tb, cm, self.device) for tb, cm in coarse_tables(cluster)] if self.coarse else []
            parts = ([mlp] if mlp is not None else []) + coarse
            model = MaxModel(parts) if len(parts) > 1 else (parts[0] if parts else None)
        self._cache[sig] = model
        return model

    def needs_training(self, cluster: Orbit) -> bool:
        return table_for(cluster) is None and load_mlp(cluster.signature, cluster.kind, "cpu", self.root) is None
