"""Method 4 network, move sets and walk generator (technical basis, Section 7.1).

* :class:`ResMLP` — the CayleyPy network (Chervov et al., Fig. 1b): one-hot stickers,
  Linear(N1)+BN+ReLU, Linear(N2)+BN+ReLU, ``Nr`` residual blocks of two Linear(N2)+BN,
  one output predicting k, the number of random steps from solved. Nr = 4 gives the
  paper's "10-layer" model.
* :func:`move_set` — the generators the paper searches with: QTM on the cube (face quarter
  turns only for 3×3×3, every slice for N ≥ 4, the Santa 2023 "UQTM"), every flip on the
  Square (its moves are all 180°).
* :func:`walks` — non-backtracking random walks (never the inverse of the last move), the
  training data of Section 7.1 step 1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from ...engine.puzzle import PuzzleSpec

# Table II of the paper: (N1, N2, Nr). Every 10-layer model has Nr = 4.
PRESETS = {
    "paper3": (700, 643, 4),  # 3×3×3, 4M parameters (model No. 8)
    "paper4": (1010, 592, 4),  # 4×4×4, 4M (No. 16)
    "paper5": (1008, 560, 4),  # 5×5×5, 4M (No. 25)
    "1M": (430, 300, 4),  # 3×3×3, 1M (No. 4)
    "small": (256, 192, 2),  # CPU-sized, for the committed Square checkpoints
    "tiny": (64, 48, 1),  # tests
}

# Max walk length from the paper (Section IV B); other sizes in kmax_for().
PAPER_KMAX = {3: 26, 4: 45, 5: 65}


@dataclass
class ResMLPConfig:
    input_size: int
    n1: int = 700
    n2: int = 643
    nr: int = 4


class ResBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, d), nn.BatchNorm1d(d))

    def forward(self, x):
        return torch.relu(x + self.f(x))


class ResMLP(nn.Module):
    def __init__(self, cfg: ResMLPConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Sequential(
            nn.Linear(cfg.input_size, cfg.n1), nn.BatchNorm1d(cfg.n1), nn.ReLU(),
            nn.Linear(cfg.n1, cfg.n2), nn.BatchNorm1d(cfg.n2), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(cfg.n2) for _ in range(cfg.nr)])
        self.out = nn.Linear(cfg.n2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(self.stem(x))).squeeze(-1)


def one_hot(states: torch.Tensor, colors: int = 6, dtype=torch.float32) -> torch.Tensor:
    """(B, S) colour labels -> (B, S·colors) one-hot."""
    return nn.functional.one_hot(states.long(), colors).flatten(1).to(dtype)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------ move sets
def move_set(puzzle: PuzzleSpec, name: str = "paper") -> list[int]:
    """Move ids the method trains and searches with.

    ``paper``: cube 3×3×3 = 12 face quarter turns (DeepCubeA's QTM); cube N ≥ 4 = every
    slice's quarter turns (Santa 2023 UQTM); Square = every flip. ``all``: every engine move
    (adds half turns and, for N = 3, middle slices)."""
    mv = puzzle.moves
    if name == "all" or puzzle.kind != "cube":
        return list(range(len(mv)))
    if name != "paper":
        raise ValueError(f"unknown move set {name!r}; use 'paper' or 'all'")
    outer = {0, puzzle.N - 1}
    return [i for i, m in enumerate(mv) if m.turn in (1, 3) and (puzzle.N > 3 or m.layer in outer)]


def successor_mask(puzzle: PuzzleSpec, subset: list[int]) -> np.ndarray:
    """(A+1, A) bool, allowed[last, next] over ``subset`` (row A = no previous move).

    Forbids the inverse of the last move (non-backtracking) and, for commuting moves on one
    axis, every order but ascending layer. Repeating a quarter turn stays legal: in QTM
    ``R R`` is how ``R2`` is spelled."""
    mv = [puzzle.moves[i] for i in subset]
    A = len(mv)
    allowed = np.ones((A + 1, A), dtype=bool)
    for i, a in enumerate(mv):
        for j, b in enumerate(mv):
            if a.axis != b.axis:
                continue
            if b.layer < a.layer or (b.layer == a.layer and (a.turn + b.turn) % 4 == 0):
                allowed[i, j] = False
    return allowed


def kmax_for(puzzle: PuzzleSpec) -> int:
    """Random-walk horizon: the paper's 26 / 45 / 65 for cubes 3-5; Squares and larger cubes
    follow the 5A rule (~ a few times N²/ln N) so every method sees comparable depths."""
    if puzzle.kind == "cube" and puzzle.N in PAPER_KMAX:
        return PAPER_KMAX[puzzle.N]
    from ..m5a_tokens.train import kmax_for as kmax_5a

    return kmax_5a(puzzle)


class WalkGenerator:
    """Fresh (state, k) pairs from non-backtracking random walks, all on ``device``."""

    def __init__(self, puzzle: PuzzleSpec, subset: list[int], kmax: int, device="cpu"):
        self.dev = torch.device(device)
        self.perms = torch.as_tensor(puzzle.move_perms[subset], dtype=torch.long, device=self.dev)
        self.solved = torch.as_tensor(puzzle.solved, device=self.dev)
        inv = {m: i for i, m in enumerate(puzzle.moves[j] for j in subset)}
        self.inverse = torch.as_tensor([inv.get(puzzle.moves[j].inverse(), -1) for j in subset], device=self.dev)
        self.kmax = kmax

    def walks(self, n_walks: int, gen: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``n_walks`` walks of length kmax -> states (n_walks·kmax, S), steps k (float)."""
        A, S = self.perms.shape
        st = self.solved.unsqueeze(0).expand(n_walks, S).clone()
        last = None
        xs, ks = [], []
        ar = torch.arange(n_walks, device=self.dev)
        for k in range(1, self.kmax + 1):
            keys = torch.rand((n_walks, A), device=self.dev, generator=gen)
            if last is not None:
                bad = self.inverse[last]
                keys[ar[bad >= 0], bad[bad >= 0]] = -1.0
            mv = keys.argmax(1)
            st = torch.gather(st, 1, self.perms[mv])
            last = mv
            xs.append(st)
            ks.append(torch.full((n_walks,), float(k), device=self.dev))
        return torch.cat(xs), torch.cat(ks)


# ------------------------------------------------------------------ save / load
def save(model: ResMLP, path, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "cfg": asdict(model.cfg), "meta": meta}, path)


def load(path, device: str = "cpu") -> tuple[ResMLP, dict]:
    ck = torch.load(path, map_location=device, weights_only=True)  # never unpickle code from a checkpoint
    model = ResMLP(ResMLPConfig(**ck["cfg"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck.get("meta", {})
