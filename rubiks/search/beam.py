"""Shared GPU beam search with tiered memory (technical basis, Section 9.3).

Used by Methods 4, 5A and 5B. One step:

1. Read the beam chunk by chunk from its store (GPU, pinned host memory, or disk).
2. Expand every allowed move (canonical-order pruning), hash the children, check for the
   goal, score them with the heuristic.
3. Keep only (score, hash, parent, move) per child — never the child state itself.
4. Merge into a global top-W with duplicate removal by hash.
5. Rebuild the W winning states from (parent, move) chunk by chunk into the next store.

So the GPU only ever holds one chunk of parents and its children; the beam itself can
live in host memory or on NVMe. States are bit-packed (3 bits per sticker for colours).
"""
from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import torch

from ..engine.moves import canonical_successor_mask
from ..engine.puzzle import Move, PuzzleSpec


# ------------------------------------------------------------------ packing
def pack3(states: torch.Tensor) -> torch.Tensor:
    """(B, S) uint8 values < 8 -> (B, ceil(S/8)*3) uint8, 3 bits per value."""
    B, S = states.shape
    pad = (-S) % 8
    x = torch.nn.functional.pad(states, (0, pad)).view(B, -1, 8).to(torch.int32)
    shifts = torch.arange(0, 24, 3, device=states.device, dtype=torch.int32)
    word = (x << shifts).sum(-1)  # (B, S/8) 24-bit words
    out = torch.stack([(word >> 0) & 255, (word >> 8) & 255, (word >> 16) & 255], -1)
    return out.view(B, -1).to(torch.uint8)


def unpack3(packed: torch.Tensor, S: int) -> torch.Tensor:
    B = packed.shape[0]
    b = packed.view(B, -1, 3).to(torch.int32)
    word = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    shifts = torch.arange(0, 24, 3, device=packed.device, dtype=torch.int32)
    vals = (word[..., None] >> shifts) & 7
    return vals.view(B, -1)[:, :S].to(torch.uint8)


# ------------------------------------------------------------------ stores
class BeamStore:
    """Holds the beam's states, packed. Backends: 'device', 'host' (pinned) or 'disk'."""

    def __init__(self, n: int, S: int, tier: str, device: torch.device, packed: bool, tmpdir: str | None = None):
        self.n, self.S, self.tier, self.device, self.packed = n, S, tier, device, packed
        width = math.ceil(S / 8) * 3 if packed else S
        self.width = width
        self.nbytes = n * width
        if tier == "device":
            self.buf = torch.empty((n, width), dtype=torch.uint8, device=device)
        elif tier == "host":
            pin = device.type == "cuda"
            self.buf = torch.empty((n, width), dtype=torch.uint8, pin_memory=pin)
        elif tier == "disk":
            fd, self.path = tempfile.mkstemp(suffix=".beam", dir=tmpdir)
            os.close(fd)
            self.buf = np.memmap(self.path, dtype=np.uint8, mode="w+", shape=(max(n, 1), width))
        else:
            raise ValueError(tier)

    def write(self, start: int, states: torch.Tensor) -> None:
        x = pack3(states) if self.packed else states
        if self.tier == "disk":
            self.buf[start : start + len(x)] = x.cpu().numpy()
        elif self.tier == "host":
            self.buf[start : start + len(x)].copy_(x, non_blocking=True)
        else:
            self.buf[start : start + len(x)] = x

    def read(self, start: int, end: int) -> torch.Tensor:
        if self.tier == "disk":
            x = torch.from_numpy(np.asarray(self.buf[start:end])).to(self.device)
        else:
            x = self.buf[start:end].to(self.device, non_blocking=True)
        return unpack3(x, self.S) if self.packed else x

    def gather(self, idx: torch.Tensor) -> torch.Tensor:
        """Rows by index (idx on any device) -> unpacked states on the compute device."""
        i = idx.cpu()
        if self.tier == "disk":
            x = torch.from_numpy(np.asarray(self.buf[i.numpy()])).to(self.device)
        elif self.tier == "host":
            x = self.buf[i].to(self.device, non_blocking=True)
        else:
            x = self.buf[idx.to(self.device)]
        return unpack3(x, self.S) if self.packed else x

    def close(self) -> None:
        if self.tier == "disk":
            del self.buf
            try:
                os.remove(self.path)
            except OSError:
                pass


# ------------------------------------------------------------------ search
class Heuristic(Protocol):
    def __call__(self, states: torch.Tensor) -> torch.Tensor:  # (B, S) uint8 -> (B,) float
        ...


@dataclass
class BeamConfig:
    width: int = 1024
    max_steps: int = 200
    device: str = "cpu"
    tier: str = "device"  # 'device' | 'host' | 'disk'
    packed: bool = True
    chunk: int = 4096  # max parents expanded per chunk
    max_chunk_mb: float = 64.0  # also cap a chunk so its children fit in this many MB
    score_batch: int = 16384  # children scored per heuristic call
    prune_canonical: bool = True
    visited_window: int = 3  # steps of hashes kept to avoid revisiting
    time_limit_s: float | None = None
    seed: int = 0


@dataclass
class BeamResult:
    moves: list[Move]
    found: bool
    steps: int
    expanded: int
    wall_time_s: float
    peak_gpu_mb: float = 0.0
    peak_host_mb: float = 0.0
    stats: dict = field(default_factory=dict)


class BeamSearch:
    def __init__(self, puzzle: PuzzleSpec, heuristic: Heuristic, config: BeamConfig | None = None,
                 move_subset: list[int] | None = None, actions: list[list[int]] | None = None):
        """``actions``: optional macro actions, each a list of move ids applied as one step
        (e.g. commutators). Single moves are always included unless ``move_subset`` limits them."""
        self.p = puzzle
        self.h = heuristic
        self.cfg = config or BeamConfig()
        self.dev = torch.device(self.cfg.device)
        mv = list(range(len(puzzle.moves))) if move_subset is None else list(move_subset)
        acts = [[m] for m in mv] + [list(a) for a in (actions or []) if len(a) > 1]
        self.actions = acts
        S = puzzle.num_stickers
        perms = []
        for a in acts:
            q = np.arange(S)
            for m in a:
                q = q[puzzle.move_perms[m]]
            perms.append(q)
        self.perms = torch.as_tensor(np.stack(perms), dtype=torch.long, device=self.dev)
        A, nb = len(acts), len(mv)
        allowed = np.ones((A + 1, A), dtype=bool)
        full_mask = canonical_successor_mask(puzzle)
        if self.cfg.prune_canonical:
            allowed[np.ix_(range(nb), range(nb))] = full_mask[np.ix_(mv, mv)]
            allowed[A, :nb] = True
        else:
            for i, a in enumerate(mv):
                for j, b in enumerate(mv):
                    ma, mb = puzzle.moves[a], puzzle.moves[b]
                    if ma.axis == mb.axis and ma.layer == mb.layer:
                        allowed[i, j] = False
        for i in range(nb, A):
            allowed[i, i] = False  # never the same macro twice in a row
        self.allowed = torch.as_tensor(allowed, device=self.dev)
        self.start_row = A
        g = torch.Generator().manual_seed(1234567)
        self.hash_w = torch.randint(1, 2**62, (S,), generator=g, dtype=torch.long).to(self.dev)
        self.packed = self.cfg.packed and puzzle.labeling == "color"
        # a heuristic with score_children(parents) -> (c, A) scores every child in one pass
        self.use_q = hasattr(heuristic, "score_children") and nb == A and move_subset is None
        per_parent = max(1, A * S)
        self.chunk = max(1, min(self.cfg.chunk, int(self.cfg.max_chunk_mb * 2**20 / per_parent)))

    def _hash(self, states: torch.Tensor) -> torch.Tensor:
        return (states.long() * self.hash_w).sum(1)

    def solve(self, start: np.ndarray, goal: Callable[[torch.Tensor], torch.Tensor] | None = None) -> BeamResult:
        """Search from ``start`` until ``goal(states) -> bool`` holds (default: puzzle solved)."""
        cfg, p, dev = self.cfg, self.p, self.dev
        t0 = time.time()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        S = p.num_stickers
        dtype = torch.uint8 if p.labeling == "color" else torch.int16
        if goal is None:
            goal = self.default_goal()
        s0 = torch.as_tensor(np.asarray(start), dtype=dtype, device=dev)[None]
        if bool(goal(s0)[0]):
            return BeamResult([], True, 0, 0, time.time() - t0)

        store = BeamStore(1, S, cfg.tier, dev, self.packed)
        store.write(0, s0)
        n = 1
        last = torch.full((1,), self.start_row, dtype=torch.long, device=dev)
        history: list[tuple[torch.Tensor, torch.Tensor]] = []
        recent: list[torch.Tensor] = [self._hash(s0)]
        expanded = 0
        peak_host = store.nbytes

        try:
            for step in range(1, cfg.max_steps + 1):
                if cfg.time_limit_s and time.time() - t0 > cfg.time_limit_s:
                    break
                visited = torch.cat(recent[-cfg.visited_window :])
                cand_score, cand_hash, cand_par, cand_mv = [], [], [], []
                found = None
                for a in range(0, n, self.chunk):
                    b = min(n, a + self.chunk)
                    par = store.read(a, b)  # (c, S)
                    q_chunk = self.h.score_children(par) if self.use_q else None  # (c, A)
                    kids = par[:, self.perms]  # (c, M, S)
                    mask = self.allowed[last[a:b]]  # (c, M)
                    pi, mi = mask.nonzero(as_tuple=True)
                    if len(pi) == 0:
                        continue
                    ks = kids[pi, mi]  # (k, S)
                    expanded += len(ks)
                    hs = self._hash(ks)
                    keep = ~torch.isin(hs, visited)
                    ks, hs, pi, mi = ks[keep], hs[keep], pi[keep], mi[keep]
                    if len(ks) == 0:
                        continue
                    g = goal(ks)
                    if bool(g.any()):
                        j = int(g.nonzero()[0, 0])
                        found = (int(pi[j]) + a, int(mi[j]))
                        break
                    if q_chunk is not None:  # Q*-style: children scored from the parents in one pass
                        sc = q_chunk[pi, mi].float()
                    else:
                        sc = torch.cat([self.h(ks[i : i + cfg.score_batch]).float()
                                        for i in range(0, len(ks), cfg.score_batch)])
                    # keep at most 2W per chunk before the global merge
                    if len(sc) > 2 * cfg.width:
                        top = torch.topk(sc, 2 * cfg.width, largest=False).indices
                        sc, hs, pi, mi = sc[top], hs[top], pi[top], mi[top]
                    cand_score.append(sc)
                    cand_hash.append(hs)
                    cand_par.append(pi + a)
                    cand_mv.append(mi)
                if found is not None:
                    path = [found[1]]
                    idx = found[0]
                    for par_idx, mv_idx in reversed(history):
                        path.append(int(mv_idx[idx]))
                        idx = int(par_idx[idx])
                    path.reverse()
                    moves = [p.moves[m] for i in path for m in self.actions[i]]
                    return self._result(moves, True, step, expanded, t0, peak_host)
                if not cand_score:
                    break
                sc = torch.cat(cand_score)
                hs = torch.cat(cand_hash)
                pa = torch.cat(cand_par)
                mv = torch.cat(cand_mv)
                # duplicate removal: keep the lowest score per hash
                order = torch.argsort(sc, stable=True)
                sc, hs, pa, mv = sc[order], hs[order], pa[order], mv[order]
                uh, inv = torch.unique(hs, return_inverse=True)
                first = torch.full((len(uh),), len(hs), dtype=torch.long, device=dev)
                first.scatter_reduce_(0, inv, torch.arange(len(hs), device=dev), reduce="amin")
                first, _ = torch.sort(first)
                sel = first[: cfg.width]
                pa, mv, hs = pa[sel], mv[sel], hs[sel]
                # rebuild the new beam from (parent, move)
                new_n = len(sel)
                new_store = BeamStore(new_n, S, cfg.tier, dev, self.packed)
                for a in range(0, new_n, self.chunk):
                    b = min(new_n, a + self.chunk)
                    par = store.gather(pa[a:b])
                    child = torch.gather(par, 1, self.perms[mv[a:b]])
                    new_store.write(a, child)
                peak_host = max(peak_host, store.nbytes + new_store.nbytes)
                store.close()
                store, n = new_store, new_n
                history.append((pa.cpu(), mv.cpu()))
                last = mv
                recent.append(hs)
            return self._result([], False, len(history), expanded, t0, peak_host)
        finally:
            store.close()

    def _result(self, moves, found, steps, expanded, t0, peak_host) -> BeamResult:
        gpu = torch.cuda.max_memory_allocated(self.dev) / 2**20 if self.dev.type == "cuda" else 0.0
        host = peak_host / 2**20 if self.cfg.tier != "device" else 0.0
        if self.cfg.tier == "device" and self.dev.type == "cpu":
            host = peak_host / 2**20
        return BeamResult(moves, found, steps, expanded, time.time() - t0, gpu, host)

    def default_goal(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Face-uniform goal: every face shows one colour."""
        slices = self.p.face_slices
        p = self.p
        face_t = torch.as_tensor(p.sticker_face, device=self.dev)

        def goal(states: torch.Tensor) -> torch.Tensor:
            c = face_t[states.long()] if p.labeling == "sticker" else states
            ok = torch.ones(len(states), dtype=torch.bool, device=states.device)
            for sl in slices:
                f = c[:, sl]
                ok &= (f == f[:, :1]).all(1)
            return ok

        return goal


def subset_goal(puzzle: PuzzleSpec, sticker_idx: np.ndarray, device: str = "cpu"):
    """Goal: the given stickers show their reference solved labels."""
    idx = torch.as_tensor(np.asarray(sticker_idx), dtype=torch.long, device=device)
    target = torch.as_tensor(puzzle.solved[np.asarray(sticker_idx)], device=device)

    def goal(states: torch.Tensor) -> torch.Tensor:
        return (states[:, idx] == target.to(states.dtype)).all(1)

    return goal
