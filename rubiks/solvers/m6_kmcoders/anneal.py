"""Simulated annealing over the move sequence (KMCoders' step 2, cube).

After the special pieces are solved, the remaining clusters (wings, centers) are only touched by inner
slices that are not the middle slice. Those moves can be inserted, changed, swapped or removed anywhere in
the sequence, even between the special-part face turns, without disturbing the special pieces. Annealing
over such edits with the objective

    length (QTM) + alpha * (stickers of wing / center clusters not at their solved colour)

brings the puzzle to a roughly solved state with cheap single moves before any commutator is used.

Every neighbour replaces a window a_t .. a_{t+k-1} of the sequence by another short window W', which
changes the final state F into F[C] with C = inv[t+k] ∘ (P_W^-1 ∘ P_W') ∘ suf[t+k], so a neighbour is
scored on the few stickers it touches. Wing parity is kept: every edit keeps, for each wing orbit, the
number of quarter turns of its slices mod 2.
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np

from ...engine.puzzle import PuzzleSpec


@nb.njit(cache=True)
def _rebuild(moves, perms, invperms, suf, inv, upto):
    """Recompute suf[tau], inv[tau] for tau = upto-1 .. 0 from suf[upto]."""
    for tau in range(upto - 1, -1, -1):
        m = moves[tau]
        p = perms[m]
        ip = invperms[m]
        nxt = suf[tau + 1]
        nxi = inv[tau + 1]
        row = suf[tau]
        irow = inv[tau]
        for x in range(row.shape[0]):
            row[x] = p[nxt[x]]
            irow[x] = nxi[ip[x]]


def _compose(perms, seq, S):
    q = np.arange(S, dtype=np.int64)
    for m in seq:
        q = q[perms[m]]
    return q


class Annealer:
    def __init__(self, P: PuzzleSpec, start: np.ndarray, moves: list[int], target: np.ndarray, alpha: float = 1.5,
                 seed: int = 0, weights: np.ndarray | None = None):
        """``weights``: per-sticker cost of being wrong (default 1 everywhere); the objective is
        length + alpha * sum of the weights of wrong stickers."""
        self.P = P
        N = P.N
        self.S = P.num_stickers
        self.perms = P.move_perms.astype(np.int32)
        self.invperms = np.argsort(self.perms, axis=1).astype(np.int32)
        self.start = np.asarray(start)
        self.target = np.asarray(target)
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        mid = N // 2 if N % 2 else -1
        # free moves: inner-slice quarter turns, middle slice excluded
        self.free = [i for i, m in enumerate(P.moves) if 0 < m.layer < N - 1 and m.layer != mid and m.turn in (1, 3)]
        self.free_set = set(self.free)
        # parity class: wing orbit index min(layer, N-1-layer)
        self.cls = {i: min(P.moves[i].layer, N - 1 - P.moves[i].layer) for i in self.free}
        self.by_cls: dict[int, list[int]] = {}
        for i, c in self.cls.items():
            self.by_cls.setdefault(c, []).append(i)
        self.w = np.ones(self.S) if weights is None else np.asarray(weights, dtype=np.float64)
        self.moves = np.array(moves, dtype=np.int64)
        self._full_rebuild()

    # ------------------------------------------------------------------ bookkeeping
    def _full_rebuild(self):
        T = len(self.moves)
        self.suf = np.empty((T + 1, self.S), dtype=np.int32)
        self.inv = np.empty_like(self.suf)
        self.suf[T] = np.arange(self.S)
        self.inv[T] = np.arange(self.S)
        _rebuild(self.moves.astype(np.int64), self.perms, self.invperms, self.suf, self.inv, T)
        self.F = self.start[self.suf[0]]
        self.wrong = float(self.w[self.F != self.target].sum())

    def _replace(self, t: int, k: int, new: list[int]):
        """Replace moves[t:t+k] by ``new`` and update suffix arrays for times <= t + len(new)."""
        L = len(new)
        self.moves = np.concatenate([self.moves[:t], np.array(new, dtype=np.int64), self.moves[t + k:]])
        if L != k:
            T = len(self.moves)
            suf = np.empty((T + 1, self.S), dtype=np.int32)
            inv = np.empty_like(suf)
            suf[t + L:] = self.suf[t + k:]
            inv[t + L:] = self.inv[t + k:]
            self.suf, self.inv = suf, inv
        _rebuild(self.moves, self.perms, self.invperms, self.suf, self.inv, t + L)

    def delta(self, t: int, k: int, new: list[int]):
        """(change in wrong stickers, positions, new values) if moves[t:t+k] became ``new``."""
        old = self.moves[t:t + k]
        D = _compose(self.perms, list(old), self.S)
        Dn = _compose(self.perms, new, self.S)
        Dinv = np.argsort(D)
        E = Dinv[Dn]  # P_W^-1 ∘ P_W'
        moved = np.nonzero(E != np.arange(self.S))[0]
        if len(moved) == 0:
            return 0, moved, None
        sufk, invk = self.suf[t + k], self.inv[t + k]
        u = invk[moved]
        vals = self.F[invk[E[moved]]]
        d = int((vals != self.target[u]).sum() - (self.F[u] != self.target[u]).sum())
        return d, u, vals

    def delta2(self, edits):
        """Exact effect of edits at up to two times. ``edits``: [(t, k, new)] with increasing t and
        non-overlapping windows. Returns (change in wrong stickers, new final state)."""
        S = self.S
        # new suffix permutation from the first edit time t0, built right to left in O(S) per edit
        (t0, k0, n0) = edits[0]
        cur = None  # new suffix starting at the current edit time
        nxt_time = None
        for t, k, new in reversed(edits):
            if cur is None:
                rest = self.suf[t + k]
            else:  # the old moves from t + k up to the next edit time, then the new suffix there
                seg = self.suf[t + k][self.inv[nxt_time]]  # Q(a_{t+k} .. a_{nxt-1})
                rest = seg[cur]
            w = np.arange(S, dtype=np.int64)
            for m in new:
                w = w[self.perms[m]]
            cur = w[rest]
            nxt_time = t
        newfinal = self.F[self.inv[t0][cur]]
        d = float(self.w[newfinal != self.target].sum()) - self.wrong
        return d, newfinal

    def _apply_edits(self, edits, newfinal, d):
        for t, k, new in reversed(edits):  # right to left keeps earlier indices valid
            self._replace(t, k, new)
        self.F = newfinal
        self.wrong += d

    # ------------------------------------------------------------------ neighbours
    def propose2(self):
        """Two-time edits (KMCoders' neighbourhood): insert, remove or move moves anywhere."""
        T = len(self.moves)
        free_pos = np.nonzero(np.isin(self.moves, self.free))[0]
        r = self.rng.random()
        if r < 0.3:  # insert two moves of one parity class at two times
            c = int(self.rng.choice(list(self.by_cls)))
            t1, t2 = sorted(int(x) for x in self.rng.integers(0, T + 1, 2))
            a, b = int(self.rng.choice(self.by_cls[c])), int(self.rng.choice(self.by_cls[c]))
            if t1 == t2:
                return [(t1, 0, [a, b])]
            return [(t1, 0, [a]), (t2, 0, [b])]
        if r < 0.5 and len(free_pos) >= 2:  # remove two free moves of one parity class
            i, j = sorted(int(x) for x in self.rng.choice(free_pos, 2, replace=False))
            if self.cls[int(self.moves[i])] != self.cls[int(self.moves[j])]:
                return None
            return [(i, 1, []), (j, 1, [])] if j > i + 0 else None
        if r < 0.7 and len(free_pos) >= 1:  # move one free move to another time
            i = int(self.rng.choice(free_pos))
            j = int(self.rng.integers(0, T + 1))
            m = int(self.moves[i])
            if j == i or j == i + 1:
                return None
            return [(j, 0, [m]), (i, 1, [])] if j < i else [(i, 1, []), (j, 0, [m])]
        if r < 0.85 and len(free_pos) >= 1:  # change one free move within its class
            i = int(self.rng.choice(free_pos))
            c = self.cls[int(self.moves[i])]
            return [(i, 1, [int(self.rng.choice(self.by_cls[c]))])]
        if T >= 2:  # swap two adjacent moves (one of them free)
            t = int(self.rng.integers(0, T - 1))
            a, b = int(self.moves[t]), int(self.moves[t + 1])
            if a in self.free_set or b in self.free_set:
                return [(t, 2, [b, a])]
        return None

    def run2(self, iters: int, T0: float = 2.0, T1: float = 0.05, deadline: float | None = None) -> dict:
        import time

        accepted = 0
        for it in range(iters):
            if deadline is not None and it % 256 == 0 and time.time() > deadline:
                break
            edits = self.propose2()
            if not edits:
                continue
            d_wrong, newfinal = self.delta2(edits)
            d_len = sum(len(n) - k for _, k, n in edits)
            d_obj = d_len + self.alpha * d_wrong
            temp = T0 * (T1 / T0) ** (it / max(1, iters - 1))
            if d_obj <= 0 or self.rng.random() < math.exp(-d_obj / temp):
                self._apply_edits(edits, newfinal, d_wrong)
                accepted += 1
        return {"accepted": accepted, "length": len(self.moves), "wrong": self.wrong}

    def propose(self):
        T = len(self.moves)
        r = self.rng.random()
        free_pos = [i for i in range(T) if int(self.moves[i]) in self.free_set] if r < 0.75 else None
        if r < 0.3 and free_pos:  # change one free move to another of the same parity class
            t = int(self.rng.choice(free_pos))
            c = self.cls[int(self.moves[t])]
            return t, 1, [int(self.rng.choice(self.by_cls[c]))]
        if r < 0.55 and T >= 2:  # swap two adjacent moves (at least one free)
            t = int(self.rng.integers(0, T - 1))
            a, b = int(self.moves[t]), int(self.moves[t + 1])
            if a in self.free_set or b in self.free_set:
                return t, 2, [b, a]
            return None
        if r < 0.75 and free_pos:  # remove a free move and an adjacent free move of the same class
            t = int(self.rng.choice(free_pos))
            if t + 1 < T and int(self.moves[t + 1]) in self.free_set and \
                    self.cls[int(self.moves[t + 1])] == self.cls[int(self.moves[t])]:
                return t, 2, []
            return None
        # insert two moves of the same parity class at time t
        t = int(self.rng.integers(0, T + 1))
        c = int(self.rng.choice(list(self.by_cls)))
        return t, 0, [int(self.rng.choice(self.by_cls[c])), int(self.rng.choice(self.by_cls[c]))]

    def run(self, iters: int, T0: float = 2.0, T1: float = 0.05, deadline: float | None = None) -> dict:
        import time

        accepted = 0
        for it in range(iters):
            if deadline is not None and it % 256 == 0 and time.time() > deadline:
                break
            prop = self.propose()
            if prop is None:
                continue
            t, k, new = prop
            if k == len(new) and list(self.moves[t:t + k]) == new:
                continue
            d_wrong, u, vals = self.delta(t, k, new)
            d_obj = (len(new) - k) + self.alpha * d_wrong
            temp = T0 * (T1 / T0) ** (it / max(1, iters - 1))
            if d_obj <= 0 or self.rng.random() < math.exp(-d_obj / temp):
                if vals is not None:
                    self.F[u] = vals
                self.wrong += d_wrong
                self._replace(t, k, new)
                accepted += 1
        return {"accepted": accepted, "length": len(self.moves), "wrong": self.wrong}
