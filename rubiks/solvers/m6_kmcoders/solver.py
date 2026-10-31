"""Method 6: our implementation of the KMCoders Kaggle Santa 2023 cube method.

KMCoders (Santa 2023; github.com/wata-orz/santa2023_permutation_puzzle) solve an N×N×N cube by
(1) solving the "special" pieces (corners; middle edges and fixed centers for odd N) as a 2×2×2 / 3×3×3,
(2) making every remaining cluster an even permutation, (3) applying short commutators, and (4) finishing
every cluster with 3-cycles ("3-rots"), where (3) and (4) *insert* each sequence at whichever time of
the current solution makes it cheapest: a sequence inserted at time t acts on the final state through
everything after t (``sequence.py``), and can cancel with the moves around it. Moves are counted in QTM
(Santa's metric: every quarter turn of every slice is one move).

Our implementation:

1. ``special``: Kociemba's two-phase solver on corners, middle edges and fixed centers (outer layers and,
   for odd N, the exact-orientation middle-slice turns); KMCoders used an optimal 3×3×3 solver.
2. ``parity``: one inner-slice quarter turn per wing orbit with an odd permutation (center clusters have
   repeated colours, so their parity can always be absorbed).
3. ``commutators``: greedy insertion of the 4-move commutators [X_i, Y_j] = X_i Y_j X_i' Y_j' of two inner
   slices on different axes (each performs two center 3-cycles), choosing at every step the (commutator,
   time) pair with the most stickers fixed per net quarter turn, over sampled times.
4. ``three_cycles``: greedy insertion of 3-cycles from each cluster's library (``cluster_tools``) at their
   cheapest time, best stickers fixed per net quarter turn first, until every cluster is solved.

Not (yet) included from KMCoders: simulated annealing over elementary moves before step 3, the 6-move
extended commutators, the exact minimum-3-rot count by cycle decomposition, and beam search in step 4.
"""
from __future__ import annotations

import time

import numpy as np

from ...engine import PuzzleSpec, canonicalize
from ...engine.orbits import clusters
from ...engine.puzzle import Move
from ..base import Budget, Solver, SolveResult, register
from ..cluster_tools import CycleLibrary, Slots, parity, piece_permutation, prepare
from ..m2_reduction.solver import CENTER_KINDS
from ..three_stage import solve_3x3_stage
from .sequence import InsertionSequence, quarter_turns


def commutator_ops(P: PuzzleSpec, with_faces: bool = True) -> dict:
    """4-move commutators that never touch the special pieces: [X_i, Y_j] with X, Y inner-slice quarter
    turns on different axes (two center 3-cycles), and, with ``with_faces``, [X_i, F] and [F, X_i] with F
    an outer-face quarter turn on another axis (X never moves corners or middle edges and F is undone, so
    they 3-cycle a whole column of wing and center pieces at once). Middle slice excluded."""
    N, idx, inv = P.N, P.move_index, P.inverse_move
    inner = [k for k in range(1, N - 1) if not (N % 2 and k == N // 2)]
    seqs = []
    if with_faces:
        for a in range(3):
            for b in range(3):
                if a == b:
                    continue
                for i in inner:
                    for fl in (0, N - 1):
                        for ta in (1, 3):
                            for tb in (1, 3):
                                X, F = idx[Move(a, i, ta)], idx[Move(b, fl, tb)]
                                seqs.append([X, F, int(inv[X]), int(inv[F])])
                                seqs.append([F, X, int(inv[F]), int(inv[X])])
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            for i in inner:
                for j in inner:
                    for ta in (1, 3):
                        for tb in (1, 3):
                            X, Y = idx[Move(a, i, ta)], idx[Move(b, j, tb)]
                            seqs.append([X, Y, int(inv[X]), int(inv[Y])])
    S = P.num_stickers
    sup, src, keep = [], [], []
    for s in seqs:
        q = np.arange(S)
        for m in s:
            q = q[P.move_perms[m]]
        moved = np.nonzero(q != np.arange(S))[0]
        if len(moved) == 0:
            continue
        pos = {int(x): k for k, x in enumerate(moved)}
        sup.append(moved)
        src.append([pos[int(q[x])] for x in moved])  # new[moved[k]] = old[moved[src[k]]]
        keep.append(s)
    width = max(len(x) for x in sup)
    M = np.full((len(sup), width), -1, dtype=np.int64)
    SRC = np.tile(np.arange(width), (len(sup), 1))
    for k, (m, s) in enumerate(zip(sup, src)):
        M[k, :len(m)] = m
        SRC[k, :len(s)] = s
    return {"seqs": np.array(keep), "support": M, "src": SRC, "valid": M >= 0}


class CycleTable:
    """A cluster's 3-cycle library as dense arrays keyed by (slot a, slot b, slot c) with
    new[a] = old[b], new[b] = old[c], new[c] = old[a] (canonical rotation: smallest slot first)."""

    def __init__(self, P: PuzzleSpec, c):
        self.slots = Slots(P, c)
        lib = CycleLibrary(P, c, metric="qtm")
        n = self.slots.n
        self.n = n
        size = n ** 3
        self.cost = np.full(size, 10**6, dtype=np.int64)
        self.first = np.zeros(size, dtype=np.int64)
        self.last = np.zeros(size, dtype=np.int64)
        self.seq: dict[int, list[int]] = {}
        # sticker (global) -> slot
        self.slot_of = {int(c.stickers[k]): int(self.slots.sticker_slot[k]) for k in range(len(c.stickers))}
        self.rep = np.array([int(c.stickers[v[0]]) for v in self.slots.slot_stickers])  # one sticker per slot
        for seq, lp in zip(lib.seqs, lib.local):
            moved = np.nonzero(lp != np.arange(len(lp)))[0]
            slots = sorted(set(self.slots.sticker_slot[moved].tolist()))
            if len(slots) != 3:
                continue
            src = {}
            for s in slots:
                k = int(self.slots.slot_stickers[s][0])
                src[s] = int(self.slots.sticker_slot[lp[k]])
            a = slots[0]
            key = self.key(a, src[a], src[src[a]])
            q = quarter_turns(P, seq)
            if len(q) < self.cost[key]:
                self.cost[key] = len(q)
                self.first[key], self.last[key] = q[0], q[-1]
                self.seq[key] = q

    def key(self, a, b, c):
        a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
        # canonical rotation: smallest first
        r1 = (a < b) & (a < c)
        r2 = (b < a) & (b < c)
        A = np.where(r1, a, np.where(r2, b, c))
        B = np.where(r1, b, np.where(r2, c, a))
        C = np.where(r1, c, np.where(r2, a, b))
        return (A * self.n + B) * self.n + C


@register("m6_kmcoders")
class KMCodersStyleSolver(Solver):
    def __init__(self, puzzle: PuzzleSpec, budget: Budget | None = None, exact_orientation: bool = False,
                 sample_times: int = 256, seed: int = 0, phase2_min_ratio: float = 0.4,
                 anneal_iters: int | None = None, alpha: float = 2.5, face_commutators: bool = True,
                 wing_weight: float = 1.0, center_weight: float = 1.0, **kwargs):
        super().__init__(puzzle, budget)
        if puzzle.kind != "cube":
            raise ValueError("Method 6 is for cubes (the KMCoders cube method)")
        self.exact = exact_orientation
        self.sample_times = sample_times
        self.rng = np.random.default_rng(seed)
        self.phase2_min_ratio = phase2_min_ratio
        self.anneal_iters = anneal_iters if anneal_iters is not None else 6000 * puzzle.N ** 2
        self.alpha = alpha
        self.weights = np.zeros(puzzle.num_stickers)
        for c in clusters(puzzle):
            if c.kind == "wing":
                self.weights[c.stickers] = wing_weight
            elif c.kind in CENTER_KINDS:
                self.weights[c.stickers] = center_weight
        self.seed = seed
        P = puzzle
        self.cl = [c for c in clusters(P) if c.kind in CENTER_KINDS + ("wing",)]
        self.ops = commutator_ops(P, face_commutators) if P.N >= 4 else None
        self._tables: dict[int, CycleTable] = {}

    def table(self, c) -> CycleTable:
        if c.index not in self._tables:
            self._tables[c.index] = CycleTable(self.puzzle, c)
        return self._tables[c.index]

    # ------------------------------------------------------------------ phases
    def _times(self, seq: InsertionSequence) -> np.ndarray:
        T = seq.T
        if T + 1 <= self.sample_times:
            return np.arange(T + 1)
        return np.unique(np.concatenate([[0, T], self.rng.choice(T + 1, self.sample_times - 2, replace=False)]))

    def _phase_commutators(self, seq: InsertionSequence, target: np.ndarray, deadline: float) -> int:
        ops = self.ops
        if ops is None or len(self.cl) == 0:
            return 0
        sup, src, valid = ops["support"], ops["src"], ops["valid"]
        first, last = ops["seqs"][:, 0], ops["seqs"][:, -1]
        length = np.full(len(first), 4)
        rounds = 0
        while time.time() < deadline:
            F = seq.final()
            ts = self._times(seq)
            best = None
            for t in ts:
                X = seq.inv[t][np.where(valid, sup, 0)]  # final positions acted on, (ops, w)
                vals = F[X]
                tgt = target[X]
                new = np.take_along_axis(vals, src, 1)
                gain = (((new == tgt) & valid).sum(1) - ((vals == tgt) & valid).sum(1)).astype(np.float64)
                cost = seq.cancel_cost(int(t), first, last, length)
                ratio = np.where(gain > 0, gain / np.maximum(cost, 1), -1)
                k = int(np.argmax(ratio))
                if best is None or ratio[k] > best[0]:
                    best = (float(ratio[k]), int(t), k)
            if best is None or best[0] < self.phase2_min_ratio:
                break
            _, t, k = best
            seq.insert(t, [int(x) for x in ops["seqs"][k]])
            rounds += 1
        return rounds

    def _phase_three_cycles(self, seq: InsertionSequence, target: np.ndarray, deadline: float,
                            candidates: int = 24) -> int:
        rounds = 0
        while time.time() < deadline:
            F = seq.final()
            best = None
            for c in self.cl:
                tab = self.table(c)
                rep = tab.rep
                ok = np.array([bool(np.all(F[c.stickers[v]] == target[c.stickers[v]]))
                               for v in tab.slots.slot_stickers])
                if ok.all():
                    continue
                cand = self._candidate_cycles(c, tab, F, target, ok, candidates)
                for (a, b, cc), gain in cand:
                    # final-coordinate cycle new[a]=old[b]... at time t: slots of suf[t][rep]
                    ya = seq.suf[:, rep[a]]
                    yb = seq.suf[:, rep[b]]
                    yc = seq.suf[:, rep[cc]]
                    sa, sb, sc = tab_slot(tab, ya), tab_slot(tab, yb), tab_slot(tab, yc)
                    keys = tab.key(sa, sb, sc)
                    cost = tab.cost[keys].copy()
                    has = cost < 10**6
                    if not has.any():
                        continue
                    # cancellations at each time
                    prev = np.array([-1] + seq.moves)
                    nxt = np.array(seq.moves + [-1])
                    inv = seq.inverse_move
                    cost = cost - 2 * (inv[tab.first[keys]] == prev) - 2 * (inv[tab.last[keys]] == nxt)
                    cost = np.where(has, cost, 10**6)
                    t = int(np.argmin(cost))
                    ratio = gain / max(1, int(cost[t]))
                    if best is None or ratio > best[0]:
                        best = (ratio, t, tab, int(keys[t]))
            if best is None:
                break
            _, t, tab, key = best
            seq.insert(t, tab.seq[key])
            rounds += 1
        return rounds

    def _candidate_cycles(self, c, tab: CycleTable, F, target, ok, k: int):
        """Final-coordinate 3-cycles (slots a, b, c: new[a] = old[b] ...) that fix the most slots."""
        n = tab.n
        slot_st = tab.slots.slot_stickers
        stick = c.stickers
        cur = [tuple(F[stick[v]]) for v in slot_st]
        want = [tuple(target[stick[v]]) for v in slot_st]
        wrong = [j for j in range(n) if not ok[j]]
        out = []
        for a in wrong:
            for b in range(n):
                if b == a or cur[b] != want[a]:
                    continue
                for cc in range(n):
                    if cc in (a, b):
                        continue
                    g = (cur[b] == want[a]) + (cur[cc] == want[b]) + (cur[a] == want[cc]) \
                        - ok[a] - ok[b] - ok[cc]
                    if g > 0:
                        out.append(((a, b, cc), int(g)))
        out.sort(key=lambda x: -x[1])
        return out[:k]

    # ------------------------------------------------------------------ solve
    def solve(self, state: np.ndarray) -> SolveResult:
        t0 = time.time()
        P = self.puzzle
        deadline = t0 + self.budget.wall_time_s
        s, pre = prepare(P, state, self.exact)
        if P.N % 2 == 0:  # even N: the special pieces are the corners, solved optimally in QTM
            from .corners import corner_solution_qtm

            special = corner_solution_qtm(P, s)
        else:  # odd N: corners, middle edges and fixed centers as a 3x3x3 (Kociemba)
            special = solve_3x3_stage(P, s)
        s1 = P.apply(s, special)
        par = []
        for c in self.cl:
            if c.kind == "wing" and parity(piece_permutation(P, c, s1)):
                k = min(int(P.sticker_layer[x, a]) for x in c.stickers for a in range(3)
                        if 0 < P.sticker_layer[x, a] < P.N - 1)
                par.append(Move(0, k, 1))
        start_moves = quarter_turns(P, pre + special + par)
        if self.exact:  # pre turns the fixed centers home; the target is the exact solved state
            base_state = np.asarray(state)
        else:  # relabelled colours: the moves solve the original up to a whole-cube rotation
            base_state, start_moves = s, quarter_turns(P, special + par)
        target = P.solved
        info = {"special": len(quarter_turns(P, special)), "parity": len(par)}
        if P.N >= 4 and self.anneal_iters:
            from .anneal import Annealer

            A = Annealer(P, base_state, start_moves, target, alpha=self.alpha, seed=self.seed, weights=self.weights)
            info["anneal"] = A.run2(self.anneal_iters, deadline=t0 + 0.5 * self.budget.wall_time_s)
            start_moves = [int(m) for m in A.moves]
        seq = InsertionSequence(P, base_state, start_moves)
        info["commutators"] = self._phase_commutators(seq, target, deadline)
        info["after_commutators"] = seq.T
        info["three_cycles"] = self._phase_three_cycles(seq, target, deadline)
        moves = [P.moves[m] for m in seq.moves]
        return SolveResult(moves=canonicalize(moves), wall_time_s=time.time() - t0, info=info)


def tab_slot(tab: CycleTable, stickers: np.ndarray) -> np.ndarray:
    """Global sticker ids -> slot ids of this cluster (vectorised)."""
    lut = getattr(tab, "_lut", None)
    if lut is None:
        size = max(tab.slot_of) + 1
        lut = np.full(size, -1, dtype=np.int64)
        for st, sl in tab.slot_of.items():
            lut[st] = sl
        tab._lut = lut
    return lut[stickers]
