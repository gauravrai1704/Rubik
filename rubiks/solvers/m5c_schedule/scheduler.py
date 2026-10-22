"""Net-gain scheduling of bulk commutators (Method 5C).

Method 3 batches a commutator type over a block of layers X × Y only when *every* cell of
the block helps its cluster, so once the clusters diverge the blocks shrink to single cells
and batching fades. Here a block is chosen for its **net** value:

    value(X, Y) = Σ_{(i, j) ∈ X × Y} gain(i, j)  -  λ · moves(X, Y)

where gain(i, j) is the change in solved slots of the cluster that cell (i, j) acts on (it
may be zero or negative) and λ is the opportunity cost of a move: the best slots-per-move
any single-cluster 3-cycle offers right now. A block is worth applying exactly when its
slots per move beat that. X and Y come from alternating maximisation (rows given columns,
columns given rows), seeded from the best rows; every candidate is then *applied* to the
state and scored exactly (several cells hitting one cluster compose, and the check that
nothing outside the clusters moved is free).

Two more terms in the round score:
* **Setup sharing**: a commutator conjugated by the same outer setup as the previous one
  costs 2·|setup| less, because ``s' s`` cancels when the moves are canonicalised.
* **Lookahead** (``lookahead=1``): the best few candidates are re-scored by their gain plus
  the best gain-per-move available after them, so a step that sets up a large block wins
  over a slightly better step that does not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..cluster_tools import CycleLibrary
from ..m3_demaine.batching import TypeBank


@dataclass
class Candidate:
    seq: list[int]
    gain: int
    moves: int  # effective moves after setup sharing
    setup: tuple
    label: str
    state: np.ndarray

    @property
    def ratio(self) -> float:
        return self.gain / max(1, self.moves)


class Flat:
    """All ops of a TypeBank in flat arrays, so one round scores every op of every cluster."""

    def __init__(self, bank: TypeBank):
        self.bank = bank
        rows = []
        for c in bank.clusters:
            for k, op in enumerate(bank.ops[c.index]):
                p = op.params + (-1,) * (2 - len(op.params))
                rows.append((op.type_id, c.index, k, p[0], p[1]))
        a = np.array(rows, dtype=np.int64).reshape(-1, 5)
        self.type, self.cluster, self.k, self.p0, self.p1 = a.T
        self.n_params = max((len(o.params) for v in bank.ops.values() for o in v), default=1)
        self.offsets = {}
        start = 0
        for c in bank.clusters:
            n = len(bank.ops[c.index])
            self.offsets[c.index] = (start, start + n)
            start += n
        self.inner = max(int(self.p0.max()), int(self.p1.max())) + 2 if len(rows) else 0

    def gains(self, state: np.ndarray, current: dict[int, int]) -> np.ndarray:
        g = np.zeros(len(self.type), dtype=np.int64)
        for c in self.bank.clusters:
            a, b = self.offsets[c.index]
            if a == b:
                continue
            sl = self.bank.slots[c.index]
            x = state[c.stickers]
            g[a:b] = sl.correct(x[self.bank.local[c.index]]) - current[c.index]
        return g


def _correct(bank: TypeBank, state: np.ndarray) -> dict[int, int]:
    return {c.index: int(bank.slots[c.index].correct(state[c.stickers][None])[0]) for c in bank.clusters}


def _setup_of(bank: TypeBank, t: int) -> tuple:
    return tuple(bank.setups[t // len(bank.templates)])


def _alternate(G: np.ndarray, valid: np.ndarray, lam: float, base: float, seeds: list[int]):
    """Maximise Σ G[X×Y] - λ(2|X| + 2|Y| + base) over X ∩ Y = ∅ with every cell valid."""
    n = G.shape[0]
    best, best_val = None, 0.0
    Gm = np.where(valid, G, 0.0)
    for r in seeds:
        X = np.zeros(n, dtype=bool)
        X[r] = True
        Y = valid[r] & (G[r] > 2 * lam)
        Y[r] = False
        for _ in range(6):
            if not Y.any() or not X.any():
                break
            okX = valid[:, Y].all(1) & ~Y
            Xn = okX & (Gm[:, Y].sum(1) > 2 * lam)
            if not Xn.any():
                break
            okY = valid[Xn, :].all(0) & ~Xn
            Yn = okY & (Gm[Xn, :].sum(0) > 2 * lam)
            if (Xn == X).all() and (Yn == Y).all():
                break
            X, Y = Xn, Yn
        if not X.any() or not Y.any() or (X & Y).any() or not valid[np.ix_(X, Y)].all():
            continue
        val = Gm[np.ix_(X, Y)].sum() - lam * (2 * X.sum() + 2 * Y.sum() + base)
        if best is None or val > best_val:
            best, best_val = (np.nonzero(X)[0].tolist(), np.nonzero(Y)[0].tolist()), val
    return best


class NetGainScheduler:
    def __init__(self, puzzle, bank: TypeBank, fallback: dict[int, CycleLibrary], top_types: int = 24,
                 lookahead: int = 1, beam: int = 4, share_setups: bool = True):
        self.P, self.bank, self.fallback = puzzle, bank, fallback
        self.flat = Flat(bank)
        self.top_types, self.lookahead, self.beam = top_types, lookahead, beam
        self.share = share_setups
        self.outside = np.ones(puzzle.num_stickers, dtype=bool)
        for c in bank.clusters:
            self.outside[c.stickers] = False

    # ---------------------------------------------------------------- candidates
    def _single(self, state, current, last_setup) -> Candidate | None:
        best = None
        for c in self.bank.clusters:
            sl = self.bank.slots[c.index]
            if current[c.index] == sl.n:
                continue
            lib = self.fallback[c.index]
            x = state[c.stickers]
            g = sl.correct(x[lib.local]) - current[c.index]
            if g.max() <= 0:
                continue
            r = np.where(g > 0, g / lib.cost, -np.inf)
            t = int(np.argmax(r))
            if best is None or r[t] > best[0]:
                best = (float(r[t]), c, t, int(g[t]))
        if best is None:
            return None
        _, c, t, g = best
        seq = self.fallback[c.index].seqs[t]
        new = state.copy()
        new[c.stickers] = state[c.stickers][self.fallback[c.index].local[t]]
        return Candidate(list(seq), g, len(seq), (), f"single {c.index}", new)

    def _blocks(self, state, current, lam, last_setup) -> list[Candidate]:
        fl, bank, P = self.flat, self.bank, self.P
        if len(fl.type) == 0:
            return []
        g = fl.gains(state, current)
        pos = np.where(g > 0, g, 0)
        score = np.bincount(fl.type, weights=pos)
        out = []
        for t in np.argsort(-score)[: self.top_types]:
            if score[t] <= 0:
                break
            m = fl.type == t
            setup = _setup_of(bank, int(t))
            disc = 2 * len(setup) if (self.share and setup and setup == last_setup) else 0
            base_moves = 4 + 2 * len(setup) - disc  # moves not proportional to |X|, |Y|
            if fl.n_params == 1:
                L = np.full(fl.inner, -10**6, dtype=np.float64)
                np.maximum.at(L, fl.p0[m], g[m].astype(np.float64))
                X = [int(i) for i in np.nonzero(L > 2 * lam)[0]]
                if not X:
                    continue
                sets = (set(X),)
            else:
                G = np.zeros((fl.inner, fl.inner))
                valid = np.zeros((fl.inner, fl.inner), dtype=bool)
                # several clusters can share a cell under one type only if they are the same cell
                np.add.at(G, (fl.p0[m], fl.p1[m]), g[m])
                valid[fl.p0[m], fl.p1[m]] = True
                rows = np.argsort(-np.where(valid, np.maximum(G, 0), 0).sum(1))[:3]
                blk = _alternate(G, valid, lam, base_moves, [int(r) for r in rows])
                if blk is None:
                    continue
                sets = (set(blk[0]), set(blk[1]))
            seq = bank.sequence(int(t), sets)
            new = P.apply(state, seq)
            if not np.array_equal(new[self.outside], state[self.outside]):
                continue  # the bulk version disturbs something outside these clusters
            after = _correct(bank, new)
            gain = sum(after.values()) - sum(current.values())
            if gain <= 0:
                continue
            out.append(Candidate(seq, gain, len(seq) - disc, setup, f"type {int(t)} x{len(sets[0])}"
                                 + (f"x{len(sets[1])}" if len(sets) > 1 else ""), new))
        return out

    def _best_ratio(self, state, last_setup) -> float:
        current = _correct(self.bank, state)
        s = self._single(state, current, last_setup)
        lam = s.ratio if s else 0.0
        b = self._blocks(state, current, lam, last_setup)
        return max([lam] + [c.ratio for c in b])

    # ---------------------------------------------------------------- main loop
    def solve(self, state: np.ndarray, max_rounds: int = 20000) -> tuple[list[int], dict]:
        state = np.asarray(state).copy()
        out: list[int] = []
        last_setup: tuple = ()
        stats = {"rounds": 0, "bulk_rounds": 0, "single_rounds": 0, "bulk_gain": 0, "single_gain": 0}
        for _ in range(max_rounds):
            current = _correct(self.bank, state)
            if all(current[c.index] == self.bank.slots[c.index].n for c in self.bank.clusters):
                break
            single = self._single(state, current, last_setup)
            lam = single.ratio if single else 0.0
            cands = self._blocks(state, current, lam, last_setup) + ([single] if single else [])
            if not cands:
                raise RuntimeError("net-gain scheduler: no improving step")
            cands.sort(key=lambda c: -c.ratio)
            if self.lookahead and len(cands) > 1:
                top = cands[: self.beam]
                # gain now + what the best next step is worth, over the moves of both
                scored = []
                for c in top:
                    nxt = self._best_ratio(c.state, c.setup)
                    scored.append(((c.gain + nxt * c.moves) / (2 * max(1, c.moves)), c))
                cands = [max(scored, key=lambda x: x[0])[1]]
            c = cands[0]
            out += c.seq
            state = c.state
            last_setup = c.setup
            stats["rounds"] += 1
            key = "single" if c.label.startswith("single") else "bulk"
            stats[f"{key}_rounds"] += 1
            stats[f"{key}_gain"] += c.gain
        else:
            raise RuntimeError("net-gain scheduler did not finish")
        return out, stats


# ------------------------------------------------------------------ Square interiors
def square_interiors_netgain(puzzle, plan, state: np.ndarray, max_rounds: int = 10000) -> tuple[list[int], dict]:
    """Interior clusters of the Square by net-gain Lemma 4 blocks.

    Gain of applying commutator type (column side, row side, inverse) to a cluster = drop in
    its exact distance (in commutators) to solved; it is negative when the commutator takes
    the cluster further away. Each round applies the block X × Y with the best net gain per
    move, or one single commutator (4 moves) when no block beats it."""
    from ...engine.puzzle import Move

    idx, N = puzzle.move_index, puzzle.N
    ics = plan.interiors
    if not ics:
        return [], {}
    h = N // 2
    state = np.asarray(state).copy()
    types = ics[0].ops
    out: list[int] = []
    stats = {"rounds": 0, "bulk_rounds": 0, "single_rounds": 0}

    def dist(ic, x):
        return ic.stage.dist[x.astype(np.int8).tobytes()]

    for _ in range(max_rounds):
        d = np.array([dist(ic, state[ic.cluster.stickers]) for ic in ics])
        if d.sum() == 0:
            break
        best = None  # (ratio, gain, moves, seq)
        for ti, op in enumerate(types):
            G = np.zeros((h, h))
            valid = np.zeros((h, h), dtype=bool)
            for k, ic in enumerate(ics):
                x = state[ic.cluster.stickers][ic.stage.local[ti]]
                G[ic.cols[0], ic.rows[0]] = d[k] - dist(ic, x)
                valid[ic.cols[0], ic.rows[0]] = True
            for lam in (0.25, 0.4):
                seeds = [int(r) for r in np.argsort(-np.maximum(G, 0).sum(1))[:3]]
                blk = _alternate(G, valid, lam, 0, seeds)
                if blk is None:
                    continue
                X, Y = blk
                a, b, f = op
                C = [idx[Move(0, x if a == 0 else N - 1 - x, 2)] for x in X]
                R = [idx[Move(1, y if b == 0 else N - 1 - y, 2)] for y in Y]
                seq = (R + C + R + C) if f else (C + R + C + R)
                gain = float(G[np.ix_(X, Y)].sum())
                if gain > 0 and (best is None or gain / len(seq) > best[0]):
                    best = (gain / len(seq), gain, len(seq), seq)
        if best is None or best[0] < 0.25:  # a single commutator always fixes 1 per 4 moves
            k = int(np.argmax(d))
            ic = ics[k]
            ai = ic.stage.solve_actions(state)[0]
            seq = ic.op_sequence(puzzle, ic.ops[ai])
            stats["single_rounds"] += 1
        else:
            seq = best[3]
            stats["bulk_rounds"] += 1
        state = puzzle.apply(state, seq)
        out += seq
        stats["rounds"] += 1
    else:
        raise RuntimeError("square net-gain scheduler did not finish")
    return out, stats
