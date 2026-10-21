"""Batching for Method 3 (technical basis, Sections 6.1-6.4).

**Blocks (Lemmas 5-6).** A set S of grid cells (column index, row index) that all need the
same fix is covered by blocks X × Y ⊆ S, and one bulk sequence fixes a whole block for
O(|X| + |Y|) moves. :func:`blocks` splits the columns into groups of ℓ, and within a group
gives each nonempty column subset T the rows whose needs are exactly T (Demaine's
construction, with ℓ chosen per call to minimise the actual cost instead of ½ log₂ |Y|).

**Square (Lemma 4).** Each interior cluster's optimal word of commutators is read off its
exact table; round r applies the r-th commutator of every cluster, one bulk
C_X R_Y C_X R_Y per block and commutator type.

**Cube (Section 6.4).** Commutator *types* with layer parameters: ``[A_i, B C_j B']`` for
centers, ``[X_i, Y Z Y']`` for wings, conjugated by an outer-face setup. One type with layer
sets I, J (``A_I`` = every slice of I turned together) applies the same 3-cycle pattern to
every cluster (i, j) in I × J. Each round greedily picks the type with the most slots fixed
per move over all clusters, applies it to every cluster it helps, and re-checks the result
sticker by sticker (a block that disturbs anything else is split until it does not).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from ...engine.orbits import Orbit
from ...engine.puzzle import Move, PuzzleSpec
from ..cluster_tools import CycleLibrary, Slots, sequence_perm


# ------------------------------------------------------------------ blocks
def _blocks_cols(S: set, ell: int) -> list[tuple[list, list]]:
    cols = sorted({c for c, _ in S})
    rows = sorted({r for _, r in S})
    out = []
    for g in range(0, len(cols), ell):
        group = cols[g : g + ell]
        by: dict[tuple, list] = {}
        for r in rows:
            T = tuple(c for c in group if (c, r) in S)
            if T:
                by.setdefault(T, []).append(r)
        out += [(list(T), Y) for T, Y in by.items()]
    return out


def blocks(S: set, cost=lambda X, Y: 2 * len(X) + 2 * len(Y)) -> list[tuple[list, list]]:
    """Cover S exactly by blocks X × Y (each a subset of S), minimising the summed cost over
    group sizes ℓ = 1..6 and both orientations (group columns, or group rows)."""
    if not S:
        return []
    best, best_cost = None, None
    T = {(r, c) for c, r in S}
    for ell in range(1, 7):
        for transposed in (False, True):
            bl = _blocks_cols(T if transposed else S, ell)
            if transposed:
                bl = [(Y, X) for X, Y in bl]
            total = sum(cost(X, Y) for X, Y in bl)
            if best_cost is None or total < best_cost:
                best, best_cost = bl, total
    return best


# ------------------------------------------------------------------ Square interiors
def square_interiors(puzzle: PuzzleSpec, plan, state: np.ndarray) -> tuple[list[int], dict]:
    """Solve every interior cluster at once, round by round (Lemmas 4-6)."""
    idx, N = puzzle.move_index, puzzle.N
    words = {}
    for k, ic in enumerate(plan.interiors):
        w = ic.stage.solve_actions(state)
        if w:
            words[k] = w
    pos = {}  # cluster -> (column-pair index, row-pair index)
    for k, ic in enumerate(plan.interiors):
        pos[k] = (ic.cols[0], ic.rows[0])
    out: list[int] = []
    rounds = max((len(w) for w in words.values()), default=0)
    n_blocks = 0
    for r in range(rounds):
        by_op: dict[tuple, set] = {}
        for k, w in words.items():
            if r < len(w):
                op = plan.interiors[k].ops[w[r]]
                by_op.setdefault(op, set()).add(pos[k])
        for (a, b, f), S in sorted(by_op.items()):
            for X, Y in blocks(S):
                cols = [x if a == 0 else N - 1 - x for x in X]
                rows = [y if b == 0 else N - 1 - y for y in Y]
                C = [idx[Move(0, x, 2)] for x in cols]
                R = [idx[Move(1, y, 2)] for y in rows]
                out += (R + C + R + C) if f else (C + R + C + R)
                n_blocks += 1
    return out, {"rounds": rounds, "blocks": n_blocks, "clusters": len(words)}


# ------------------------------------------------------------------ cube types
@dataclass(frozen=True)
class Template:
    """A commutator with layer parameters. Tokens: ("m", move id) or ("p", slot, axis, turn)."""

    tokens: tuple

    def expand(self, puzzle: PuzzleSpec, layer_sets: tuple) -> list[int]:
        idx = puzzle.move_index
        out = []
        for t in self.tokens:
            if t[0] == "m":
                out.append(t[1])
            else:
                _, slot, axis, turn = t
                out += [idx[Move(axis, layer, turn)] for layer in sorted(layer_sets[slot])]
        return out

    @property
    def n_params(self) -> int:
        return 1 + max((t[1] for t in self.tokens if t[0] == "p"), default=-1)


def center_templates(puzzle: PuzzleSpec) -> list[Template]:
    idx, N = puzzle.move_index, puzzle.N
    out = []
    for a in range(3):
        for ta in (1, 3):
            for fa in range(3):
                if fa == a:
                    continue
                for fl in (0, N - 1):
                    for tb in (1, 3):
                        B, Bi = idx[Move(fa, fl, tb)], idx[Move(fa, fl, 4 - tb)]
                        for tc in (1, 3):
                            out.append(Template((("p", 0, a, ta), ("m", B), ("p", 1, a, tc), ("m", Bi),
                                                 ("p", 0, a, 4 - ta), ("m", B), ("p", 1, a, 4 - tc), ("m", Bi))))
    return out


def wing_templates(puzzle: PuzzleSpec) -> list[Template]:
    idx, N = puzzle.move_index, puzzle.N
    faces = [(ax, layer) for ax in range(3) for layer in (0, N - 1)]
    out = []
    for a in range(3):
        for t in (1, 3):
            for fa, fl in faces:
                for tb in (1, 3):
                    Y, Yi = idx[Move(fa, fl, tb)], idx[Move(fa, fl, 4 - tb)]
                    for za, zl in faces:
                        if za == fa:
                            continue
                        for tz in (1, 2, 3):
                            Z, Zi = idx[Move(za, zl, tz)], idx[Move(za, zl, 4 - tz)]
                            out.append(Template((("p", 0, a, t), ("m", Y), ("m", Z), ("m", Yi),
                                                 ("p", 0, a, 4 - t), ("m", Y), ("m", Zi), ("m", Yi))))
    return out


@dataclass
class Op:
    type_id: int  # (setup, template) index
    params: tuple  # concrete layers
    local: np.ndarray  # local sticker permutation on the cluster
    cost: int  # moves when applied alone


class TypeBank:
    """For every cluster of the given kinds, every (setup, template, layers) that is a pure
    3-cycle on that cluster alone."""

    def __init__(self, puzzle: PuzzleSpec, cl: list[Orbit], templates: list[Template], setup_depth: int = 1):
        self.puzzle = puzzle
        N, S = puzzle.N, puzzle.num_stickers
        inv = puzzle.inverse_move
        self.clusters = cl
        self.slots = {c.index: Slots(puzzle, c) for c in cl}
        owner = np.full(S, -1, dtype=np.int64)
        localpos = np.full(S, -1, dtype=np.int64)
        for c in cl:
            owner[c.stickers] = c.index
            localpos[c.stickers] = np.arange(len(c.stickers))
        outer = [i for i, m in enumerate(puzzle.moves) if m.layer in (0, N - 1)]
        setups = [()] + [(m,) for m in outer]
        if setup_depth >= 2:
            setups += [(m1, m2) for m1 in outer for m2 in outer if puzzle.moves[m1].axis != puzzle.moves[m2].axis]
        self.setups, self.templates = setups, templates
        inner = list(range(1, N - 1))
        self.ops: dict[int, list[Op]] = {c.index: [] for c in cl}
        ident = np.arange(S)
        for ti, tpl in enumerate(templates):
            for params in product(inner, repeat=tpl.n_params):
                if len(set(params)) < len(params):
                    continue
                seq = tpl.expand(puzzle, tuple({p} for p in params))
                q = sequence_perm(puzzle, seq)
                moved = np.nonzero(q != ident)[0]
                if len(moved) == 0:
                    continue
                own = owner[moved]
                if (own < 0).any() or (own != own[0]).any():
                    continue
                c_index = int(own[0])
                sl = self.slots[c_index]
                c = sl.cluster
                lp = localpos[q[c.stickers]]
                if sl.slot_cycle(lp) is None:
                    continue
                for si, s in enumerate(setups):
                    lps = lp
                    if s:
                        a = sl.local_perm(list(s))
                        ai = sl.local_perm([int(inv[m]) for m in reversed(s)])
                        lps = a[lp][ai]
                        if np.array_equal(lps, np.arange(len(lps))):
                            continue
                    self.ops[c_index].append(Op(si * len(templates) + ti, params, lps, len(seq) + 2 * len(s)))
        self.local = {k: (np.stack([o.local for o in v]) if v else None) for k, v in self.ops.items()}
        self.type_ids = {k: np.array([o.type_id for o in v]) for k, v in self.ops.items()}

    def sequence(self, type_id: int, layer_sets: tuple) -> list[int]:
        si, ti = divmod(type_id, len(self.templates))
        s = list(self.setups[si])
        inv = self.puzzle.inverse_move
        return s + self.templates[ti].expand(self.puzzle, layer_sets) + [int(inv[m]) for m in reversed(s)]


def batched_solve(puzzle: PuzzleSpec, bank: TypeBank, state: np.ndarray, fallback: dict[int, CycleLibrary],
                  max_rounds: int = 5000, top_types: int = 32) -> tuple[list[int], dict]:
    """Greedy rounds over all clusters of ``bank`` (Section 6.4). Returns (move ids, stats)."""
    P = puzzle
    state = np.asarray(state).copy()
    out: list[int] = []
    stats = {"rounds": 0, "bulk_rounds": 0, "single_rounds": 0, "clusters_per_round": []}
    for _ in range(max_rounds):
        Ts, Gs, Cs, Ks = [], [], [], []
        best_single = None
        unsolved = 0
        for c in bank.clusters:
            sl = bank.slots[c.index]
            x = state[c.stickers]
            cur = int(sl.correct(x[None])[0])
            if cur == sl.n:
                continue
            unsolved += 1
            L = bank.local[c.index]
            if L is not None:
                gain = sl.correct(x[L]) - cur
                k = np.nonzero(gain > 0)[0]
                Ts.append(bank.type_ids[c.index][k])
                Gs.append(gain[k])
                Cs.append(np.full(len(k), c.index))
                Ks.append(k)
            lib = fallback[c.index]
            g2 = sl.correct(x[lib.local]) - cur
            if g2.max() > 0:
                r = np.where(g2 > 0, g2 / lib.cost, -np.inf)
                t = int(np.argmax(r))
                if best_single is None or r[t] > best_single[0]:
                    best_single = (float(r[t]), c.index, t)
        if unsolved == 0:
            break
        stats["rounds"] += 1
        best = None
        if Ts and sum(len(t) for t in Ts):
            T, G, C, K = (np.concatenate(a) for a in (Ts, Gs, Cs, Ks))
            # rank types by total gain, then plan the best few exactly
            score = np.bincount(T, weights=G)
            for t in np.argsort(-score)[:top_types]:
                if score[t] <= 0:
                    break
                m = T == t
                g, ci, k = G[m], C[m], K[m]
                order = np.lexsort((-g, ci))  # per cluster, best gain first
                chosen: dict[int, tuple] = {}
                for j in order:
                    chosen.setdefault(int(ci[j]), (int(g[j]), int(k[j])))
                plan = _type_plan(bank, int(t), chosen)
                gain = sum(v[0] for v in chosen.values())
                moves = sum(len(seq) for seq, _ in plan)
                ratio = gain / max(1, moves)
                if best is None or ratio > best[0]:
                    best = (ratio, int(t), chosen, plan)
        if best is not None and (best_single is None or best[0] >= best_single[0]):
            _, t, chosen, plan = best
            applied = 0
            for seq, members in plan:
                ok, state2 = _check(P, bank, state, seq, members)
                if not ok:  # split into single clusters
                    for ci, k in members:
                        op = bank.ops[ci][k]
                        s1 = bank.sequence(op.type_id, tuple({p} for p in op.params))
                        ok1, st1 = _check(P, bank, state, s1, [(ci, k)])
                        if ok1:
                            state = st1
                            out += s1
                            applied += 1
                    continue
                state = state2
                out += seq
                applied += len(members)
            stats["bulk_rounds"] += 1
            stats["clusters_per_round"].append(applied)
            if applied:
                continue
        if best_single is None:
            raise RuntimeError("batched_solve: no improving 3-cycle")
        _, ci, t = best_single
        seq = fallback[ci].seqs[t]
        state = P.apply(state, seq)
        out += seq
        stats["single_rounds"] += 1
    else:
        raise RuntimeError("batched_solve did not finish")
    per = stats.pop("clusters_per_round")
    stats["mean_clusters_per_bulk"] = round(float(np.mean(per)), 2) if per else 0.0
    return out, stats


def _type_plan(bank: TypeBank, t: int, chosen: dict[int, tuple]) -> list[tuple[list[int], list]]:
    """Group the chosen clusters' layer parameters into blocks; one sequence per block."""
    ops = {ci: bank.ops[ci][k] for ci, (g, k) in chosen.items()}
    n_par = len(next(iter(ops.values())).params)
    if n_par == 1:
        layers = {}
        for ci, op in ops.items():
            layers.setdefault(op.params[0], []).append((ci, chosen[ci][1]))
        # one parameter: every helped cluster in one bulk move (distinct layers only)
        members = [v[0] for v in layers.values()]
        rest = [x for v in layers.values() for x in v[1:]]
        plan = [(bank.sequence(t, (set(layers),)), members)]
        for ci, k in rest:
            op = bank.ops[ci][k]
            plan.append((bank.sequence(t, ({op.params[0]},)), [(ci, k)]))
        return plan
    cell: dict[tuple, tuple] = {}
    for ci, op in ops.items():
        cell.setdefault(op.params, (ci, chosen[ci][1]))
    plan = []
    S = set(cell)
    for X, Y in blocks(S):
        if set(X) & set(Y):  # a layer cannot be both an A-slice and a C-slice
            for x in X:
                for y in Y:
                    plan.append((bank.sequence(t, ({x}, {y})), [cell[(x, y)]]))
            continue
        plan.append((bank.sequence(t, (set(X), set(Y))), [cell[(x, y)] for x in X for y in Y]))
    used = {m for _, mem in plan for m in mem}
    for ci, op in ops.items():  # clusters whose cell was taken by another cluster
        if (ci, chosen[ci][1]) not in used:
            plan.append((bank.sequence(t, tuple({p} for p in op.params)), [(ci, chosen[ci][1])]))
    return plan


def _check(P: PuzzleSpec, bank: TypeBank, state: np.ndarray, seq: list[int], members) -> tuple[bool, np.ndarray]:
    """Apply ``seq``; accept only if each member cluster changed exactly by its op's 3-cycle
    and nothing else changed."""
    new = P.apply(state, seq)
    expect = state.copy()
    for ci, k in members:
        c = bank.slots[ci].cluster
        expect[c.stickers] = state[c.stickers][bank.ops[ci][k].local]
    return bool(np.array_equal(new, expect)), new
