"""Cluster-level tools shared by Methods 1-3 (technical basis, Sections 4-6).

* :func:`orient` — relabel colours so the target is exactly ``puzzle.solved``. For odd N
  the fixed centers pick the rotation; for even N the rotation that already matches the
  most centers. A relabelled reachable state is still reachable (a colour permutation
  that comes from a rotation equals that rotation of the solved state).
* :class:`Slots` — a cluster's cubies ("slots"), their stickers, the slot permutation of
  every move, and piece identification (which home piece sits in each slot, from colours).
* :func:`pure_three_cycles` — automatic search for commutators that 3-cycle the cubies of
  one cluster and touch nothing else: ``[A, B C B']`` for centers (A, C inner slices on one
  axis, B an outer face), ``[X, Y Z Y']`` for wings (X an inner slice, Y, Z outer faces).
* :class:`CycleLibrary` — every 3-cycle of a cluster's slots, each as a short move
  sequence, by breadth-first conjugation of the base commutators with setup moves
  (``s B s'`` is pure whenever B is).
* :func:`greedy_solve` — solve one cluster with library 3-cycles, at each step the cycle
  that fixes the most slots per move.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..engine.orbits import Orbit, clusters
from ..engine.puzzle import Move, PuzzleSpec


# ------------------------------------------------------------------ orientation
def rotation_colour_maps(puzzle: PuzzleSpec) -> list[np.ndarray]:
    """For each whole-puzzle rotation, the colour map face f -> colour shown on face f."""
    out = []
    for R in puzzle.rotations:
        rs = puzzle.solved[R]
        out.append(np.array([int(rs[sl][0]) for sl in puzzle.face_slices], dtype=np.int64))
    return out


def orient(puzzle: PuzzleSpec, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relabel colours so the goal is ``puzzle.solved``. Returns (relabelled state, the colour
    map used: new = inv[old]). Any move sequence that solves the relabelled state solves the
    original up to a whole-puzzle rotation, which ``is_solved`` accepts."""
    state = np.asarray(state)
    best, best_score = None, -1
    fixed = [c for c in clusters(puzzle) if c.kind in ("center_fixed", "sq_center")]
    for cmap in rotation_colour_maps(puzzle):
        inv = np.empty(6, dtype=np.int64)
        inv[cmap] = np.arange(6)
        new = inv[state.astype(np.int64)].astype(state.dtype)
        if fixed and puzzle.kind == "cube":
            st = np.concatenate([c.stickers for c in fixed])
            if not np.array_equal(new[st], puzzle.solved[st]):
                continue
            return new, inv
        score = int((new == puzzle.solved).sum())
        if score > best_score:
            best, best_score = (new, inv), score
    if best is None:
        raise ValueError("no rotation matches the fixed centers: state is not reachable")
    return best


def exact_orientation_moves(puzzle: PuzzleSpec, state: np.ndarray) -> list[Move]:
    """Moves that bring an odd cube's fixed centers to the reference orientation (middle-slice quarter
    turns, shortest by BFS over the 24 orientations), for benchmarks such as Kaggle Santa 2023 that
    require the exact solved state rather than any rotation of it. Empty for even N."""
    fixed = [c for c in clusters(puzzle) if c.kind == "center_fixed"]
    if not fixed:
        return []
    st = np.concatenate([c.stickers for c in fixed])
    mid = puzzle.N // 2
    gens = [Move(a, mid, t) for a in range(3) for t in (1, 2, 3)]
    start = np.asarray(state)[st]
    goal = puzzle.solved[st]
    sub = {m: np.searchsorted(st, puzzle.move_perms[puzzle.move_index[m]][st]) for m in gens}
    frontier, seen = [(start, [])], {start.tobytes()}
    while frontier:
        nxt = []
        for x, path in frontier:
            if np.array_equal(x, goal):
                return path
            for m, p in sub.items():
                y = x[p]
                if y.tobytes() not in seen:
                    seen.add(y.tobytes())
                    nxt.append((y, path + [m]))
        frontier = nxt
    raise ValueError("fixed centers cannot reach the reference orientation")


def prepare(puzzle: PuzzleSpec, state: np.ndarray, exact: bool = False) -> tuple[np.ndarray, list[Move]]:
    """The state a staged solver works on, and moves to prepend to its answer. ``exact=False``: relabel
    colours (:func:`orient`, the solution may end in any rotation). ``exact=True``: no relabelling;
    the answer must reach ``puzzle.solved`` itself (odd cubes first turn their fixed centers home)."""
    if not exact:
        return orient(puzzle, state)[0], []
    pre = exact_orientation_moves(puzzle, state) if puzzle.kind == "cube" else []
    return puzzle.apply(np.asarray(state), pre), pre


# ------------------------------------------------------------------ slots and pieces
def _compose(perms: np.ndarray, seq) -> np.ndarray:
    """Permutation of a move sequence: ``state[result] == apply(state, seq)``."""
    q = np.arange(perms.shape[1])
    for m in seq:
        q = q[perms[m]]
    return q


@dataclass
class Slots:
    """A cluster's cubie slots. Local sticker k = ``cluster.stickers[k]``."""

    puzzle: PuzzleSpec
    cluster: Orbit
    slot_stickers: list[np.ndarray] = field(default_factory=list)  # local sticker ids per slot
    sticker_slot: np.ndarray | None = None  # (K,) local sticker -> slot

    def __post_init__(self):
        P, c = self.puzzle, self.cluster
        by: dict[tuple, list[int]] = {}
        for k, s in enumerate(c.stickers):
            by.setdefault(tuple(P.sticker_pos[s]), []).append(k)
        self.slot_stickers = [np.array(v) for v in by.values()]
        self.sticker_slot = np.empty(len(c.stickers), dtype=np.int64)
        for j, v in enumerate(self.slot_stickers):
            self.sticker_slot[v] = j
        self.local = {int(m): p for m, p in zip(c.gen_moves, c.gen_perms)}

    @property
    def n(self) -> int:
        return len(self.slot_stickers)

    def local_perm(self, seq) -> np.ndarray:
        q = np.arange(len(self.cluster.stickers))
        for m in seq:
            p = self.local.get(int(m))
            if p is not None:
                q = q[p]
        return q

    def slot_cycle(self, lp: np.ndarray) -> tuple | None:
        """The slot 3-cycle of a local sticker permutation, as (a, b, c) meaning the piece in a
        moves to b, b to c, c to a; None if it is not a pure 3-cycle of slots."""
        moved = np.nonzero(lp != np.arange(len(lp)))[0]
        slots = sorted(set(self.sticker_slot[moved].tolist()))
        if len(slots) != 3:
            return None
        dest = {}
        for j in slots:
            k = int(self.slot_stickers[j][0])
            # new[k] = old[lp[k]]: the piece now at slot j came from slot of lp[k]
            src = int(self.sticker_slot[lp[k]])
            dest[src] = j
        if sorted(dest) != slots or sorted(dest.values()) != slots:
            return None
        a = slots[0]
        return (a, dest[a], dest[dest[a]])

    def correct(self, local_states: np.ndarray) -> np.ndarray:
        """(B, K) local states -> (B,) number of slots showing their solved stickers."""
        ok = local_states == self.cluster.solved[None]
        out = np.zeros(len(local_states), dtype=np.int64)
        for v in self.slot_stickers:
            out += ok[:, v].all(1)
        return out


@lru_cache(maxsize=32)
def _piece_table(dims: tuple, cluster_index: int, walks: int = 400, seed: int = 0) -> dict:
    """(slot, colours shown) -> home slot, learned from random walks on a sticker-labelled copy."""
    Ps = PuzzleSpec(dims, "sticker")
    Pc = PuzzleSpec(dims)
    c = clusters(Pc)[cluster_index]
    sl = Slots(Pc, c)
    home_slot = {int(c.stickers[k]): int(sl.sticker_slot[k]) for k in range(len(c.stickers))}
    rng = np.random.default_rng(seed)
    table: dict = {}
    gm = c.gen_moves
    for _ in range(walks):
        seq = rng.choice(gm, size=60)
        st = Ps.apply(Ps.solved, seq)
        for j, v in enumerate(sl.slot_stickers):
            ids = st[c.stickers[v]]
            cols = tuple(int(Pc.sticker_face[i]) for i in ids)
            table[(j, cols)] = home_slot[int(ids[0])]
    return table


def piece_permutation(puzzle: PuzzleSpec, cluster: Orbit, state: np.ndarray) -> np.ndarray:
    """perm[j] = home slot of the piece now in slot j (clusters with distinguishable pieces)."""
    table = _piece_table(puzzle.dims, cluster.index)
    sl = Slots(puzzle, cluster)
    loc = np.asarray(state)[cluster.stickers]
    return np.array([table[(j, tuple(int(x) for x in loc[v]))] for j, v in enumerate(sl.slot_stickers)])


def parity(perm: np.ndarray) -> int:
    seen = np.zeros(len(perm), dtype=bool)
    p = 0
    for i in range(len(perm)):
        if not seen[i]:
            j, L = i, 0
            while not seen[j]:
                seen[j] = True
                j = perm[j]
                L += 1
            p ^= (L - 1) & 1
    return p


# ------------------------------------------------------------------ commutator search
def _outer(puzzle: PuzzleSpec) -> list[int]:
    N = puzzle.N
    return [i for i, m in enumerate(puzzle.moves) if m.layer in (0, N - 1)]


def pure_three_cycles(puzzle: PuzzleSpec, cluster: Orbit, limit: int = 12) -> list[list[int]]:
    """Up to ``limit`` commutators that 3-cycle this cluster's cubies and move nothing else."""
    P, idx, N = puzzle, puzzle.move_index, puzzle.N
    sl = Slots(P, cluster)
    own = np.zeros(P.num_stickers, dtype=bool)
    own[cluster.stickers] = True
    layers = sorted({int(P.sticker_layer[s, a]) for s in cluster.stickers for a in range(3)} - {0, N - 1})
    faces = [(ax, l) for ax in range(3) for l in (0, N - 1)]
    out, seen = [], set()
    pos = np.full(P.num_stickers, -1, dtype=np.int64)
    pos[cluster.stickers] = np.arange(len(cluster.stickers))

    def keep(seq) -> bool:
        q = _compose(P.move_perms, seq)
        moved = q != np.arange(len(q))
        if not moved.any() or moved[~own].any():
            return False
        lp = pos[q[cluster.stickers]]
        cyc = sl.slot_cycle(lp)
        if cyc is None or cyc in seen:
            return False
        seen.add(cyc)
        out.append(list(seq))
        return True

    if cluster.kind == "wing":
        for a in range(3):
            for k in layers:
                for t in (1, 3):
                    X, Xi = idx[Move(a, k, t)], idx[Move(a, k, 4 - t)]
                    for fa, fl in faces:
                        for tb in (1, 3):
                            Y, Yi = idx[Move(fa, fl, tb)], idx[Move(fa, fl, 4 - tb)]
                            for za, zl in faces:
                                if za == fa:
                                    continue
                                for tz in (1, 2, 3):
                                    Z, Zi = idx[Move(za, zl, tz)], idx[Move(za, zl, 4 - tz)]
                                    keep([X, Y, Z, Yi, Xi, Y, Zi, Yi])
                                    if len(out) >= limit:
                                        return out
    elif cluster.kind.startswith("center_") and cluster.kind != "center_fixed":
        for a in range(3):
            for k in layers:
                for ta in (1, 3):
                    A, Ai = idx[Move(a, k, ta)], idx[Move(a, k, 4 - ta)]
                    for fa, fl in faces:
                        if fa == a:
                            continue
                        for tb in (1, 3):
                            B, Bi = idx[Move(fa, fl, tb)], idx[Move(fa, fl, 4 - tb)]
                            for j in layers:
                                for tc in (1, 3):
                                    C, Ci = idx[Move(a, j, tc)], idx[Move(a, j, 4 - tc)]
                                    keep([A, B, C, Bi, Ai, B, Ci, Bi])
                                    if len(out) >= limit:
                                        return out
    else:
        raise ValueError(f"no pure 3-cycle search for {cluster.kind}")
    return out


class CycleLibrary:
    """Every 3-cycle of one cluster's slots as a move sequence ``s B s'`` (B a base commutator).

    Built by breadth-first conjugation: level d holds cycles whose shortest setup has d moves.
    ``ops``: list of (slot cycle, sequence); ``local``: (T, K) local sticker permutations."""

    def __init__(self, puzzle: PuzzleSpec, cluster: Orbit, bases: list[list[int]] | None = None,
                 setup_moves: list[int] | None = None, max_depth: int = 4, metric: str = "htm"):
        """``metric="qtm"``: quarter-turn setups only, more base commutators, and for every cycle the
        sequence with the fewest quarter turns found (Method 6; Kaggle Santa's metric)."""
        self.puzzle, self.cluster = puzzle, cluster
        self.slots = Slots(puzzle, cluster)
        qtm = metric == "qtm"
        mv = puzzle.moves
        cost = (lambda q: sum(2 if mv[m].turn == 2 else 1 for m in q)) if qtm else len
        bases = bases if bases is not None else pure_three_cycles(puzzle, cluster, limit=64 if qtm else 12)
        if not bases:
            raise ValueError(f"no pure 3-cycles found for cluster {cluster.index} ({cluster.kind})")
        inv = puzzle.inverse_move
        G = [int(m) for m in (setup_moves if setup_moves is not None else cluster.gen_moves)]
        G = [m for m in G if int(m) in self.slots.local and (not qtm or mv[m].turn != 2)]
        Pg = np.stack([self.slots.local[m] for m in G])
        Pgi = np.stack([self.slots.local[int(inv[m])] for m in G])
        n = self.slots.n
        full = n * (n - 1) * (n - 2) // 3
        ops: dict[tuple, tuple[list[int], np.ndarray]] = {}
        frontier = []
        for b in bases + [[int(inv[m]) for m in reversed(b)] for b in bases]:
            lp = self.slots.local_perm(b)
            cyc = self.slots.slot_cycle(lp)
            if cyc is not None and (cyc not in ops or cost(b) < cost(ops[cyc][0])):
                if cyc not in ops:
                    frontier.append(cyc)
                ops[cyc] = (list(b), lp)
        for _ in range(max_depth):
            if (len(ops) >= full and not qtm) or not frontier:
                break
            nxt = []
            for cyc in frontier:
                seq, lp = ops[cyc]
                cand = np.take_along_axis(Pg[:, lp], Pgi, 1)  # (G, K): g, seq, g'
                for gi in range(len(G)):
                    c2 = self.slots.slot_cycle(cand[gi])
                    if c2 is None:
                        continue
                    new = [G[gi]] + seq + [int(inv[G[gi]])]
                    if c2 not in ops or (qtm and cost(new) < cost(ops[c2][0])):
                        if c2 not in ops:
                            nxt.append(c2)
                        ops[c2] = (new, cand[gi])
            frontier = nxt
        self.complete = len(ops) >= full
        self.cycles = list(ops)
        self.seqs = [ops[c][0] for c in self.cycles]
        self.local = np.stack([ops[c][1] for c in self.cycles])
        self.cost = np.array([cost(s) for s in self.seqs], dtype=np.float64)


def greedy_solve(lib: CycleLibrary, local_state: np.ndarray, max_steps: int = 400) -> tuple[list[int], np.ndarray]:
    """Solve one cluster with library 3-cycles: each step takes the cycle with the best
    (slots fixed) / (moves) ratio. Returns (move ids, final local state)."""
    x = np.asarray(local_state).copy()
    sl = lib.slots
    goal = sl.n
    moves: list[int] = []
    cur = int(sl.correct(x[None])[0])
    for _ in range(max_steps):
        if cur == goal:
            return moves, x
        cand = x[lib.local]  # (T, K)
        gain = sl.correct(cand) - cur
        if gain.max() <= 0:
            raise RuntimeError(f"cluster {lib.cluster.index}: no improving 3-cycle "
                               f"({cur}/{goal} slots; library complete={lib.complete})")
        score = np.where(gain > 0, gain / lib.cost, -np.inf)
        t = int(np.argmax(score))
        moves += lib.seqs[t]
        x = cand[t]
        cur += int(gain[t])
    raise RuntimeError("greedy_solve did not finish")


# ------------------------------------------------------------------ exact projected stages
def sequence_perm(puzzle: PuzzleSpec, seq) -> np.ndarray:
    return _compose(puzzle.move_perms, seq)


def commutator(a: list[int], b: list[int], inv: np.ndarray) -> list[int]:
    """[a, b] = a b a' b' as move ids."""
    ai = [int(inv[m]) for m in reversed(a)]
    bi = [int(inv[m]) for m in reversed(b)]
    return list(a) + list(b) + ai + bi


class ExactStage:
    """Exact distances over a sticker subset under a set of actions (move sequences), by BFS
    from solved; then optimal (in actions) descent. The action set must be closed under
    inverses. Used for the Square frame and edge groups, and single Square clusters."""

    def __init__(self, puzzle: PuzzleSpec, stickers: np.ndarray, actions: list[list[int]], cap: int = 3_000_000):
        self.puzzle = puzzle
        self.stickers = np.asarray(stickers)
        self.actions = [list(a) for a in actions]
        pos = {int(s): k for k, s in enumerate(self.stickers)}
        self.local = []
        for a in self.actions:
            q = sequence_perm(puzzle, a)
            if not set(q[self.stickers].tolist()) <= set(pos):
                raise ValueError("an action moves stickers into the subset from outside it")
            self.local.append(np.array([pos[int(x)] for x in q[self.stickers]]))
        start = puzzle.solved[self.stickers].astype(np.int8)
        self.dist = {start.tobytes(): 0}
        frontier, d = [start], 0
        while frontier:
            d += 1
            nxt = []
            for st in frontier:
                for p in self.local:
                    k = st[p]
                    b = k.tobytes()
                    if b not in self.dist:
                        self.dist[b] = d
                        nxt.append(k)
            if len(self.dist) > cap:
                raise RuntimeError(f"exact stage over {len(self.stickers)} stickers exceeds {cap} states")
            frontier = nxt

    def solve_actions(self, state: np.ndarray) -> list[int]:
        """Indices into ``actions`` of an optimal path to solved."""
        x = np.asarray(state)[self.stickers].astype(np.int8)
        d = self.dist.get(x.tobytes())
        if d is None:
            raise RuntimeError("stage subset not reachable with this action set")
        out: list[int] = []
        while d > 0:
            for ai, p in enumerate(self.local):
                y = x[p]
                dy = self.dist.get(y.tobytes())
                if dy is not None and dy < d:
                    out.append(ai)
                    x, d = y, dy
                    break
        return out

    def solve(self, state: np.ndarray) -> list[int]:
        return [m for ai in self.solve_actions(state) for m in self.actions[ai]]
