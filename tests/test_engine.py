"""Move-group identities, notation, canonical form, metrics and orbit counts (Section 2-3)."""
import numpy as np
import pytest

from rubiks.engine import (Move, canonicalize, clusters, cube, metrics, orbit_summary, parse_notation, scramble,
                           square, to_notation)

SIZES = [cube(3), cube(4), cube(5), cube(6), cube(7), square(3), square(4), square(5), square(8)]


@pytest.mark.parametrize("P", SIZES, ids=repr)
def test_move_identities(P):
    S = P.num_stickers
    ident = np.arange(S)
    for i, m in enumerate(P.moves):
        p = P.move_perms[i]
        order = 2 if m.turn == 2 else 4
        q = ident
        for _ in range(order):
            q = q[p]
        assert np.array_equal(q, ident), f"{m} does not have order {order}"
        assert np.array_equal(ident[p][P.move_perms[P.inverse_move[i]]], ident)


@pytest.mark.parametrize("P", SIZES, ids=repr)
def test_sticker_counts_and_moves(P):
    N = P.N
    if P.kind == "cube":
        assert P.num_stickers == 6 * N * N
        assert len(P.moves) == 9 * N  # 3 axes x N layers x 3 turns
    else:
        assert P.num_stickers == 2 * N * N + 4 * N
        assert len(P.moves) == 2 * N  # one 180° flip per row and column


def test_commuting_same_axis():
    P = cube(5)
    a, b = Move(0, 1, 1), Move(0, 3, 2)
    s = P.apply(P.solved, [Move(1, 0, 1)])
    assert np.array_equal(P.apply(s, [a, b]), P.apply(s, [b, a]))


@pytest.mark.parametrize("P", [cube(3), cube(6), square(7)], ids=repr)
def test_notation_roundtrip(P, rng):
    mv = scramble(P, 40, rng)
    assert parse_notation(to_notation(mv, P), P) == mv


def test_canonical_and_metrics():
    P = cube(4)
    mv = [Move(0, 2, 1), Move(0, 0, 1), Move(0, 2, 3), Move(1, 0, 2)]
    c = canonicalize(mv)
    assert c == [Move(0, 0, 1), Move(1, 0, 2)]
    m = metrics(mv, P)
    assert m == {"htm": 2, "qtm": 3, "stm": 2}
    wide = [Move(0, 2, 3), Move(0, 3, 3)]  # 2-layer block turn = 1 in STM
    assert metrics(wide, P)["stm"] == 1


def test_scramble_is_canonical(rng):
    P = cube(5)
    mv = scramble(P, 200, rng)
    assert canonicalize(mv) == mv


# Orbit counts per N from the technical basis, Section 2.2 (cluster level).
@pytest.mark.parametrize(
    "N,wing_layers,centers,midge",
    [(3, 0, 0, True), (4, 1, 1, False), (5, 1, 2, True), (6, 2, 4, False), (7, 2, 6, True), (10, 4, 16, False),
     (20, 9, 81, False)],
)
def test_cube_cluster_counts(N, wing_layers, centers, midge):
    s = orbit_summary(cube(N), level="cluster")
    assert s.get("wing", 0) == wing_layers
    assert sum(v for k, v in s.items() if k.startswith("center_") and k != "center_fixed") == centers
    assert ("midge" in s) == midge
    assert s["corner"] == 1


@pytest.mark.parametrize("N", [4, 5, 8, 9, 20])
def test_square_cluster_counts(N):
    s = orbit_summary(square(N), level="cluster")
    assert s.get("sq_interior", 0) == (N // 2 - 1) ** 2
    assert s.get("sq_edge", 0) == 2 * (N // 2 - 1)
    assert ("sq_center" in s) == (N % 2 == 1)


def test_cluster_signatures_are_n_invariant():
    """Same cluster type = same projected puzzle at every N: the basis of 5A and 5B."""
    sigs = {}
    for N in (5, 7, 9):
        for c in clusters(cube(N)):
            sigs.setdefault(c.kind, set()).add(c.signature)
    for kind in ("corner", "midge", "wing", "center_x", "center_plus"):
        assert len(sigs[kind]) == 1, kind
