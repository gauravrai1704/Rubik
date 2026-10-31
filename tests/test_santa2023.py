"""Santa 2023 adapter: derived sticker/move mapping, KMCoders' published solutions verify, exact goal.
Needs the competition data (downloaded on first use); skipped without network."""
import numpy as np
import pytest

from rubiks.engine import cube

santa = pytest.importorskip("rubiks.bench.santa2023")


@pytest.fixture(scope="module", autouse=True)
def data():
    try:
        santa.fetch()
    except Exception as e:  # no network: nothing to test against
        pytest.skip(f"Santa 2023 data unavailable: {e}")


@pytest.mark.parametrize("N", [2, 3, 4, 5])
def test_mapping_replays_kmcoders(N):
    P = cube(N)
    ins = santa.instances(N)
    assert ins
    for i in ins[:5]:
        mv = santa.kmcoders_solution(int(i.id.split("-")[-1]), N)
        assert P.is_identity(P.apply(i.state, mv))  # Santa's goal is the exact solved state


def test_move_names_round_trip():
    m = santa.santa_map(4)
    text = "f0.-r2.d3.-d1"
    assert m.moves_to_santa(m.moves_to_engine(text)) == text


@pytest.mark.parametrize("name", ["m2_reduction", "m5c_schedule"])
def test_exact_orientation_reaches_santa_goal(name):
    from rubiks.solvers import get_solver

    P = cube(5)
    i = santa.instances(5)[0]
    r = get_solver(name, P, exact_orientation=True).solve(i.state)
    assert P.is_identity(P.apply(i.state, r.moves))


def test_dwalton76_notation_and_facelets():
    """No dwalton76 install needed: wide turns expand to slice turns; facelet order covers every sticker."""
    from rubiks.engine import parse_notation
    from rubiks.solvers.external.dwalton76 import facelet_order, to_moves

    P = cube(5)
    assert sorted(facelet_order(P).tolist()) == list(range(P.num_stickers))
    assert to_moves("R U' F2", P) == parse_notation("R U' F2", P)
    assert to_moves("Rw", P) == parse_notation("R 2R", P)
    assert to_moves("3Uw2", P) == parse_notation("U2 2U2 3U2", P)
    assert to_moves("2L'", P) == parse_notation("2L'", P)
