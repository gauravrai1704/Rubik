"""Demo server: a local web page that scrambles, solves and replays Rubik's Squares and Cubes with our engine.

    python demo/server.py            # then open http://localhost:8000
    python demo/server.py --port 8080

Standard library only (plus the project itself). Solvers run in a pool of worker processes so several can run at
once; every solution is re-applied by the engine before it is shown ("verified"). The Santa 2023 tab replays results
precomputed by demo/precompute_santa.py; nothing slow runs live.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# name shown, registry name, kwargs, puzzles and sizes it runs at live (seconds on a 4-core laptop)
SOLVERS = {
    "IDA*-PDB": {"reg": "m1_idastar", "kw": {}, "square": (3, 4), "cube": None, "note": "optimal"},
    "Reduction": {"reg": "m2_reduction", "kw": {}, "square": (3, 20), "cube": (3, 8)},
    "Demaine": {"reg": "m3_demaine", "kw": {}, "square": (3, 20), "cube": (3, 8)},
    "CayleyPy": {"reg": "m4_cayleypy", "kw": {"width": 1024}, "square": (3, 8), "cube": None},
    "OneNet": {"reg": "m5a_tokens", "kw": {}, "square": (4, 8), "cube": None},
    "NetGain": {"reg": "m5c_schedule", "kw": {}, "square": (3, 20), "cube": (3, 8)},
    "SymPort": {"reg": "m7_portfolio", "kw": None, "square": (3, 20), "cube": None},
}


@lru_cache(maxsize=None)
def puzzle(kind: str, N: int):
    from rubiks.engine import make

    return make(kind, N)


def symport_kw(N: int) -> dict:
    # the paper's Square configuration (CayleyPy at W = 1024 instead of 4096 to stay interactive)
    inners = ([("m4_cayleypy", {"width": 1024}), ("m5c_schedule", {})] if N <= 8
              else [("m5c_schedule", {}), ("m3_demaine", {})])
    return {"copies": 8, "workers": 2, "inners": inners}


def available(kind: str, N: int) -> list[str]:
    out = []
    for name, s in SOLVERS.items():
        rng = s[kind]
        if rng and rng[0] <= N <= rng[1]:
            out.append(name)
    return out


# ------------------------------------------------------------------ work done in the worker processes
def _solve(kind: str, N: int, state: list[int], name: str) -> dict:
    from rubiks.engine import metrics
    from rubiks.solvers import Budget, get_solver, load_all

    load_all()
    P = puzzle(kind, N)
    s = np.asarray(state, dtype=P.solved.dtype)
    spec = SOLVERS[name]
    kw = symport_kw(N) if name == "SymPort" else spec["kw"]
    t0 = time.time()
    try:
        r = get_solver(spec["reg"], P, Budget(wall_time_s=120), **kw).solve(s)
        moves = r.moves
    except Exception as e:  # shown as a failure, like the harness does
        return {"solver": name, "ok": False, "error": repr(e)[-200:], "time_s": round(time.time() - t0, 1)}
    ok = bool(P.is_solved(P.apply(s, moves)))  # re-applied here: the solver never decides success
    m = metrics(moves, P) if ok else {"htm": None, "qtm": None}
    return {"solver": name, "ok": ok, "htm": m["htm"], "qtm": m["qtm"], "time_s": round(time.time() - t0, 2),
            "ids": [int(P.move_index[x]) for x in moves]}


def _warm() -> bool:
    import torch

    torch.set_num_threads(1)  # several workers share the CPU: one thread each avoids oversubscription
    from rubiks.solvers import load_all

    load_all()
    return True


# ------------------------------------------------------------------ geometry for drawing
def geometry(kind: str, N: int) -> dict:
    """Where each sticker is drawn: face, row and column; plus the move permutations for client-side replay."""
    from rubiks.engine import clusters, to_notation

    P = puzzle(kind, N)
    pos, nrm = P.sticker_pos, P.sticker_normal
    # (right, down) vectors per outward normal: a standard cube net, U on top of L F R B, D below
    frames = {(0, 1, 0): ("U", (1, 0, 0), (0, 0, 1)), (0, 0, 1): ("F", (1, 0, 0), (0, -1, 0)),
              (1, 0, 0): ("R", (0, 0, -1), (0, -1, 0)), (0, 0, -1): ("B", (-1, 0, 0), (0, -1, 0)),
              (-1, 0, 0): ("L", (0, 0, 1), (0, -1, 0)), (0, -1, 0): ("D", (1, 0, 0), (0, 0, -1))}
    if kind == "square":  # big faces are +-z; both drawn as seen from the top, sides around the top view
        frames = {(0, 0, 1): ("top", (1, 0, 0), (0, -1, 0)), (0, 0, -1): ("bottom", (1, 0, 0), (0, -1, 0)),
                  (0, 1, 0): ("north", (1, 0, 0), (0, 0, 1)), (0, -1, 0): ("south", (1, 0, 0), (0, 0, 1)),
                  (1, 0, 0): ("east", (0, -1, 0), (0, 0, 1)), (-1, 0, 0): ("west", (0, -1, 0), (0, 0, 1))}
    st = []
    for p, n in zip(pos, nrm):
        face, r, d = frames[tuple(int(x) for x in n)]
        col = (int(np.dot(p, r)) + N - 1) // 2
        row = (int(np.dot(p, d)) + N - 1) // 2
        st.append([face, row, col])
    kind_of = [""] * P.num_stickers
    for c in clusters(P):
        for i in c.stickers:
            kind_of[int(i)] = c.kind
    face_colour = {}  # colour index -> the net face whose solved stickers carry it
    for i, v in enumerate(P.solved):
        face_colour.setdefault(int(v), st[i][0])
    return {"kind": kind, "N": N, "stickers": st, "cluster": kind_of, "solved": [int(x) for x in P.solved],
            "colour_face": face_colour, "perms": P.move_perms.tolist(),
            "labels": [to_notation([m], P) for m in P.moves], "solvers": available(kind, N)}


# ------------------------------------------------------------------ HTTP
class Handler(SimpleHTTPRequestHandler):
    pool: ProcessPoolExecutor

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE / "static"), **k)

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/santa"):
            return self._json(json.loads((HERE / "data" / "santa.json").read_text()))
        if self.path.startswith("/api/geometry"):
            from urllib.parse import parse_qs, urlparse

            q = parse_qs(urlparse(self.path).query)
            return self._json(geometry(q["kind"][0], int(q["N"][0])))
        return super().do_GET()

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        kind, N = req.get("kind"), int(req.get("N", 0))
        if self.path == "/api/scramble":
            from rubiks.engine import random_state

            P = puzzle(kind, N)
            s = random_state(P, np.random.default_rng(int(req.get("seed", 0))))
            return self._json({"state": [int(x) for x in s]})
        if self.path == "/api/solve":
            if req["solver"] not in available(kind, N):
                return self._json({"solver": req["solver"], "ok": False, "error": "not offered at this size"})
            fut = self.pool.submit(_solve, kind, N, req["state"], req["solver"])
            return self._json(fut.result())
        if self.path == "/api/verify":  # independent check of any move list
            P = puzzle(kind, N)
            s = np.asarray(req["state"], dtype=P.solved.dtype)
            end = P.apply(s, [int(i) for i in req["ids"]])
            exact = bool(np.array_equal(end, P.solved))
            return self._json({"exact_goal": exact, "solved_any_orientation": bool(P.is_solved(end)),
                               "moves": len(req["ids"])})
        self._json({"error": "unknown endpoint"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    Handler.pool = ProcessPoolExecutor(a.workers, initializer=_warm)
    for f in [Handler.pool.submit(_warm) for _ in range(a.workers)]:  # start the workers before the first click
        f.result()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"demo running at http://localhost:{a.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
