"""Build demo/data/santa.json: real SymPort results on a few Santa 2023 cubes, ready to replay in the demo.

For each chosen puzzle it stores the start state (engine sticker order), every symmetric copy SymPort ran (its
quarter-turn length), the moves of SymPort's best copy, KMCoders' published moves, and one run of their program's
length. Every stored solution is re-applied here and must reach the exact Santa goal.

    python demo/precompute_santa.py

Needs the Santa puzzle files (downloaded on first use into data/santa2023/) and SymPort's copy cache
(docs/results/santa/m7_copy_cache, written by experiments/m7_santa.py); nothing is re-run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rubiks.bench import santa2023
from rubiks.engine import cube, metrics
from rubiks.engine.puzzle import Move

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results" / "santa"
CACHE = RES / "m7_copy_cache"
OUT = Path(__file__).resolve().parent / "data" / "santa.json"

# (N, puzzle id): wins up to N = 7, then one win and one loss at N = 8 and at N = 9
CHOSEN = [(4, 162), (4, 192), (4, 194), (4, 171), (5, 232), (6, 254), (7, 261), (8, 262), (8, 263), (9, 268), (9, 271)]


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def spec_key(N: int) -> str:
    kw = {"binary": "solve_cube_beam", "sa_iters": 1_000_000, "stage1": "nxopt" if N % 2 else "kociemba"}
    return json.dumps([["kmcoders", kw]], sort_keys=True, default=str)


def main() -> None:
    pub = {r["id"]: r for r in rows(RES / "kmcoders.jsonl")}
    out = []
    for N, pid in CHOSEN:
        P = cube(N)
        ins = next(i for i in santa2023.instances(N) if i.id == f"santa-cube{N}-{pid:04d}")
        h = hashlib.sha1(np.asarray(ins.state).tobytes() + spec_key(N).encode()).hexdigest()[:16]
        f = CACHE / f"cube{N}_{h}.jsonl"
        copies = [c for c in rows(f) if c["ok"]]
        if not copies:
            print("no saved copies for", ins.id)
            continue
        lens = []
        for c in copies:
            mv = [Move(*m) for m in c["moves"]]
            assert np.array_equal(P.apply(ins.state, mv), P.solved), (ins.id, c["r"])
            lens.append({"copy": c["r"], "qtm": metrics(mv, P)["qtm"], "ids": [P.move_index[m] for m in mv]})
        best = min(lens, key=lambda x: x["qtm"])
        pmoves = santa2023.kmcoders_solution(pid, N)
        assert np.array_equal(P.apply(ins.state, pmoves), P.solved), ins.id
        one = {r["id"]: r for r in rows(RES / f"kmcoders_program_N{N}.jsonl")}.get(ins.id)
        full = {r["id"]: r for r in rows(RES / f"m7_full_N{N}.jsonl")}.get(ins.id)
        out.append({
            "id": ins.id, "N": N, "pid": pid, "state": [int(x) for x in ins.state],
            "copies": [{"copy": x["copy"], "qtm": x["qtm"]} for x in lens],
            "symport": {"qtm": best["qtm"], "copy": best["copy"], "ids": best["ids"]},
            "symport_reported": full["qtm"] if full else None,
            "published": {"qtm": metrics(pmoves, P)["qtm"], "ids": [P.move_index[m] for m in pmoves]},
            "one_run": one["qtm"] if one and one.get("solved") else None,
        })
        print(ins.id, "copies", len(lens), "best", best["qtm"], "published", out[-1]["published"]["qtm"],
              "one run", out[-1]["one_run"])
    # Table 2 of the paper (mean QTM per size), for the summary panel
    table = []
    for N in range(4, 10):
        ids = [i for i, r in pub.items() if r["N"] == N]
        m7 = {r["id"]: r for r in rows(RES / f"m7_full_N{N}.jsonl")}
        if N == 4:  # the first 10 puzzles come from the earlier 48- and 8-copy runs
            a = {r["id"]: r for r in rows(RES / "m7_beam_sa1M_48copies_N4.jsonl")}
            b = {r["id"]: r for r in rows(RES / "m7_default_8copies_N4.jsonl")}
            for i in set(a) & set(b):
                m7.setdefault(i, {"qtm": min(a[i]["qtm"], b[i]["qtm"]), "solved": True})
        if N == 5:
            for r in rows(RES / "m7_beam_sa1M_24copies_nxopt_N5.jsonl"):
                m7.setdefault(r["id"], r)
        one = {r["id"]: r for r in rows(RES / f"kmcoders_program_N{N}.jsonl")}
        mean = lambda src: round(float(np.mean([src[i]["qtm"] for i in ids])), 1) if all(i in src for i in ids) else None
        table.append({"N": N, "n": len(ids), "symport": mean(m7), "published": mean(pub), "one_run": mean(one)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"puzzles": out, "table": table}))
    print("wrote", OUT, f"{OUT.stat().st_size / 1e3:.0f} kB")
    for t in table:
        print(t)


if __name__ == "__main__":
    main()
