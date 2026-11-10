"""The paper's two main tables: every method per size on the Rubik's Square (random states, HTM) and on the Rubik's
Cube (Santa 2023 puzzles, QTM = the competition score). In each row the best mean is red and the second best green;
only methods that solved every instance of that row are ranked, and ties share the colour.

Run by make_tables.py: ``build(macro)``. Also prints which methods are best or second best at some size.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "paper" / "results"
GEN = ROOT / "paper" / "generated"


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def by_id(rs) -> dict:
    return {r["id"]: r for r in rs}


def fmt(x, nd=1):
    return "--" if x is None else f"{x:,.{nd}f}"


def rank_cells(vals: dict, order: list[str], nd=1) -> tuple[list[str], dict]:
    """vals[k] = (mean, solved, total) or None. Returns formatted cells and {k: 1 or 2} for the ranked ones."""
    full = {k: v[0] for k, v in vals.items() if v and v[1] == v[2] and v[2] > 0}
    distinct = sorted(set(round(x, 6) for x in full.values()))
    place = {}
    for k, x in full.items():
        r = distinct.index(round(x, 6)) + 1
        if r <= 2:
            place[k] = r
    cells = []
    for k in order:
        v = vals.get(k)
        if not v or v[1] == 0:
            cells.append("--" if not v else f"--$^{{0/{v[2]}}}$")
            continue
        s = fmt(v[0], nd)
        if v[1] < v[2]:
            s += f"$^{{{v[1]}/{v[2]}}}$"
        if place.get(k) == 1:
            s = f"\\best{{{s}}}"
        elif place.get(k) == 2:
            s = f"\\second{{{s}}}"
        cells.append(s)
    return cells, place


def stat(src: dict, ids, key="htm"):
    sub = [src[i] for i in ids if i in src]
    if not sub:
        return None
    ok = [r[key] for r in sub if r.get("solved") and r.get(key) is not None]
    return (float(np.mean(ok)) if ok else None, len(ok), len(ids))




def build(macro):
    top = defaultdict(set)

    # ------------------------------------------------------------------ Square (random states, HTM)
    base = by_id(rows(RES / "m5c_schedule_square.jsonl"))
    ids_by_N = defaultdict(list)
    for i, r in base.items():
        ids_by_N[r["N"]].append(i)
    opt = json.loads((RES / "optimal_square.json").read_text()) if (RES / "optimal_square.json").exists() else {}
    optlen = {i: v for N in opt for i, v in opt[N]["instances"].items()}
    for i, r in by_id(rows(RES / "m1_square.jsonl")).items():
        if r["solved"] and r.get("info", {}).get("optimal"):
            optlen.setdefault(i, r["htm"])
    m1 = {i: {"solved": True, "htm": v} for i, v in optlen.items()}
    m4 = by_id(rows(RES / "m4_square_W4096.jsonl"))
    for r in rows(RES / "m4_square_cpu_vs_gpu_large.jsonl"):  # N = 8: the GPU-trained model is the default
        if r["N"] == 8 and r["model"] == "gpu_large":
            m4[r["id"]] = r
    src = {"M1": m1, "M2": by_id(rows(RES / "m2_reduction_square.jsonl")),
           "M3": by_id(rows(RES / "m3_demaine_square.jsonl")), "M4": m4,
           "5A": by_id(rows(RES / "m5a_square_gpu_large.jsonl")), "5C": base,
           "M7": by_id(rows(RES / "m7_square.jsonl"))}
    order = ["M7", "M1", "M4", "5A", "5C", "M3", "M2"]
    lines = []
    for N in sorted(ids_by_N):
        ids = sorted(ids_by_N[N])
        vals = {}
        for k in order:
            st = stat(src[k], ids)
            if k == "M1" and st and st[1] < len(ids):  # optimum known only for some states: not a method row then
                st = (st[0], st[1], len(ids)) if st[1] else None
            vals[k] = st
        cells, place = rank_cells(vals, order)
        for k, r in place.items():
            top[("square", k)].add((N, r))
        lines.append(f"{N} & {len(ids)} & " + " & ".join(cells) + " \\\\")
    (GEN / "tab_main_square_rows.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")

    # ------------------------------------------------------------------ Cube (Santa 2023, QTM)
    S = RES / "santa"
    pub = by_id(rows(S / "kmcoders.jsonl"))
    one = {}
    for N in range(4, 34):
        one.update(by_id(rows(S / f"kmcoders_program_N{N}.jsonl")))
    import santa_part

    m7 = santa_part.M7BEST
    csrc = {"M7": m7, "B1": pub, "B2": one, "B3": by_id(rows(S / "dwalton76.jsonl")),
            "M6": by_id(rows(S / "m6_kmcoders.jsonl")), "5C": by_id(rows(S / "m5c_schedule.jsonl")),
            "M3": by_id(rows(S / "m3_demaine.jsonl")), "M2": by_id(rows(S / "m2_reduction.jsonl"))}
    corder = ["M7", "B1", "B2", "B3", "M6", "5C", "M3", "M2"]
    lines = []
    for N in (4, 5, 6, 7, 8, 9):  # stopped after two consecutive sizes where SymPort lost (N = 8, 9)
        ids = sorted(i for i, r in pub.items() if r["N"] == N)
        vals = {k: stat(csrc[k], ids, "qtm") for k in corder}
        if not (vals["M7"] and vals["M7"][1] == vals["M7"][2]):  # SymPort not run on every puzzle of this size: "--"
            vals["M7"] = None
        cells, place = rank_cells(vals, corder)
        for k, r in place.items():
            top[("cube", k)].add((N, r))
        lines.append(f"{N} & {len(ids)} & " + " & ".join(cells) + " \\\\")
    (GEN / "tab_main_cube_rows.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")

    for puz, nsizes in (("square", len(ids_by_N)), ("cube", 6)):
        tag = "Square" if puz == "square" else "Cube"
        red = sorted(N for N, r in top.get((puz, "M7"), set()) if r == 1)
        green = sorted(N for N, r in top.get((puz, "M7"), set()) if r == 2)
        macro(f"MsevenRed{tag}", len(red))
        macro(f"MsevenGreen{tag}", len(green))
        macro(f"Sizes{tag}", nsizes)
        macro(f"MsevenRedList{tag}", ", ".join(map(str, red)) if red else "none")
    for (puz, k), v in sorted(top.items()):
        print(f"top-2 {puz:6s} {k:3s}", sorted(v))
    return top
