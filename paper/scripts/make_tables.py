"""Build every table, figure and in-text number of the paper from result files.

    python paper/scripts/make_tables.py

Reads results/paper/*.jsonl (harness rows), results/paper/optimal_square.json and
checkpoints/m5a/square.json. Writes paper/generated/*.tex and paper/figures/*.pdf.
Nothing in the paper's tables is typed by hand.
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
FIG = ROOT / "paper" / "figures"
GEN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# categorical slots 1-7 of the reference palette (validated: scripts/validate_palette.js)
C5B, CM4, C5A, COPT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CM2, CM3, C5C = "#e87ba4", "#008300", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def rows(name: str) -> list[dict]:
    p = RES / name
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def by_id(rs):
    return {r["id"]: r for r in rs}


def scale(N: int) -> float:
    return N * N / math.log(N)


def ci95(v):
    return 1.96 * float(np.std(v, ddof=1)) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def fmt(x, nd=1):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{nd}f}"


macros: dict[str, str] = {}


def macro(name: str, value) -> None:
    macros[name] = str(value)


# ------------------------------------------------------------------ load
m5b_sq = rows("m5b_square.jsonl")
m5b_cu = rows("m5b_cube.jsonl") + rows("m5b_cube_15_20.jsonl")
m4 = {W: by_id(rows(f"m4_square_W{W}.jsonl")) for W in (16, 64, 256, 1024, 4096)}
m5a = by_id(rows("m5a_square.jsonl") + rows("m5a_square_large.jsonl"))
opt = json.loads((RES / "optimal_square.json").read_text()) if (RES / "optimal_square.json").exists() else {}
optlen = {i: v for N in opt for i, v in opt[N]["instances"].items()}
# Methods 1-3 and 5C (same instances)
m1_sq = by_id(rows("m1_square.jsonl"))
for i, r in m1_sq.items():  # IDA* to every orientation is optimal where it finishes (N = 5 here)
    if r["solved"] and r["info"].get("optimal") and i not in optlen:
        optlen[i] = r["htm"]
base_sq = {"M2": by_id(rows("m2_reduction_square.jsonl")), "M3": by_id(rows("m3_demaine_square.jsonl")),
           "5C": by_id(rows("m5c_schedule_square.jsonl"))}
base_cu = {"M2": by_id(rows("m2_reduction_cube.jsonl")), "M3": by_id(rows("m3_demaine_cube.jsonl")),
           "5C": by_id(rows("m5c_schedule_cube.jsonl"))}
m1_d12 = rows("m1_cube3_d12.jsonl")
m1_rand = rows("m1_cube3_random.jsonl")
gods = json.loads((RES / "gods_square.json").read_text()) if (RES / "gods_square.json").exists() else {}

# ------------------------------------------------------------------ square table
sq_by_N = defaultdict(list)
for r in m5b_sq:
    sq_by_N[r["N"]].append(r)
lines = []
paired_m4, paired_5a = [], []  # (N, 5B/other ratio) over instances both solved
for N in sorted(sq_by_N):
    rs = sq_by_N[N]
    ok = [r for r in rs if r["solved"]]
    h = [r["htm"] for r in ok]
    o = [optlen[r["id"]] for r in rs if r["id"] in optlen]
    m4r = [m4[4096].get(r["id"]) for r in rs]
    m4ok = [x for x in m4r if x and x["solved"]]
    a = [m5a.get(r["id"]) for r in rs]
    aok = [x for x in a if x and x["solved"]]
    for r in ok:
        x = m4[4096].get(r["id"])
        if x and x["solved"]:
            paired_m4.append((N, r["htm"], x["htm"]))
        y = m5a.get(r["id"])
        if y and y["solved"]:
            paired_5a.append((N, r["htm"], y["htm"]))

    def cell(sol, tried):
        if not tried:
            return "--"
        v = np.mean([x["htm"] for x in sol]) if sol else float("nan")
        s = fmt(v)
        return s if len(sol) == len(tried) else f"{s}$^{{{len(sol)}/{len(tried)}}}$"

    m4tried = [x for x in m4r if x]
    atried = [x for x in a if x]
    # mean over this size's instances for every method; bold the shortest that solved all
    means = {}
    for key, src in (("M4", m4[4096]), ("5A", m5a), ("5B", by_id(rs)), *base_sq.items()):
        tried = [src[r["id"]] for r in rs if r["id"] in src]
        sol = [x for x in tried if x["solved"]]
        means[key] = (cell(sol, tried), np.mean([x["htm"] for x in sol]) if sol and len(sol) == len(tried) else None)
    full = {k: v[1] for k, v in means.items() if v[1] is not None}
    best = min(full.values()) if full else None
    cells = [("\\textbf{" + v[0] + "}" if v[1] is not None and abs(v[1] - best) < 1e-9 else v[0])
             for v in means.values()]
    lines.append(f"{N} & {len(rs)} & {fmt(np.mean(o)) if len(o) == len(rs) else '--'} & " + " & ".join(cells) + " \\\\")
(GEN / "tab_square_rows.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
sq_total = len(m5b_sq)
sq_solved = sum(r["solved"] for r in m5b_sq)
macro("SqTotal", sq_total)
macro("SqSolved", sq_solved)
if m5b_sq:
    t20 = [r["wall_time_s"] for r in sq_by_N[20]]
    macro("SqTwentyTime", f"{np.mean(t20):.0f}")
    macro("SqTwentyHTM", f"{np.mean([r['htm'] for r in sq_by_N[20]]):.0f}")
    macro("SqNormThree", f"{np.mean([r['htm'] for r in sq_by_N[3]]) / scale(3):.2f}")
    macro("SqNormTwenty", f"{np.mean([r['htm'] for r in sq_by_N[20]]) / scale(20):.2f}")
if paired_m4:
    rat = [b / c for _, b, c in paired_m4]
    macro("RatioFiveBMfour", f"{np.mean(rat):.2f}")
    macro("PairedMfourN", len(paired_m4))
if paired_5a:
    rat = [b / c for _, b, c in paired_5a]
    macro("RatioFiveBFiveA", f"{np.mean(rat):.2f}")
    macro("PairedFiveAN", len(paired_5a))
    seen = [b / c for N, b, c in paired_5a if N > 8]
    if seen:
        macro("RatioFiveBFiveAUnseen", f"{np.mean(seen):.2f}")
        macro("PairedFiveAUnseenN", len(seen))

# Method 4 width sweep facts
for W in (16, 64, 256, 1024, 4096):
    rs = list(m4[W].values())
    if not rs:
        continue
    tag = {16: "Sixteen", 64: "SixtyFour", 256: "TwoFiftySix", 1024: "OneK", 4096: "FourK"}[W]
    e = [r for r in rs if r["N"] == 8]
    macro(f"MfourEightSolvedW{tag}", f"{sum(r['solved'] for r in e)}/{len(e)}")
    ok = [r["htm"] for r in e if r["solved"]]
    macro(f"MfourEightHTMW{tag}", fmt(np.mean(ok)) if ok else "--")
    macro(f"MfourSolvedW{tag}", f"{sum(r['solved'] for r in rs)}/{len(rs)}")
    macro(f"MfourEightTimeW{tag}", f"{np.mean([r['wall_time_s'] for r in e]):.1f}")

# 5A coverage per N
cov = defaultdict(lambda: [0, 0])
for r in m5a.values():
    cov[r["N"]][1] += 1
    cov[r["N"]][0] += int(r["solved"])
cov_txt = ", ".join(f"$N{{=}}{N}$: {s}/{t}" for N, (s, t) in sorted(cov.items()))
macro("FiveACoverage", cov_txt)

# ------------------------------------------------------------------ optimality (N = 3, 4)
opt_lines = []
for N in ("3", "4"):
    if N not in opt:
        continue
    ids = list(opt[N]["instances"])
    entries = []
    same = all(base_sq["M2"].get(i, {}).get("htm") == base_sq["M3"].get(i, {}).get("htm")
               == base_sq["5C"].get(i, {}).get("htm") for i in ids)
    assert same, "Methods 2, 3, 5C differ on the smallest Squares: give them separate rows"
    for label, src in (("Method 4 ($W{=}4096$)", m4[4096]), ("Method 5A", m5a), ("Method 5B", by_id(m5b_sq)),
                       ("Methods 2, 3, 5C", base_sq["5C"])):
        got = [(src[i]["htm"], opt[N]["instances"][i]) for i in ids if i in src and src[i]["solved"]]
        tried = [i for i in ids if i in src]
        if not tried:
            entries.append(f"{label} & -- & -- & --")
            continue
        gap = [a - b for a, b in got]
        n_opt = sum(1 for a, b in got if a == b)
        entries.append(f"{label} & {len(got)}/{len(tried)} & {np.mean(gap):.1f} & {n_opt}/{len(got)}")
        tag = {"Method 4 ($W{=}4096$)": "Mfour", "Method 5A": "FiveA", "Method 5B": "FiveB",
               "Methods 2, 3, 5C": "Base"}[label]
        macro(f"Gap{tag}{'Three' if N == '3' else 'Four'}", f"{np.mean(gap):.1f}")
        macro(f"Opt{tag}{'Three' if N == '3' else 'Four'}", f"{n_opt}/{len(got)}")
    for k, e in enumerate(entries):
        lead = (f"\\multirow{{4}}{{*}}{{{N}$\\times${N}}} & \\multirow{{4}}{{*}}{{{opt[N]['gods_number']}}} & "
                f"\\multirow{{4}}{{*}}{{{opt[N]['mean_random']:.1f}}}") if k == 0 else " & & "
        opt_lines.append(f"{lead} & {e} \\\\")
    opt_lines.append("\\midrule")
    macro(f"GodSquare{'Three' if N == '3' else 'Four'}", opt[N]["gods_number"])
    macro(f"StatesSquare{'Three' if N == '3' else 'Four'}", f"{opt[N]['states']:,}")
if opt_lines:
    opt_lines.pop()
(GEN / "tab_optimal_rows.tex").write_text("\n".join(opt_lines) + "\n\\bottomrule\n")

# ------------------------------------------------------------------ cube table
cu_by_N = defaultdict(list)
for r in m5b_cu:
    cu_by_N[r["N"]].append(r)
lines = []
for N in sorted(cu_by_N):
    rs = cu_by_N[N]
    ok = [r for r in rs if r["solved"]]
    h = [r["htm"] for r in ok]
    cols = [np.mean(h) if len(ok) == len(rs) else None]
    times = {}
    for key in ("M2", "M3", "5C"):
        got = [base_cu[key].get(r["id"]) for r in rs]
        good = [x for x in got if x and x["solved"]]
        cols.append(np.mean([x["htm"] for x in good]) if got and len(good) == len(rs) else None)
        times[key] = np.mean([x["wall_time_s"] for x in got if x]) if any(got) else None
    best = min(c for c in cols if c is not None)
    txt = [("\\textbf{" + fmt(c) + "}" if c is not None and abs(c - best) < 1e-9 else fmt(c)) for c in cols]
    q5c = [base_cu["5C"][r["id"]]["qtm"] for r in rs if r["id"] in base_cu["5C"]]
    norm = cols[3] / scale(N) if cols[3] else None
    lines.append(f"{N} & {len(rs)} & " + " & ".join(txt) + f" & {fmt(np.mean(q5c)) if q5c else '--'} & "
                 f"{fmt(norm)} & {fmt(times['5C'], 0)} \\\\")
(GEN / "tab_cube_rows.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
macro("CuTotal", len(m5b_cu))
macro("CuSolved", sum(r["solved"] for r in m5b_cu))
if 20 in cu_by_N:
    macro("CuTwentyHTM", f"{cu_by_N[20][0]['htm']:,}")
    macro("CuTwentyMin", f"{cu_by_N[20][0]['wall_time_s'] / 60:.0f}")
    macro("CuFifteenHTM", f"{cu_by_N[15][0]['htm']:,}")
    macro("CuFifteenMin", f"{cu_by_N[15][0]['wall_time_s'] / 60:.0f}")
    macro("CuThreeHTM", f"{np.mean([r['htm'] for r in cu_by_N[3]]):.0f}")

# ------------------------------------------------------------------ ablations
names = {
    "full": "full \\FiveB",
    "no_macros": "$-$ macros",
    "no_exact_stages": "$-$ exact stages",
    "no_center_macro_restriction": "$-$ center restriction",
    "corner_mlp_plus_tables": "corners: max(MLP, tab.)",
    "no_coarse_tables": "$-$ coarse tables",
}
abl = []
for puzzle, title in (("square", "Square"), ("cube", "Cube")):
    variants = [v for v in names if rows(f"ablations/{puzzle}_{v}.jsonl")]
    if not variants:
        continue
    sizes = sorted({r["N"] for r in rows(f"ablations/{puzzle}_full.jsonl")})
    width = "0.40" if puzzle == "square" else "0.56"
    abl.append(f"\\begin{{minipage}}[t]{{{width}\\textwidth}}\\centering")
    abl.append("\\begin{tabular}[t]{@{}l" + "rrr" * len(sizes) + "@{}}\n\\toprule")
    abl.append(f"{title} & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{$N={N}$}}" for N in sizes) + " \\\\")
    abl.append(" & " + " & ".join("solved & HTM & s" for _ in sizes) + " \\\\\n\\midrule")
    for var in variants:
        rs = rows(f"ablations/{puzzle}_{var}.jsonl")
        cells = []
        for N in sizes:
            sub = [r for r in rs if r["N"] == N]
            ok = [r for r in sub if r["solved"]]
            hh = np.mean([r["htm"] for r in ok]) if ok else None
            tt = np.mean([r["wall_time_s"] for r in sub])
            tts = fmt(tt, 1 if tt < 10 else 0)
            cells.append(f"{len(ok)}/{len(sub)} & {fmt(hh, 0)} & {tts}")
            key = "Abl" + "".join(w.capitalize() for w in (puzzle + "_" + var).split("_")) + ("N" + "abcdefghijklmnopqrstu"[N])
            macro(key + "Solved", f"{len(ok)}/{len(sub)}")
            macro(key + "HTM", fmt(hh, 0))
            macro(key + "Time", tts)
        abl.append(f"{names[var]} & " + " & ".join(cells) + " \\\\")
    abl.append("\\bottomrule\n\\end{tabular}\n\\end{minipage}%")
    if puzzle == "square":
        abl.append("\\hfill")
(GEN / "tab_ablation.tex").write_text("\n".join(abl) + "\n" if abl else "(ablations pending)\n")

# ------------------------------------------------------------------ 5A generalisation
meta = json.loads((ROOT / "checkpoints" / "m5a" / "square.json").read_text())
ev = meta["log"][-1]["eval"]
train_N = set(meta["train_N"])
cells = " & ".join(f"{e['N']}{'' if e['N'] in train_N else '$^\\ast$'}" for e in ev)
corr = " & ".join(f"{e['corr']:.2f}" for e in ev)
rmse = " & ".join(f"{e['rmse_norm']:.2f}" for e in ev)
(GEN / "tab_5a_gen.tex").write_text(
    "\\begin{tabular}{l" + "r" * len(ev) + "}\n\\toprule\n$N$ & " + cells + " \\\\\n\\midrule\n"
    "Pearson $r$ & " + corr + " \\\\\nnormalised RMSE & " + rmse + " \\\\\n\\bottomrule\n\\end{tabular}\n")
macro("FiveAParams", f"{198658:,}")
macro("FiveASteps", f"{meta['steps']:,}")
macro("FiveATrainMin", f"{meta['log'][-1]['time_s'] / 60:.0f}")
macro("FiveACorrTwenty", f"{[e for e in ev if e['N'] == 20][0]['corr']:.2f}")
macro("FiveACorrTen", f"{[e for e in ev if e['N'] == 10][0]['corr']:.2f}")
macro("FiveACorrEight", f"{[e for e in ev if e['N'] == 8][0]['corr']:.2f}")


# ------------------------------------------------------------------ Methods 1-3 and 5C
def mean_htm(src, ids):
    v = [src[i]["htm"] for i in ids if i in src and src[i]["solved"]]
    return np.mean(v) if v and len(v) == len(ids) else None


words = {3: "Three", 4: "Four", 5: "Five", 6: "Six", 8: "Eight", 10: "Ten", 15: "Fifteen", 20: "Twenty"}
for key, tag in (("M2", "Mtwo"), ("M3", "Mthree"), ("5C", "FiveC")):
    sq_rs, cu_rs = list(base_sq[key].values()), list(base_cu[key].values())
    macro(f"SqSolved{tag}", f"{sum(r['solved'] for r in sq_rs)}/{len(sq_rs)}")
    macro(f"CuSolved{tag}", f"{sum(r['solved'] for r in cu_rs)}/{len(cu_rs)}")
    for N in (6, 10, 15, 20):
        ids = [r["id"] for r in cu_by_N.get(N, [])]
        m = mean_htm(base_cu[key], ids)
        if m is not None:
            macro(f"Cu{words[N]}HTM{tag}", f"{m:,.0f}")
            macro(f"CuNorm{words[N]}{tag}", f"{m / scale(N):.1f}")
    ids20 = [r["id"] for r in sq_by_N.get(20, [])]
    m = mean_htm(base_sq[key], ids20)
    if m is not None:
        macro(f"SqTwentyHTM{tag}", f"{m:.0f}")
        macro(f"SqNormTwenty{tag}", f"{m / scale(20):.2f}")
for N in (6, 10, 15, 20):
    ids = [r["id"] for r in cu_by_N.get(N, [])]
    a, b = mean_htm(base_cu["5C"], ids), mean_htm(base_cu["M3"], ids)
    if a and b:
        macro(f"FiveCGain{words[N]}", f"{100 * (1 - a / b):.0f}")
ids = [r["id"] for r in cu_by_N.get(20, [])]
if ids and ids[0] in base_cu["5C"]:
    a, c = mean_htm(base_cu["5C"], ids), mean_htm(by_id(m5b_cu), ids)
    macro("FiveCvsFiveBTwenty", f"{100 * (1 - a / c):.0f}")
    macro("FiveCvsMtwoTwenty", f"{100 * (1 - a / mean_htm(base_cu['M2'], ids)):.0f}")
    macro("MthreevsMtwoTwenty", f"{100 * (1 - mean_htm(base_cu['M3'], ids) / mean_htm(base_cu['M2'], ids)):.0f}")
    r5, r3 = base_cu["5C"][ids[0]], base_cu["M3"][ids[0]]
    macro("FiveCTwentyMin", f"{r5['wall_time_s'] / 60:.1f}")
    macro("MthreeTwentyMin", f"{r3['wall_time_s'] / 60:.1f}")
    b3 = r3["info"]["batching"]["centers"]
    macro("MthreeRoundsTwenty", b3["rounds"])
    macro("MthreeSingleRoundsTwenty", b3["single_rounds"])
    sc = r5["info"]["schedule"]["centers"]
    macro("FiveCRoundsTwenty", sc["rounds"])
    macro("FiveCBulkRoundsTwenty", sc["bulk_rounds"])
    st = {x["stage"]: x["moves"] for x in r5["info"]["stages"]}
    macro("FiveCCenterShareTwenty", f"{100 * st['centers (scheduled)'] / r5['htm']:.0f}")
# Square: where 5C is strictly shorter than Method 3 (N >= 9), and ties
better = tie = 0
for N in range(9, 21):
    ids = [r["id"] for r in sq_by_N.get(N, [])]
    a, b = mean_htm(base_sq["5C"], ids), mean_htm(base_sq["M3"], ids)
    if a is None or b is None:
        continue
    better += a < b - 1e-9
    tie += abs(a - b) < 1e-9
macro("SqFiveCBetter", better)
macro("SqFiveCTie", tie)

# Method 1
if m1_d12:
    ok = [r for r in m1_d12 if r["solved"] and r["info"].get("optimal")]
    macro("MoneDTwelveSolved", f"{len(ok)}/{len(m1_d12)}")
    macro("MoneDTwelveHTM", f"{np.mean([r['htm'] for r in ok]):.1f}")
    macro("MoneDTwelveQTM", f"{np.mean([r['qtm'] for r in ok]):.1f}")
    macro("MoneDTwelveMin", min(r["htm"] for r in ok))
    macro("MoneDTwelveMax", max(r["htm"] for r in ok))
    macro("MoneDTwelveTime", f"{np.mean([r['wall_time_s'] for r in ok]):.0f}")
if m1_rand:
    macro("MoneRandTried", len(m1_rand))
    macro("MoneRandSolved", sum(r["solved"] for r in m1_rand))
    macro("MoneRandMin", f"{np.mean([r['wall_time_s'] for r in m1_rand]) / 60:.0f}")
    macro("MoneRandNodes", f"{np.mean([r['info']['nodes'] for r in m1_rand]) / 1e10:.1f}")
    macro("MoneRandBound", min(r["info"]["bound"] for r in m1_rand) - 1)
m1_cov = defaultdict(lambda: [0, 0, []])
for r in m1_sq.values():
    m1_cov[r["N"]][1] += 1
    if r["solved"]:
        m1_cov[r["N"]][0] += 1
        m1_cov[r["N"]][2].append(r["wall_time_s"])
macro("MoneSquareCoverage", ", ".join(f"$N{{=}}{N}$: {s}/{t}" for N, (s, t, _) in sorted(m1_cov.items())))
if 5 in m1_cov and m1_cov[5][2]:
    macro("MoneSquareFiveTime", f"{np.mean(m1_cov[5][2]):.0f}")
for N, v in gods.items():
    macro(f"GodSquare{words[int(N)]}", v["gods_number"])
    macro(f"StatesSquare{words[int(N)]}", f"{v['states']:,}")

# 5C ablations
def abl_mean(name, N):
    rs = [r for r in rows(f"ablations/{name}.jsonl") if r["N"] == N and r["solved"]]
    return np.mean([r["htm"] for r in rs]) if rs else None


abl5 = []
for N in (6, 7, 8, 10, 15, 20):
    ids = [r["id"] for r in cu_by_N.get(N, [])][:2]
    full5 = mean_htm(base_cu["5C"], ids)
    la = (base_cu["5C"][ids[0]]["info"].get("lookahead") if ids else None)
    nola = abl_mean("cube5c_no_lookahead", N) if N != 20 else full5
    withla = full5 if N != 20 else abl_mean("cube5c_lookahead_n20", N)
    noshare = abl_mean("cube5c_no_setup_sharing", N)
    m3 = mean_htm(base_cu["M3"], ids)
    if full5 is None:
        continue
    abl5.append(f"{N} & {fmt(m3)} & {fmt(nola)} & {fmt(withla)} & {fmt(noshare)} \\\\")
    if N == 20 and withla and nola:
        macro("FiveCLookaheadTwenty", f"{withla:,.0f}")
        macro("FiveCNoLookaheadTwenty", f"{nola:,.0f}")
(GEN / "tab_5c_ablation.tex").write_text("\n".join(abl5) + "\n\\bottomrule\n")

# ------------------------------------------------------------------ lower bounds
lb_lines = []
for N in (6, 8, 10, 15, 20):
    k = (N // 2 - 1) ** 2
    sq = math.ceil(k * math.log(6) / math.log(2 * N) - 1)
    cu = math.ceil(k * (math.lgamma(25) - 6 * math.lgamma(5)) / math.log(9 * N) - 1)
    s5b = np.mean([r["htm"] for r in sq_by_N.get(N, []) if r["solved"]]) if sq_by_N.get(N) else None
    c5b = np.mean([r["htm"] for r in cu_by_N.get(N, []) if r["solved"]]) if cu_by_N.get(N) else None
    s5c = mean_htm(base_sq["5C"], [r["id"] for r in sq_by_N.get(N, [])])
    c5c = mean_htm(base_cu["5C"], [r["id"] for r in cu_by_N.get(N, [])])
    lb_lines.append(f"{N} & {max(sq, 0)} & {fmt(s5b, 0)} & {fmt(s5c, 0)} & {max(cu, 0):,} & {fmt(c5b, 0)} & "
                    f"{fmt(c5c, 0)} \\\\")
(GEN / "tab_lb_rows.tex").write_text("\n".join(lb_lines) + "\n\\bottomrule\n")

(GEN / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))

# ------------------------------------------------------------------ figures
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
})


def style(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# Figure: normalised length vs N, both puzzles
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax = axes[0]
Ns = sorted(sq_by_N)
y5b = [np.mean([r["htm"] for r in sq_by_N[N] if r["solved"]]) / scale(N) for N in Ns]
ax.plot(Ns, y5b, "-o", color=C5B, lw=1.6, ms=3.5, label="Method 5B")
for label, src, col, mk in (("Method 4 (W=4096)", m4[4096], CM4, "s"), ("Method 5A", m5a, C5A, "^")):
    pts = defaultdict(list)
    for r in src.values():
        if r["solved"]:
            pts[r["N"]].append(r["htm"])
    xs = sorted(pts)
    if xs:
        ax.plot(xs, [np.mean(pts[N]) / scale(N) for N in xs], "-" + mk, color=col, lw=1.6, ms=3.5, label=label)
for label, src, col, mk in (("Method 2", base_sq["M2"], CM2, "v"), ("Method 3", base_sq["M3"], CM3, "P"),
                            ("Method 5C", base_sq["5C"], C5C, "*")):
    pts = defaultdict(list)
    for r in src.values():
        if r["solved"]:
            pts[r["N"]].append(r["htm"])
    xs = sorted(pts)
    if xs:
        ax.plot(xs, [np.mean(pts[N]) / scale(N) for N in xs], "-" + mk, color=col, lw=2.2 if "5C" in label else 1.2,
                ms=4.5 if "5C" in label else 3.5, label=label)
if opt:
    xs = [int(N) for N in opt]
    ax.plot(xs, [opt[str(N)]["mean_random"] / scale(N) for N in xs], "D", color=COPT, ms=4.5, label="optimal")
ax.set_xlabel("Square size $N$")
ax.set_ylabel("HTM / ($N^2/\\ln N$)")
ax.set_xticks([3, 5, 8, 10, 12, 15, 20])
ax.set_title("(a) N×N×1 Rubik's Square", fontsize=8, loc="left", color=INK)
ax.set_ylim(bottom=0)
style(ax)
ax = axes[1]
Nc = sorted(cu_by_N)
ax.plot(Nc, [np.mean([r["htm"] for r in cu_by_N[N] if r["solved"]]) / scale(N) for N in Nc], "-o",
        color=C5B, lw=1.6, ms=3.5, label="Method 5B")
for label, src, col, mk in (("Method 2", base_cu["M2"], CM2, "v"), ("Method 3", base_cu["M3"], CM3, "P"),
                            ("Method 5C", base_cu["5C"], C5C, "*")):
    pts = defaultdict(list)
    for r in src.values():
        if r["solved"] and r["N"] in cu_by_N:
            pts[r["N"]].append(r["htm"])
    xs = sorted(pts)
    if xs:
        ax.plot(xs, [np.mean(pts[N]) / scale(N) for N in xs], "-" + mk, color=col, lw=2.2 if "5C" in label else 1.2,
                ms=4.5 if "5C" in label else 3.5, label=label)
ax.set_xlabel("Cube size $N$")
ax.set_ylabel("HTM / ($N^2/\\ln N$)")
ax.set_xticks([3, 5, 8, 10, 15, 20])
ax.set_title("(b) N×N×N Rubik's Cube", fontsize=8, loc="left", color=INK)
ax.set_ylim(bottom=0)
style(ax)
handles, labels = {}, []
for a_ in axes:
    for h_, l_ in zip(*a_.get_legend_handles_labels()):
        if l_ not in handles:
            handles[l_] = h_
            labels.append(l_)
fig.legend([handles[l_] for l_ in labels], labels, loc="lower center", ncol=len(labels), fontsize=7,
           handlelength=2.2, columnspacing=1.4)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(FIG / "scaling.pdf")
plt.close(fig)

# Figure: Method 4 length vs W (ordinal N, sequential blue ramp)
ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#1c5cab", "#104281"]
fig, ax = plt.subplots(figsize=(3.3, 2.2))
Ws = [W for W in (16, 64, 256, 1024, 4096) if m4[W]]
for i, N in enumerate(range(3, 9)):
    ys, full = [], []
    for W in Ws:
        sub = [r for r in m4[W].values() if r["N"] == N]
        ok = [r["htm"] for r in sub if r["solved"]]
        ys.append(np.mean(ok) if ok else np.nan)
        full.append(len(ok) == len(sub))
    if Ws:
        ax.plot(Ws, ys, "-", color=ramp[i], lw=1.4)
        for W, y, f in zip(Ws, ys, full):
            ax.plot([W], [y], "o", ms=3.6, color=ramp[i], mfc=ramp[i] if f else "white", mew=1.0)
        ax.annotate(f"N={N}", (Ws[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5,
                    color=INK2, va="center")
ax.set_xscale("log", base=2)
ax.set_xlabel("beam width $W$")
ax.set_ylabel("mean HTM (solved states)")
style(ax)
fig.tight_layout()
fig.savefig(FIG / "m4_width.pdf")
plt.close(fig)
print("wrote", sorted(p.name for p in GEN.iterdir()), sorted(p.name for p in FIG.iterdir()))
print(macros)
