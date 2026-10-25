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

# categorical slots 1-4 of the reference palette (validated: scripts/validate_palette.js)
C5B, CM4, C5A, COPT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
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
    lines.append(
        f"{N} & {len(rs)} & {fmt(np.mean(o)) if o else '--'} & {cell(m4ok, m4tried)} & {cell(aok, atried)} & "
        f"{fmt(np.mean(h))}\\,$\\pm$\\,{fmt(ci95(h))} & {len(ok)}/{len(rs)} & {np.mean(h) / scale(N):.2f} & "
        f"{np.mean([r['wall_time_s'] for r in rs]):.1f} \\\\"
    )
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
    for label, src in (("Method 4 ($W{=}4096$)", m4[4096]), ("Method 5A", m5a), ("Method 5B", by_id(m5b_sq))):
        got = [(src[i]["htm"], opt[N]["instances"][i]) for i in ids if i in src and src[i]["solved"]]
        tried = [i for i in ids if i in src]
        if not tried:
            entries.append(f"{label} & -- & -- & --")
            continue
        gap = [a - b for a, b in got]
        n_opt = sum(1 for a, b in got if a == b)
        entries.append(f"{label} & {len(got)}/{len(tried)} & {np.mean(gap):.1f} & {n_opt}/{len(got)}")
        tag = {"Method 4 ($W{=}4096$)": "Mfour", "Method 5A": "FiveA", "Method 5B": "FiveB"}[label]
        macro(f"Gap{tag}{'Three' if N == '3' else 'Four'}", f"{np.mean(gap):.1f}")
        macro(f"Opt{tag}{'Three' if N == '3' else 'Four'}", f"{n_opt}/{len(got)}")
    for k, e in enumerate(entries):
        lead = (f"\\multirow{{3}}{{*}}{{{N}$\\times${N}}} & \\multirow{{3}}{{*}}{{{opt[N]['gods_number']}}} & "
                f"\\multirow{{3}}{{*}}{{{opt[N]['mean_random']:.1f}}}") if k == 0 else " & & "
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
    q = [r["qtm"] for r in ok]
    ci = f"\\,$\\pm$\\,{fmt(ci95(h))}" if len(h) > 1 else ""
    lines.append(f"{N} & {len(rs)} & {len(ok)}/{len(rs)} & {fmt(np.mean(h))}{ci} & {fmt(np.mean(q))} & "
                 f"{np.mean(h) / scale(N):.1f} & {fmt(np.mean([r['wall_time_s'] for r in rs]), 0)} \\\\")
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

# ------------------------------------------------------------------ lower bounds
lb_lines = []
for N in (6, 8, 10, 15, 20):
    k = (N // 2 - 1) ** 2
    sq = math.ceil(k * math.log(6) / math.log(2 * N) - 1)
    cu = math.ceil(k * (math.lgamma(25) - 6 * math.lgamma(5)) / math.log(9 * N) - 1)
    s5b = np.mean([r["htm"] for r in sq_by_N.get(N, []) if r["solved"]]) if sq_by_N.get(N) else None
    c5b = np.mean([r["htm"] for r in cu_by_N.get(N, []) if r["solved"]]) if cu_by_N.get(N) else None
    lb_lines.append(f"{N} & {max(sq, 0)} & {fmt(s5b, 0)} & {max(cu, 0):,} & {fmt(c5b, 0)} \\\\")
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
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
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
if opt:
    xs = [int(N) for N in opt]
    ax.plot(xs, [opt[str(N)]["mean_random"] / scale(N) for N in xs], "D", color=COPT, ms=4.5, label="optimal")
ax.set_xlabel("Square size $N$")
ax.set_ylabel("HTM / ($N^2/\\ln N$)")
ax.set_xticks([3, 5, 8, 10, 12, 15, 20])
ax.set_title("(a) N×N×1 Rubik's Square", fontsize=8, loc="left", color=INK)
ax.set_ylim(bottom=0)
style(ax)
ax.legend(loc="upper left", fontsize=7)
ax = axes[1]
Nc = sorted(cu_by_N)
ax.plot(Nc, [np.mean([r["htm"] for r in cu_by_N[N] if r["solved"]]) / scale(N) for N in Nc], "-o",
        color=C5B, lw=1.6, ms=3.5, label="Method 5B")
ax.set_xlabel("Cube size $N$")
ax.set_ylabel("HTM / ($N^2/\\ln N$)")
ax.set_xticks([3, 5, 8, 10, 15, 20])
ax.set_title("(b) N×N×N Rubik's Cube, Method 5B", fontsize=8, loc="left", color=INK)
ax.set_ylim(bottom=0)
style(ax)
fig.tight_layout()
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
