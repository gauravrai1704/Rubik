"""Santa 2023 / KMCoders / Method 7 / counting-bound part of the paper's tables, figures and numbers.

Run by make_tables.py (``build(macro)`` before numbers.tex is written, ``figures(plt, style)`` after).
Reads paper/results/santa/*.jsonl and paper/results/stats/*.jsonl (copies of docs/results/...).
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

INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#9a9993", "#e6e5e1"
CM2, CM3, C5C, CM7 = "#e87ba4", "#008300", "#6250d6", "#e34948"  # validated in this legend order


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def fmt(x, nd=1):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{nd}f}"


def word(n: int) -> str:
    return ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"][n] if n <= 10 else str(n)


S = RES / "santa"
pub = {r["id"]: r for r in rows(S / "kmcoders.jsonl")}
one = {}
for N in range(4, 11):
    for r in rows(S / f"kmcoders_program_N{N}.jsonl"):
        one[r["id"]] = r
meth = {"dwalton76": rows(S / "dwalton76.jsonl"), "M6": rows(S / "m6_kmcoders.jsonl"),
        "5C": rows(S / "m5c_schedule.jsonl"), "M3": rows(S / "m3_demaine.jsonl"), "M2": rows(S / "m2_reduction.jsonl")}
meth = {k: {r["id"]: r for r in v} for k, v in meth.items()}
M7 = {"N4_48": {r["id"]: r for r in rows(S / "m7_beam_sa1M_48copies_N4.jsonl")},
      "N4_8": {r["id"]: r for r in rows(S / "m7_default_8copies_N4.jsonl")},
      "N5_24": {r["id"]: r for r in rows(S / "m7_beam_sa1M_24copies_N5.jsonl")},
      "N5_24nx": {r["id"]: r for r in rows(S / "m7_beam_sa1M_24copies_nxopt_N5.jsonl")}}

NS = [4, 5, 6, 7, 8, 9, 10, 19, 33]


def bound(N):
    from rubiks.analysis.counting_bound import bound_cube_qtm

    return bound_cube_qtm(N)


def mean_on(src: dict, ids):
    v = [src[i]["qtm"] for i in ids if i in src and src[i]["solved"]]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def build(macro):
    # ---------------------------------------------------------------- Santa table (all methods, mean QTM)
    lines = []
    for N in NS:
        ids = sorted(i for i, r in pub.items() if r["N"] == N)
        p, n = mean_on(pub, ids)
        b = bound(N)
        cells = [str(N), str(n), fmt(b, 0), f"\\textbf{{{fmt(p)}}}"]
        o, no = mean_on(one, ids)
        cells.append(fmt(o) + (f"$^{{{no}}}$" if 0 < no < n else ""))
        for k in ("dwalton76", "M6", "5C", "M3", "M2"):
            m, nm = mean_on(meth[k], ids)
            cells.append(fmt(m) + (f"$^{{{nm}}}$" if 0 < nm < n else ""))
        cells.append(fmt(p / b, 2) if p else "--")
        lines.append(" & ".join(cells) + " \\\\")
    (GEN / "tab_santa_rows.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
    macro("SantaTotal", sum(1 for r in pub.values() if r["N"] >= 2))

    # ratios used in the text
    for N, tag in ((4, "Four"), (5, "Five"), (6, "Six"), (7, "Seven"), (10, "Ten"), (19, "Nineteen"), (33, "ThirtyThree")):
        ids = [i for i, r in pub.items() if r["N"] == N]
        p, _ = mean_on(pub, ids)
        macro(f"Bound{tag}", fmt(bound(N), 1))
        macro(f"Pub{tag}", fmt(p, 1))
        macro(f"PubOverBound{tag}", fmt(p / bound(N), 2))
        m6, _ = mean_on(meth["M6"], ids)
        if m6:
            macro(f"MsixOverPub{tag}", fmt(m6 / p, 2))
        o, no = mean_on(one, ids)
        if o:
            macro(f"OneOverPub{tag}", fmt(100 * (o / p - 1), 0))
            macro(f"One{tag}", fmt(o, 1))
    for key, tag in (("5C", "FiveC"), ("M6", "Msix")):
        rs = []
        for N in (4, 5, 6, 7, 8, 9, 10, 19):
            ids = [i for i, r in pub.items() if r["N"] == N]
            m, n = mean_on(meth[key], ids)
            if m and n == len(ids):
                rs.append(m / mean_on(pub, ids)[0])
        macro(f"{tag}OverPubMin", fmt(min(rs), 1))
        macro(f"{tag}OverPubMax", fmt(max(rs), 1))
    # Method 7 with nxopt on N = 5 (first puzzles)
    nx = sorted(M7["N5_24nx"].values(), key=lambda r: r["id"])
    macro("MsevenNxoptDone", len(nx))
    if nx:
        macro("MsevenNxoptFirst", nx[0]["qtm"])
        macro("MsevenNxoptFirstPub", pub[nx[0]["id"]]["qtm"])
        macro("MsevenNxoptMean", fmt(np.mean([r["qtm"] for r in nx]), 1))
        macro("MsevenNxoptPubMean", fmt(np.mean([pub[r["id"]]["qtm"] for r in nx]), 1))
    # n^2 constants of the published lengths
    for N, tag in ((10, "Ten"), (19, "Nineteen"), (33, "ThirtyThree")):
        ids = [i for i, r in pub.items() if r["N"] == N]
        macro(f"PubPerNsq{tag}", fmt(mean_on(pub, ids)[0] / N ** 2, 2))

    # ---------------------------------------------------------------- reproduction table
    rep = []
    for N in (4, 5, 6, 7, 8, 9):
        ids = sorted(i for i, r in pub.items() if r["N"] == N)
        p, _ = mean_on(pub, ids)
        o, no = mean_on(one, ids)
        m6, _ = mean_on(meth["M6"], ids)
        rep.append(f"{N} & {fmt(p)} & {fmt(o)} ({no}) & {fmt(o / p if o else None, 2)} & {fmt(m6)} & "
                   f"{fmt(m6 / p if m6 else None, 2)} \\\\")
    (GEN / "tab_repro_kmc_rows.tex").write_text("\n".join(rep) + "\n\\bottomrule\n")

    # ---------------------------------------------------------------- Method 7 per puzzle
    lines, wins = [], defaultdict(lambda: [0, 0, 0])
    sums = defaultdict(list)
    for key, N in (("N4", 4), ("N5", 5)):
        ids = sorted(set(M7["N4_48"] if N == 4 else list(M7["N5_24"]) + list(M7["N5_24nx"])))
        for i in ids:
            p = pub[i]["qtm"]
            if N == 4:
                a, b8 = M7["N4_48"].get(i), M7["N4_8"].get(i)
                best = min(x["qtm"] for x in (a, b8) if x)
                cells = [f"\\Cu{{{N}}}", i.split("-")[-1], str(p), fmt(a["qtm"] if a else None, 0),
                         fmt(b8["qtm"] if b8 else None, 0), "--", f"{best}", fmt(one[i]["qtm"] if i in one else None, 0)]
            else:
                a, nx = M7["N5_24"].get(i), M7["N5_24nx"].get(i)
                best = nx["qtm"] if nx else (a["qtm"] if a else None)
                cells = [f"\\Cu{{{N}}}", i.split("-")[-1], str(p), fmt(a["qtm"] if a else None, 0), "--",
                         fmt(nx["qtm"] if nx else None, 0), fmt(best, 0), fmt(one[i]["qtm"] if i in one else None, 0)]
            if best is None:
                continue
            col = 6
            cells[col] = f"\\textbf{{{best}}}" if best < p else cells[col]
            lines.append(" & ".join(cells) + " \\\\")
            wins[N][0 if best < p else (1 if best == p else 2)] += 1
            sums[N].append((p, best, one[i]["qtm"] if i in one else None))
    (GEN / "tab_m7_rows.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")
    for N, tag in ((4, "Four"), (5, "Five")):
        if not sums[N]:
            continue
        w, t, l = wins[N]
        macro(f"MsevenWins{tag}", w)
        macro(f"MsevenTies{tag}", t)
        macro(f"MsevenLosses{tag}", l)
        macro(f"MsevenN{tag}", len(sums[N]))
        P = np.mean([x[0] for x in sums[N]])
        B = np.mean([x[1] for x in sums[N]])
        macro(f"MsevenPub{tag}", fmt(P, 1))
        macro(f"MsevenBest{tag}", fmt(B, 1))
        macro(f"MsevenGain{tag}", fmt(100 * (1 - B / P), 1))
        ones = [x[2] for x in sums[N] if x[2] is not None]
        if ones:
            macro(f"MsevenOne{tag}", fmt(np.mean(ones), 1))
    # N = 4 portfolio components
    a = list(M7["N4_48"].values())
    if a:
        macro("MsevenFortyEightMeanFour", fmt(np.mean([r["qtm"] for r in a]), 1))
        macro("MsevenEightMeanFour", fmt(np.mean([r["qtm"] for r in M7["N4_8"].values()]), 1))
        cpu = [sum(x or 0 for x in r["cpu_s"]) / 60 for r in a]
        macro("MsevenCpuMinFour", f"{min(cpu):.0f}--{max(cpu):.0f}")
        allc = [c for r in a for c in r["copies"] if c]
        macro("MsevenCopyMinFour", min(allc))
        macro("MsevenCopyMaxFour", max(allc))
        macro("MsevenCopyMeanFour", fmt(np.mean(allc), 1))
        spreads = [max(x for x in r["copies"] if x) - min(x for x in r["copies"] if x) for r in a]
        macro("MsevenSpreadMinFour", min(spreads))
        macro("MsevenSpreadMaxFour", max(spreads))
    b = list(M7["N5_24nx"].values()) or list(M7["N5_24"].values())
    if b:
        cpu = [sum(x or 0 for x in r["cpu_s"]) / 60 for r in b]
        macro("MsevenCpuMinFive", f"{min(cpu):.0f}--{max(cpu):.0f}")
    k5 = [r["qtm"] for r in M7["N5_24"].values()]
    if k5:
        macro("MsevenKociembaFive", fmt(np.mean(k5), 1))

    # ---------------------------------------------------------------- counting bound table (Square calibration)
    from rubiks.analysis.counting_bound import exact_word_bound, square_block_coeffs

    cal = []
    for N, states, gods in ((3, 768, 6), (4, 82944, 10), (5, 10616832, 12)):
        bnd = exact_word_bound(math.log(states / 4), square_block_coeffs(N), 2)
        cal.append(f"{N} & {states:,} & {bnd} & {gods} & {gods / bnd:.2f} \\\\")
    (GEN / "tab_cal_rows.tex").write_text("\n".join(cal) + "\n\\bottomrule\n")

    # ---------------------------------------------------------------- 20-state CIs (random states)
    st = RES / "stats"
    ci = []
    for N in range(3, 11):
        cells = [str(N)]
        vals = {}
        for k, f in (("M2", "m2_reduction_cube20.jsonl"), ("M3", "m3_demaine_cube20.jsonl"),
                     ("5C", "m5c_schedule_cube20.jsonl")):
            rs = {r["id"]: r for r in rows(st / f) if r["N"] == N and r["solved"]}
            vals[k] = rs
            v = [r["htm"] for r in rs.values()]
            cells.append(f"{np.mean(v):,.1f} $\\pm$ {1.96 * np.std(v, ddof=1) / math.sqrt(len(v)):.1f}" if len(v) > 1 else "--")
        common = sorted(set(vals["M3"]) & set(vals["5C"]))
        d = [vals["5C"][i]["htm"] - vals["M3"][i]["htm"] for i in common]
        if len(d) > 1:
            m, h = np.mean(d), 1.96 * np.std(d, ddof=1) / math.sqrt(len(d))
            rel = 100 * m / np.mean([vals["M3"][i]["htm"] for i in common])
            sig = "$^\\ast$" if m + h < 0 or m - h > 0 else ""
            cells.append(f"{m:+.1f} $\\pm$ {h:.1f} ({rel:+.1f}\\%){sig}")
        else:
            cells.append("--")
        cells.append(str(len(common)))
        ci.append(" & ".join(cells) + " \\\\")
    (GEN / "tab_ci_rows.tex").write_text("\n".join(ci) + "\n\\bottomrule\n")


def build_more(macro):
    # ---------------------------------------------------------------- time per Santa puzzle (wall s, mean)
    lines = []
    for N in (4, 5, 6, 7, 8, 9, 10, 19):
        ids = [i for i, r in pub.items() if r["N"] == N]
        cells = [str(N)]
        for src in (one, meth["dwalton76"], meth["M6"], meth["5C"], meth["M3"], meth["M2"]):
            v = [src[i]["wall_time_s"] for i in ids if i in src and src[i]["solved"]]
            cells.append(fmt(np.mean(v), 0) if v else "--")
        lines.append(" & ".join(cells) + " \\\\")
    (GEN / "tab_santa_time_rows.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")

    # ---------------------------------------------------------------- Method 4: CPU- vs GPU-trained Square models
    ab = rows(RES / "m4_square_cpu_vs_gpu_large.jsonl")
    by = defaultdict(dict)
    for r in ab:
        by[(r["N"], r["id"])][r["model"]] = r
    lines = []
    for N in range(3, 9):
        pairs = [v for (n, _), v in by.items() if n == N and len(v) == 2 and all(x["solved"] for x in v.values())]
        c = [p["cpu"]["htm"] for p in pairs]
        g = [p["gpu_large"]["htm"] for p in pairs]
        sh = sum(b < a for a, b in zip(c, g))
        lo = sum(b > a for a, b in zip(c, g))
        opt = [p["cpu"]["optimal_len"] for p in pairs if p["cpu"].get("optimal_len") is not None]
        lines.append(f"{N} & {len(pairs)} & {fmt(np.mean(opt), 2) if len(opt) == len(pairs) else '--'} & "
                     f"{np.mean(c):.2f} & {np.mean(g):.2f} & {sh} / {lo} \\\\")
        if N == 8:
            macro("MfourGpuEightCpu", f"{np.mean(c):.1f}")
            macro("MfourGpuEightGpu", f"{np.mean(g):.1f}")
            macro("MfourGpuEightShorter", sh)
    (GEN / "tab_m4_gpu_rows.tex").write_text("\n".join(lines) + "\n\\bottomrule\n")


def figures(plt, style):
    # ---------------------------------------------------------------- (1) Santa: published lengths and methods over the bound
    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    series = [("KMCoders published", pub, INK, "-", "D", 1.8), ("KMCoders program, one run", one, INK2, "--", "o", 1.2),
              ("dwalton76", meth["dwalton76"], INK3, ":", "s", 1.2), ("Method 6 (our reproduction)", meth["M6"], INK2, "-.", "x", 1.0),
              ("Method 2", meth["M2"], CM2, "-", "v", 1.2), ("Method 3", meth["M3"], CM3, "-", "P", 1.2),
              ("Method 5C", meth["5C"], C5C, "-", "*", 1.8)]
    for label, src, col, ls, mk, lw in series:
        xs, ys = [], []
        for N in NS:
            ids = [i for i, r in pub.items() if r["N"] == N]
            m, n = mean_on(src, ids)
            if m and n == len(ids):
                xs.append(N)
                ys.append(m / bound(N))
        if xs:
            ax.plot(xs, ys, ls, marker=mk, color=col, lw=lw, ms=3.6, label=label)
    m7x, m7y = [], []
    for N, key in ((4, "N4_48"), (5, "N5_24nx")):
        src = dict(M7[key])
        if N == 4:
            for i, r in M7["N4_8"].items():
                if i in src and r["qtm"] < src[i]["qtm"]:
                    src[i] = r
        if src:
            m7x.append(N)
            m7y.append(np.mean([r["qtm"] for r in src.values()]) / bound(N))
    if m7x:
        ax.plot(m7x, m7y, "o", color=CM7, ms=6, mec="white", mew=1.0, label="Method 7 (subsets)", zorder=5)
    ax.set_xscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels([str(N) for N in NS])
    ax.minorticks_off()
    ax.set_xlabel("cube size $N$ (Santa 2023 puzzles)")
    ax.set_ylabel("mean QTM / counting bound")
    ax.set_ylim(bottom=0)
    style(ax)
    ax.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=2.6,
              columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "santa_ratio.pdf")
    plt.close(fig)

    # ---------------------------------------------------------------- (2) Method 7: copy lengths vs published, N = 4
    a = M7["N4_48"]
    if a:
        ids = sorted(a)
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        for k, i in enumerate(ids):
            c = [x for x in a[i]["copies"] if x]
            jitter = (np.random.default_rng(k).random(len(c)) - 0.5) * 0.36
            ax.plot(k + jitter, c, "o", ms=2.4, color=CM7, alpha=0.45, mew=0)
            ax.plot([k - 0.3, k + 0.3], [pub[i]["qtm"]] * 2, "-", color=INK, lw=1.6)
            if i in one:
                ax.plot([k], [one[i]["qtm"]], "x", color=INK2, ms=4.5, mew=1.0)
        ax.plot([], [], "o", color=CM7, ms=3, label="one copy (48 per puzzle)")
        ax.plot([], [], "-", color=INK, lw=1.6, label="KMCoders published")
        ax.plot([], [], "x", color=INK2, ms=4.5, label="their program, one run")
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels([str(int(i.split("-")[-1])) for i in ids], rotation=0, fontsize=6)
        ax.set_xlabel("Santa 2023 4×4×4 puzzle")
        ax.set_ylabel("QTM")
        style(ax)
        ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.6,
                  columnspacing=0.9)
        fig.tight_layout()
        fig.savefig(FIG / "m7_copies.pdf")
        plt.close(fig)

    # ---------------------------------------------------------------- (3) best of k copies (portfolio curve), N = 4
    if a:
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        ks = [1, 2, 4, 8, 16, 24, 32, 48]
        rng = np.random.default_rng(0)
        ys = []
        for k in ks:
            v = []
            for i in a:
                c = np.array([x for x in a[i]["copies"] if x])
                v.append(np.mean([c[rng.choice(len(c), k, replace=False)].min() for _ in range(200)]))
            ys.append(np.mean(v))
        ax.plot(ks, ys, "-o", color=CM7, lw=1.8, ms=3.6, label="Method 7, best of $k$ copies")
        P = np.mean([pub[i]["qtm"] for i in a])
        ax.axhline(P, color=INK, lw=1.2, ls="--")
        ax.annotate("KMCoders published", (ks[0], P), xytext=(2, -9), textcoords="offset points", ha="left",
                    fontsize=6.5, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
        ax.set_xlabel("copies $k$ (rotations and mirror images)")
        ax.set_ylabel("mean QTM, 10 puzzles")
        style(ax)
        fig.tight_layout()
        fig.savefig(FIG / "m7_portfolio.pdf")
        plt.close(fig)
