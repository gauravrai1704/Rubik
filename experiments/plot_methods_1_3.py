"""Figures for Methods 1-3 (Sections 4.5, 6.5) from the results files in docs/results/.

    python experiments/plot_methods_1_3.py      # writes docs/figures/*.png
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "docs" / "results", ROOT / "docs" / "figures"
# categorical slots 1-3 of the validated default palette (light surface #fcfcfb)
SERIES = {"Method 2 (one cluster at a time)": "#2a78d6", "Method 3 (batched)": "#eb6834", "Method 5B": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#1d1d1b", "#6b6a63", "#e4e3dd", "#fcfcfb"


def rows(*names):
    out = []
    for n in names:
        p = RES / n
        if p.exists():
            out += [json.loads(line) for line in p.open() if line.strip()]
    return out


def norm_by_n(rs):
    by = defaultdict(list)
    for r in rs:
        if r["solved"]:
            N = r["N"]
            by[N].append(r["htm"] / (N * N / math.log(N)))
    return sorted((N, sum(v) / len(v)) for N, v in by.items())


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel("N", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def scaling():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    data = {
        "Rubik's Square": {"Method 2 (one cluster at a time)": rows("m2_reduction_square_random.jsonl"),
                           "Method 3 (batched)": rows("m3_demaine_square_random.jsonl"),
                           "Method 5B": rows("m5b_square_random.jsonl")},
        "Rubik's Cube": {"Method 2 (one cluster at a time)": rows("m2_reduction_cube_random.jsonl"),
                         "Method 3 (batched)": rows("m3_demaine_cube_random.jsonl"),
                         "Method 5B": rows("m5b_cube_random.jsonl", "m5b_cube_random_15_20.jsonl")},
    }
    for ax, (title, series) in zip(axes, data.items()):
        style(ax, title, "HTM / (N² / ln N)")
        for name, rs in series.items():
            pts = norm_by_n(rs)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=SERIES[name], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
            ax.annotate(name.split(" (")[0], (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=8.5)
        ax.set_xlim(2, 23.5)
        ax.set_xticks([3, 5, 10, 15, 20])
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    fig.suptitle("Solution length relative to Demaine's N²/ln N rate (random states, all solved)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "methods_2_3_scaling.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def ida_success():
    rs = rows("m1_square_random.jsonl")
    by = defaultdict(list)
    for r in rs:
        by[r["N"]].append(r["solved"] and r["info"].get("optimal", False))
    if not by:
        return
    Ns = sorted(by)
    rate = [100 * sum(by[N]) / len(by[N]) for N in Ns]
    fig, ax = plt.subplots(figsize=(5.5, 3.6), facecolor=SURFACE)
    style(ax, "Method 1 IDA*: Square states solved optimally within the budget", "solved (%)")
    bars = ax.bar(Ns, rate, width=0.6, color="#2a78d6")
    for b, v, N in zip(bars, rate, Ns):
        ax.annotate(f"{v:.0f}% ({sum(by[N])}/{len(by[N])})", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points", ha="center", color=INK, fontsize=8.5)
    ax.set_xticks(Ns)
    ax.set_ylim(0, 115)
    fig.tight_layout()
    fig.savefig(OUT / "m1_square_ida_success.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    scaling()
    ida_success()
    print(f"wrote {sorted(p.name for p in OUT.glob('*.png'))}")
