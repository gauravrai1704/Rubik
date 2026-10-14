"""Benchmark harness (technical basis, Sections 3.6 and 10).

Runs a registered solver on a dataset, re-applies every returned solution with the
engine, and writes one results row per instance. A solver never marks itself solved.
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from ..engine import make, metrics
from ..solvers import Budget, get_solver
from .datasets import Instance

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=Path(__file__).parent).decode().strip()
    except Exception:
        return "unknown"


def hardware() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"{platform.node()} / {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    return f"{platform.node()} / CPU {platform.processor() or platform.machine()}"


def run(solver_name: str, instances: list[Instance], budget: Budget, out_path: Path | None = None,
        solver_kwargs: dict | None = None, verbose: bool = True) -> list[dict]:
    rows = []
    commit, hw = git_commit(), hardware()
    solvers: dict[tuple, object] = {}
    fh = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(out_path, "w")
    try:
        for ins in instances:
            P = make(ins.puzzle, ins.N)
            key = (ins.puzzle, ins.N)
            if key not in solvers:
                solvers[key] = get_solver(solver_name, P, budget, **(solver_kwargs or {}))
            solver = solvers[key]
            t0 = time.time()
            error = None
            try:
                res = solver.solve(ins.state.copy())
                moves = res.moves
            except NotImplementedError:
                raise
            except Exception as e:  # a crash counts as a failure, never hidden
                res, moves, error = None, [], f"{type(e).__name__}: {e}"
            wall = time.time() - t0
            solved = bool(P.is_solved(P.apply(ins.state, moves))) if error is None else False
            m = metrics(moves, P)
            row = {
                "id": ins.id, "solver": solver_name, "puzzle": ins.puzzle, "N": ins.N, "depth": ins.depth,
                "solved": solved, **m, "wall_time_s": round(wall, 4),
                "nodes": getattr(res, "nodes_expanded", 0) if res else 0,
                "peak_gpu_mb": round(getattr(res, "peak_gpu_mb", 0.0), 1) if res else 0.0,
                "peak_host_mb": round(getattr(res, "peak_host_mb", 0.0), 1) if res else 0.0,
                "optimal_len": ins.optimal_len, "error": error, "info": getattr(res, "info", {}) if res else {},
                "commit": commit, "hardware": hw, "seed": ins.seed,
            }
            rows.append(row)
            if fh:
                fh.write(json.dumps(row, default=str) + "\n")
                fh.flush()
            if verbose:
                flag = "ok " if solved else "FAIL"
                print(f"[{flag}] {ins.id:28s} htm={m['htm']:5d} qtm={m['qtm']:5d} t={wall:7.2f}s"
                      + (f"  {error}" if error else ""))
    finally:
        if fh:
            fh.close()
    return rows


def _ci95(x: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    return 1.96 * float(np.std(x, ddof=1)) / math.sqrt(len(x))


def summarize(rows: list[dict]) -> list[dict]:
    """Per (solver, puzzle, N): success rate, mean ± 95% CI and max length, time.

    Failed instances count as failed and are excluded only from length statistics,
    never from the success rate (Section 10.5)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["solver"], r["puzzle"], r["N"]), []).append(r)
    out = []
    for (s, pz, N), rs in sorted(groups.items()):
        ok = [r for r in rs if r["solved"]]
        row = {"solver": s, "puzzle": pz, "N": N, "n": len(rs), "success": len(ok) / len(rs)}
        for k in ("htm", "qtm", "stm"):
            v = [r[k] for r in ok]
            row[f"{k}_mean"] = float(np.mean(v)) if v else float("nan")
            row[f"{k}_ci95"] = _ci95(v)
            row[f"{k}_max"] = max(v) if v else None
        row["norm_htm"] = row["htm_mean"] / (N * N / math.log(N)) if ok else float("nan")
        row["time_mean_s"] = float(np.mean([r["wall_time_s"] for r in rs]))
        out.append(row)
    return out


def print_summary(summary: list[dict]) -> None:
    print(f"{'solver':14s} {'puzzle':6s} {'N':>3s} {'n':>4s} {'succ':>6s} {'HTM mean±ci':>16s} {'QTM':>8s} "
          f"{'HTM/(N²/lnN)':>13s} {'time s':>8s}")
    for r in summary:
        print(f"{r['solver']:14s} {r['puzzle']:6s} {r['N']:3d} {r['n']:4d} {r['success']:6.0%} "
              f"{r['htm_mean']:9.1f}±{r['htm_ci95']:<6.1f} {r['qtm_mean']:8.1f} {r['norm_htm']:13.2f} "
              f"{r['time_mean_s']:8.2f}")
