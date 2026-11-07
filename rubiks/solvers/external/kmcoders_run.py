"""KMCoders' own Santa 2023 cube solver (github.com/wata-orz/santa2023_permutation_puzzle, Rust), run on
any cube state as a subprocess and re-verified by our harness.

Unlike ``santa_kmcoders`` (their published, best-of-many-runs submission, Santa instances only), this
runs their program once on the state we give it, so it is a baseline on our random datasets too.

Setup (once): clone the repository, ``cargo build --release --bin solve_cube --bin solve_cube_beam`` in
``solution/`` and point ``KMCODERS_DIR`` at the checkout. The state is written in their input format
(Santa sticker order and move permutations from ``puzzle_info.csv``, which our Santa adapter maps to the
engine); the answer is Santa move names, converted back to engine quarter turns.

Their program caches per-puzzle intermediate results under ``solution/cache/`` keyed by puzzle id; each
call uses a fresh id and removes its cache files afterwards. For odd N their stage 1 reads an optimal
3x3x3 solution from ``cache/corner/<id>.txt`` when one exists (in the competition it came from an
external optimal solver; otherwise it calls dwalton76's solver, i.e. Kociemba, from a fixed path). We
always write that file: ``stage1="kociemba"`` (default, the program's own fallback) or ``"optimal"``.
"""
from __future__ import annotations

import itertools
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from ...bench import santa2023
from ...engine import PuzzleSpec
from ..base import Budget, Solver, SolveResult, register

_ids = itertools.count(int(time.time() * 1000) % 10**6 * 1000 + os.getpid() % 1000)


def santa_input(P: PuzzleSpec, state: np.ndarray, pid: int) -> str:
    """Our state in KMCoders' input format."""
    N = P.N
    m = santa2023.santa_map(N)
    _, info, _ = santa2023._tables()
    # their program indexes moves as axis * N + layer with axes in the order d, f, r (as in their files)
    gens = sorted(((k, v) for k, v in info[f"cube_{N}/{N}/{N}"].items() if not k.startswith("-")),
                  key=lambda kv: ("dfr".index(kv[0][0]), int(kv[0][1:])))
    S = 6 * N * N
    # their solver expects face k of the goal to have colour k (Santa sticker order: face blocks of N*N)
    target = P.solved[m.to_engine]
    relabel = np.empty(int(P.solved.max()) + 1, dtype=np.int64)
    relabel[target[::N * N]] = np.arange(6)
    start = relabel[np.asarray(state)[m.to_engine]]
    target = relabel[target]
    lines = [f"{pid} cube_{N}/{N}/{N} {S} {len(gens)} 0"]
    lines += [f"{k} " + " ".join(map(str, v)) for k, v in gens]
    lines += [" ".join(map(str, target.tolist())), " ".join(map(str, start.tolist()))]
    return "\n".join(lines) + "\n"


class _Run:
    def __init__(self, stdout, stderr, returncode, cpu_s):
        self.stdout, self.stderr, self.returncode, self.cpu_s = stdout, stderr, returncode, cpu_s


def _run_with_rusage(cmd, stdin_text, cwd, env, timeout) -> _Run:
    """Run a process and return its own CPU time (user + system, all threads), exact even when several
    runs share the machine: the child is reaped with ``os.wait4``, which reports that child's usage."""
    import tempfile

    with tempfile.TemporaryFile("w+") as fi, tempfile.TemporaryFile("w+") as fo, tempfile.TemporaryFile("w+") as fe:
        fi.write(stdin_text)
        fi.seek(0)
        p = subprocess.Popen(cmd, stdin=fi, stdout=fo, stderr=fe, cwd=cwd, env=env, text=True)
        t_end = time.time() + timeout
        while True:
            pid, status, ru = os.wait4(p.pid, os.WNOHANG)
            if pid:
                break
            if time.time() > t_end:
                p.kill()
                pid, status, ru = os.wait4(p.pid, 0)
                break
            time.sleep(0.05)
        p.returncode = os.waitstatus_to_exitcode(status)
        fo.seek(0)
        fe.seek(0)
        return _Run(fo.read(), fe.read(), p.returncode, round(ru.ru_utime + ru.ru_stime, 2))


@register("kmcoders")
class KMCodersRun(Solver):
    def __init__(self, puzzle: PuzzleSpec, budget: Budget | None = None, root: str | None = None,
                 binary: str = "solve_cube", stage1: str = "kociemba", seed: int = 0,
                 sa_iters: int | None = None, nxopt_timeout: float = 180.0, **kwargs):
        """``seed`` (XORed into their fixed seeds; 0 = their program unchanged) and ``sa_iters`` (stage-2
        annealing iterations, theirs: 20M) need the two-line patch ``kmcoders_env.patch``."""
        super().__init__(puzzle, budget)
        if puzzle.kind != "cube":
            raise ValueError("KMCoders' solver is for cubes")
        self.root = Path(root or os.environ.get("KMCODERS_DIR", "")) / "solution"
        self.exe = self.root / "target" / "release" / binary
        if not self.exe.exists():
            raise FileNotFoundError("set KMCODERS_DIR to a checkout with solution/target/release/" + binary)
        if stage1 not in ("kociemba", "nxopt", "optimal"):
            raise ValueError("stage1: 'kociemba', 'nxopt' (optimal QTM, Rokicki's solver) or 'optimal' (Method 1)")
        self.binary, self.stage1 = binary, stage1
        self.nxopt_timeout = nxopt_timeout
        self.env = dict(os.environ)
        if seed:
            self.env["KMC_SEED"] = str(seed)
        if sa_iters:
            self.env["KMC_TL"] = str(int(sa_iters))
        self.map = santa2023.santa_map(puzzle.N)

    def _write_corner_cache(self, state, pid) -> int:
        """Stage 1 for odd N: middle-slice turns that bring the fixed centers home, then the 3x3x3 made of
        corners, middle edges and centers: Kociemba's two-phase algorithm (what their program falls back
        to, through dwalton76's solver) or an optimal QTM solution (our Method 1; slow)."""
        from ...engine import cube
        from ...engine.puzzle import Move
        from ..cluster_tools import exact_orientation_moves

        P, N = self.puzzle, self.puzzle.N
        pre = exact_orientation_moves(P, state)
        s = P.apply(np.asarray(state), pre)
        if self.stage1 in ("kociemba", "nxopt"):
            from ..three_stage import solve_3x3_stage

            koc = solve_3x3_stage(P, s)
            if self.stage1 == "nxopt":  # Kociemba's answer only defines the position; nxopt solves it optimally
                from . import nxopt

                small = [Move(m.axis, 0 if m.layer == 0 else 2, m.turn) for m in koc]
                try:  # nxopt needs minutes on a few hard positions: then keep Kociemba's answer for this copy
                    koc = [Move(m.axis, 0 if m.layer == 0 else N - 1, m.turn)
                           for m in nxopt.optimal_from_solution(small, self.nxopt_timeout)]
                except (TimeoutError, RuntimeError):
                    self.last_stage1 = "kociemba (nxopt timed out)"
            big = pre + koc
        else:
            from ..m1_idastar.cube3 import ida_star

            P3 = cube(3)
            lay = np.array([0, N // 2, N - 1])
            look = {(tuple(int(v) for v in l), tuple(int(v) for v in n)): i
                    for i, (l, n) in enumerate(zip(P.sticker_layer, P.sticker_normal))}
            idx = [look[(tuple(int(v) for v in lay[l]), tuple(int(v) for v in n))]
                   for l, n in zip(P3.sticker_layer, P3.sticker_normal)]
            moves, _ = ida_star(s[idx], "qtm", 10**12, self.budget.wall_time_s)
            if moves is None:
                raise RuntimeError("optimal 3x3x3 stage hit its time limit")
            big = pre + [Move(m.axis, int(lay[m.layer]), m.turn) for m in (P3.moves[i] for i in moves)]
        text = self.map.moves_to_santa(big)
        (self.root / "cache" / "corner").mkdir(parents=True, exist_ok=True)
        (self.root / "cache" / "corner" / f"{pid:04d}.txt").write_text(text)
        return len(text.split(".")) if text else 0

    def solve(self, state: np.ndarray) -> SolveResult:
        t0 = time.time()
        pid = next(_ids)
        info = {"binary": self.binary, "seed": int(self.env.get("KMC_SEED", 0)),
                "sa_iters": int(self.env.get("KMC_TL", 20_000_000))}
        try:
            if self.puzzle.N % 2 == 1 and self.puzzle.N > 3:
                self.last_stage1 = self.stage1
                info["stage1_qtm"] = self._write_corner_cache(state, pid)
                info["stage1"] = self.last_stage1
            r = _run_with_rusage([str(self.exe)], santa_input(self.puzzle, state, pid), self.root, self.env,
                                 self.budget.wall_time_s)
            info["cpu_s"] = r.cpu_s
            log_dir = os.environ.get("KMCODERS_LOG_DIR")
            if log_dir:  # keep their progress log (per-iteration length / misplaced stickers) for analysis
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                (Path(log_dir) / f"N{self.puzzle.N}_{pid}.log").write_text(r.stderr)
                info["log"] = f"N{self.puzzle.N}_{pid}.log"
        finally:
            for f in itertools.chain((self.root / "cache").glob(f"*/{pid:04d}.txt"),
                                     (self.root / "cache").glob(f"*/{pid:04d}_*")):
                f.unlink(missing_ok=True)
        out = [x for x in r.stdout.splitlines() if x.strip()]
        if r.returncode != 0 or not out:
            raise RuntimeError("KMCoders' solver failed: " + r.stderr[-300:])
        text = out[-1].strip()  # the program prints improving solutions; the last is its answer
        import re

        sa = re.findall(r"^0\.99: (-?[\d.]+)$", r.stderr, re.M)  # stage-2 annealing objective at its end
        first = re.search(r"^[\d.]+: len = (\d+), diff = (\d+)", r.stderr, re.M)  # state entering stage 3
        if sa:
            info["stage2_score"] = float(sa[-1])
        if first:
            info["stage3_start"] = {"len": int(first.group(1)), "diff": int(first.group(2))}
        return SolveResult(moves=self.map.moves_to_engine(text), wall_time_s=time.time() - t0, info=info)
