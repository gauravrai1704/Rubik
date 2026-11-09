"""Run the remaining experiments as ordered, resumable batches (well under 100 hours on one 4-core machine).

    KMCODERS_DIR=... NXOPT=... NXOPT_DIR=... nohup python experiments/run_batches.py &

Each batch is a resumable script (it skips puzzles already in its result file); a batch is marked done in
docs/results/batches_done.txt when its script exits normally, so a restart continues with the first unfinished batch.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DONE = ROOT / "docs" / "results" / "batches_done.txt"
PY = sys.executable
BATCHES = [
    ("m7_N5", [PY, "experiments/m7_santa.py", "5"]),           # ~5 h
    ("m7_N6_N7", [PY, "experiments/m7_santa.py", "6", "7"]),   # ~7 h
    ("m7_N4", [PY, "experiments/m7_santa.py", "4"]),           # ~10 h
    ("b2_N10", [PY, "experiments/kmc_one_run.py", "10"]),      # ~3 h
    ("m7_N8_N10", [PY, "experiments/m7_santa.py", "8", "9", "10"]),  # ~17 h
    ("b3_N8_N10", [PY, "experiments/b3_santa.py", "8", "9", "10"]),  # dwalton76; also started by hand
    ("seed_control_N4", [PY, "experiments/m7_seed_control.py"]),  # ~3 h (after the main-table runs)
    # N = 19 left out of the paper's runs (cost: tens of hours per configuration)
]

# do not run two campaigns at once: wait for one started by hand to finish
while subprocess.run(["pgrep", "-f", "experiments/m7_santa.py"], capture_output=True).returncode == 0:
    import time

    time.sleep(60)

done = set(DONE.read_text().split()) if DONE.exists() else set()
for name, cmd in BATCHES:
    if name in done:
        continue
    print("batch", name, flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print("batch", name, "failed with", r.returncode, flush=True)
        sys.exit(r.returncode)
    with DONE.open("a") as fh:
        fh.write(name + "\n")
print("all batches done", flush=True)
