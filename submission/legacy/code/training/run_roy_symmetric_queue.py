"""Bounded two-stage queue; all output is persisted across client disconnects."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

root = Path("/root/roy_route_20260910")
source = root / "source"
launch = json.loads((root/"held15/launch_retry.json").read_text())
deadline = time.monotonic() + 7200 - (time.time()-launch["started_epoch"])
environment = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}


def save(name, value):
    (root/name).write_text(json.dumps(value, indent=2))


def run(command, log, cap):
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise TimeoutError("two-fold queue budget")
    assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    save("progress.json", {"command": command, "log": str(log), "started_epoch": time.time()})
    with log.open("x") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                       check=True, timeout=min(cap, remaining), env=environment, cwd=source)


try:
    save("progress.json", {"phase": "waiting_for_first_point", "pid": launch["pid"]})
    while True:
        process = subprocess.run(["ps", "-p", str(launch["pid"]), "-o", "stat="], capture_output=True, text=True)
        if process.returncode != 0 or process.stdout.strip().startswith("Z"):
            break
        if time.time()-launch["started_epoch"] > 2120:
            raise TimeoutError("first point did not leave its time cap")
        time.sleep(10)
    assert (root/"held15/point/point_endpoint.json").exists(), "first point failed; inspect point_retry.log"
    results = {}
    modes = ["single", "tta"]
    for angle in (15, 20):
        folder = root/f"held{angle}"
        if angle == 20:
            folder.mkdir()
            command = launch["command"][2:]
            command[command.index("--held-aoa")+1] = "20"
            command[command.index("--output-dir")+1] = str(folder/"point")
            run(command, folder/"point.log", 2100)
        command = [sys.executable, str(source/"code/training/run_symmetric_uq_screen.py"),
                   "--root", str(folder), "--angle", str(angle), "--modes", *modes, "--wall-cap", "2100"]
        run(command, folder/"uq_screen.log", 2200)
        result = json.loads((folder/"RESULT.json").read_text())
        results[str(angle)] = result
        modes = result["passed_modes"]
        save("progress.json", {"completed_angles": list(results), "passed_modes": modes})
        if not modes:
            break
    save("RESULT.json", {"decision": "PASS_TWO_STAGE_SCREENS_NEEDS_REMAINING_FOLDS" if len(results)==2 and modes else "STOP_SYMMETRIC_REAL8000_CONFIG",
         "passed_modes": modes, "folds": results, "finished_epoch": time.time(),
         "official_gain": None, "release_verified": False})
except Exception:
    save("FAILED.json", {"traceback": traceback.format_exc(), "finished_epoch": time.time()})
    raise
