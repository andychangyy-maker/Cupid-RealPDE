"""Execute the preregistered two-arm screen and replay the sealed fresh controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import train_b210_held_residual_uq as held


def main(root):
    begun = time.monotonic()
    assets = root / "assets"
    manifest = Path("/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json")
    real = Path("/root/autodl-tmp/realpde/data/real")
    source = Path(__file__).resolve().parents[2]
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "OPENBLAS_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}

    def run(name, command, budget):
        remaining = 3300 - (time.monotonic() - begun)
        assert remaining > 0
        held.write_json(root / "progress.json", {"phase": name, "state": "RUNNING",
                                                "elapsed_seconds": time.monotonic()-begun})
        print("START", name, flush=True)
        with (root / (name + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           env=env, check=True, timeout=min(budget, remaining))
        print("SEALED", name, flush=True)

    rows, split = held.load_split(manifest)
    point_receipt = assets / "point_endpoint.json"
    receipt_sha = "38bf52aa61f3190914a3d257b99fb0c05a5662b9b02a8118c044e7177e66db6c"
    point = held.read_bound(point_receipt, receipt_sha)
    held.validate_pure_start(point["sim_receipt"])
    assert held.sha256(assets / "point.pth") == point["checkpoint_sha256"]
    inventory = json.loads((root / "transfer_manifest.json").read_text())
    for rel, spec in inventory.items():
        assert held.sha256(root / rel) == spec["sha256"], rel
    held.write_json(root / "preflight.json", {"decision": "PASS_ASSET_AND_SPLIT_IDENTITY",
        "split_sha256": split["split_sha256"], "point_sha256": point["checkpoint_sha256"],
        "pure_sim_receipt_sha256": point["sim_receipt_sha256"], "transfer": inventory})
    run("diagnosis", [sys.executable, str(source / "code/evaluation/diagnose_sps_loss_918343.py"),
        "--zip", str(assets / "baseline918343.zip"), "--manifest", str(manifest),
        "--real-dir", str(real), "--output", str(root / "diagnosis"), "--budget", "290"], 300)
    common = [sys.executable, str(Path(__file__).with_name("train_b210_held_residual_uq.py")),
              "--execute", "--clean-manifest", str(manifest), "--real-dir", str(real),
              "--point-receipt", str(point_receipt), "--point-receipt-sha256", receipt_sha,
              "--point-checkpoint", str(assets / "point.pth"), "--initialization", "point"]
    for arm in ("control", "candidate"):
        run("inherit_" + arm, common + ["--phase", arm, "--output-dir", str(root / ("inherit_"+arm)),
                                       "--wall-budget-seconds", "1190"], 1200)
    evaluations = {}
    for initialization in ("inherit", "fresh"):
        receipts = {}
        for arm in ("control", "candidate"):
            if initialization == "inherit":
                path = root / ("inherit_"+arm) / "uq_endpoint.json"
            else:
                item = json.loads((assets / ("fresh_"+arm+"_uq_endpoint.json")).read_text())
                item["checkpoint_path"] = str(assets / ("fresh_"+arm+"_uq3200.pth"))
                path = assets / ("relocated_fresh_"+arm+"_receipt.json")
                held.write_json(path, item)
            receipts[arm] = path
        command = common + ["--phase", "eval", "--output-dir", str(root / (initialization+"_eval")),
                            "--wall-budget-seconds", "280"]
        for arm, path in receipts.items():
            command += ["--"+arm+"-receipt", str(path), "--"+arm+"-receipt-sha256", held.sha256(path)]
        run(initialization+"_eval", command, 290)
        evaluations[initialization] = json.loads((root / (initialization+"_eval") / "decision.json").read_text())
    new, old = evaluations["inherit"], evaluations["fresh"]
    for key in ("input_sha256", "truth_sha256", "point_sha256"):
        assert new["results"]["control"][key] == old["results"]["control"][key], key
    delta = new["candidate_minus_control"]["sps_score"]
    angles = new["aoa_sps_deltas"]
    retained = new["results"]["candidate"]["metrics"]["sps_score"] - old["results"]["candidate"]["metrics"]["sps_score"]
    checks = {"pooled_sps_at_least_0.30": delta >= .30,
              "four_aoa_positive": sum(v > 0 for v in angles.values()) >= 4,
              "worst_aoa_at_least_minus_0.10": min(angles.values()) >= -.10,
              "no_worse_than_fresh_unseen": retained >= 0}
    result = {"decision": "PASS_TO_REPLICATION" if all(checks.values()) else "STOP_RESIDUAL_MATCH_CONFIG",
        "gate_checks": checks, "inherit_unseen_minus_seen_sps": delta, "aoa_sps_deltas": angles,
        "inherit_unseen_minus_fresh_unseen_sps": retained, "evaluations": evaluations,
        "seconds": time.monotonic()-begun,
        "scope": "One independent trajectory holdout, five AoA groups; not five retrained folds or official gain."}
    held.write_json(root / "RESULT.json", result)
    held.write_json(root / "progress.json", {"phase": "complete", "state": result["decision"]})
    print(result["decision"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    try:
        main(args.root)
    except Exception:
        held.write_json(args.root / "FAILED.json", {"traceback": traceback.format_exc()})
        raise
