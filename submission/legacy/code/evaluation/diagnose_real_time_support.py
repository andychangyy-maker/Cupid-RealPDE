"""Descriptive time-support check; no model fitting or validation claim."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np


def main(args):
    begun = time.monotonic()
    files = json.loads(args.files.read_text())
    rows = []
    for name in files:
        with h5py.File(args.real_dir / name, "r") as h:
            length = h["u"].shape[0]
            blocks = []
            for fraction in (0, .25, .5, .75, 1):
                start = round((length - 40) * fraction)
                z = np.stack([h[k][start:start+40, ::2, ::2] for k in ("u", "v")], -1)
                history, future = z[:20], z[20:]
                den = np.linalg.norm(future)
                fluct = future - future.mean(0, keepdims=True)
                blocks.append({"fraction": fraction, "start": start,
                    "mean_uv": z.mean((0, 1, 2)).tolist(),
                    "rms_uv": np.sqrt(np.mean(z*z, (0, 1, 2))).tolist(),
                    "future_fluctuation_rms": float(np.sqrt(np.mean(fluct*fluct))),
                    "persistence_rel": float(np.linalg.norm(future-history[-1:]) / den),
                    "history_mean_rel": float(np.linalg.norm(future-history.mean(0, keepdims=True)) / den),
                    "adjacent_frame_rms": float(np.sqrt(np.mean(np.diff(z, axis=0)**2)))})
            rows.append({"file": name, "frames": length, "aoa": int(h["aoa"][()]),
                         "re": int(h["re"][()]), "blocks": blocks})
    regular = [r for r in rows if r["frames"] == 868]
    keys = ("future_fluctuation_rms", "persistence_rel", "history_mean_rel", "adjacent_frame_rms")
    summary = {k: [float(np.median([r["blocks"][i][k] for r in regular])) for i in range(5)] for k in keys}
    for k in keys:
        ratios = np.array([r["blocks"][0][k] / np.mean([b[k] for b in r["blocks"][1:]]) for r in regular])
        summary[k+"_early_to_later"] = {"median": float(np.median(ratios)),
            "fraction_above_1.25": float(np.mean(ratios > 1.25)),
            "fraction_below_0.8": float(np.mean(ratios < .8))}
    result = {"scope": "Descriptive public training data, all clean81; no GPU or learned predictor",
        "file_list_sha256": hashlib.sha256(args.files.read_bytes()).hexdigest(),
        "rows": rows, "regular_868_count": len(regular), "summary": summary,
        "seconds": time.monotonic()-begun}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"summary": summary, "seconds": result["seconds"]}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--files", type=Path, required=True)
    p.add_argument("--real-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
