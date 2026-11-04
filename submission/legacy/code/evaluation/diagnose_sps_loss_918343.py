"""Fixed descriptive SPS decomposition of the actual 918343 ZIP; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

import evaluate_b210_transfer as ev

ZIP_SHA = "b8938d06c100cc68bb4b7e310de112754a9053b4d2834134cb2b4e6f4b5c2a35"


def main(args):
    begun = time.monotonic()
    assert ev.sha256(args.zip) == ZIP_SHA
    args.output.mkdir()
    runtime = args.output / "runtime"
    with zipfile.ZipFile(args.zip) as archive:
        archive.extractall(runtime)
    sys.path.insert(0, str(runtime))
    submission = ev.import_file(runtime / "submission.py", "diagnostic_submission")
    official = ev.import_file(ev.SCORER, "diagnostic_scoring")
    torch.set_num_threads(4)
    rows = ev.clean_rows(args.manifest)
    angles = {row["file"]: row["aoa"] for row in rows}
    sums, metric_states = {}, []
    hashes = {key: hashlib.sha256() for key in ("input", "truth", "point")}

    def add(name, fields, mask, selector):
        valid = mask[selector]
        if name not in sums:
            sums[name] = np.zeros(6, dtype=np.float64)
        sums[name] += np.asarray([valid.sum()] + [
            np.sum(field[selector], where=valid, dtype=np.float64) for field in fields])

    for x, y, labels in ev.tail_batches(rows, args.real_dir):
        if time.monotonic() - begun > args.budget:
            raise TimeoutError("diagnostic budget")
        output = submission.predict(x)
        pred, lower, upper = (output[k] for k in ("prediction", "lower", "upper"))
        metric_states.append(ev.metric_state(official, pred, y, lower, upper))
        for key, value in (("input", x), ("truth", y), ("point", pred)):
            hashes[key].update(value.tobytes())
        errors = (official.rel_l2_per_sample(pred, y, 2),
                  official.tke_rel_l2_per_sample(pred, y, 2),
                  official.mvpe_rel_l2_per_sample(pred, y))
        accuracy = sum(w * .5 / (.5 + e.astype(np.float64))
                       for w, e in zip((.5, .3, .2), errors))
        assert np.isfinite(accuracy).all()
        accuracy = accuracy[:, None, None, None, None]
        truth = y[..., :2]
        mask = truth != 0
        inside = (truth >= lower[..., :2]) & (truth <= upper[..., :2])
        width_factor = np.exp(-(upper[..., :2] - lower[..., :2]) / official.SIGMA_GLOBAL)
        fields = [np.broadcast_to(1 - accuracy, truth.shape), accuracy * ~inside,
                  accuracy * inside * (1 - width_factor), accuracy * inside * width_factor,
                  inside.astype(np.float64)]
        add("all", fields, mask, (...,))
        for start in (0, 5, 10, 15):
            add(f"horizon_{start+1}_{start+5}", fields, mask,
                (slice(None), slice(start, start+5), slice(None), slice(None), slice(None)))
        for channel, name in enumerate(("u", "v")):
            add(name, fields, mask, (..., channel))
        for aoa in sorted(set(angles.values())):
            selected = np.asarray([angles[label] == aoa for label in labels])
            add(f"aoa_{aoa}", fields, mask, (selected,))
        edges = np.linspace(0, truth.shape[3], 4, dtype=int)
        for region in range(3):
            add(f"x_index_third_{region+1}", fields, mask,
                (slice(None), slice(None), slice(None), slice(edges[region], edges[region+1]), slice(None)))
    metrics = ev.finalize(official, ev.merge_states(metric_states))
    groups = {}
    for name, totals in sums.items():
        n, point_loss, missed, width, score, covered = totals
        assert n > 0
        values = np.asarray([point_loss, missed, width, score]) * 100 / n
        assert abs(values.sum() - 100) < 1e-6, (name, values)
        groups[name] = dict(zip(("point_factor_loss", "missed_coverage_loss",
                                "covered_width_loss", "sps"), values.tolist()))
        groups[name].update(scored_elements=int(n), coverage=float(covered/n))
    assert abs(groups["all"]["sps"] - metrics["sps_score"]) < 1e-4
    report = {"zip_sha256": ZIP_SHA, "split": "clean81/716 tail",
              "metrics": metrics, "groups": groups,
              "hashes": {k: v.hexdigest() for k, v in hashes.items()},
              "seconds": time.monotonic()-begun,
              "scope": "Training-overlap descriptive identity, not attainable headroom or independent gain."}
    (args.output / "diagnosis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"all": groups["all"], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("zip", "manifest", "real-dir", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--budget", type=float, default=300)
    main(p.parse_args())
