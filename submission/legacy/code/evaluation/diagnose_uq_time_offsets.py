"""Fixed in-fit time-offset diagnostic; no held trajectories or fitting."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/training")]
import train_b210_held_residual_uq as held


def main(point_grid=False):
    torch, dense, native, prefix = held.runtime()
    import numpy as np
    from torch.utils.data import DataLoader
    from models.ensemble_unet2d import EnsembleUNet2d
    import evaluate_b210_transfer as ev
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    ev.TIME_SCORE = 92.03266035232576
    start = time.monotonic()
    source = held.read_bound(Path("/root/autodl-tmp/track1_reflection_exposure_stage_20260909/held20/control/uq/uq_endpoint.json"),
        "96ea76a6504b2b36b31515ccf36642d9450824b2600486f14f9664d38a7044d1")
    assert held.sha256(source["checkpoint_path"]) == source["checkpoint_sha256"]
    payload = torch.load(source["checkpoint_path"], map_location="cpu", weights_only=False)
    model = EnsembleUNet2d(dense.ft12.validate_b210_model_config(payload["model_config"])).float()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    normalizer, report = prefix.normalizer_from_point(payload, torch.device("cuda:0"))
    assert report["sha256"] == source["normalizer_sha256"]
    model.cuda().eval()
    manifest = Path("/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json")
    assert held.sha256(manifest) == ev.MANIFEST_SHA
    files = {f"{nominal}_{angle}.h5" for nominal in (10125, 19050) for angle in (0, 5, 10, 15)}
    rows = [r for r in ev.clean_rows(manifest) if r["file"] in files]
    assert len(rows) == 8 and all(r["file"] in source["point_receipt"]["fit_files"] for r in rows)
    audit = held.RoleAudit(rows)
    trajectories, metadata, _ = native.dr1.load_prefix_rows(Path("/root/autodl-tmp/realpde/data/real"), rows, "fit", audit)
    repro = native.ensemble_train.common.repro
    offsets = (-2, -1, 1, 2) if point_grid else (5, 10, 15)
    base_windows = [repro.Window(i, s) for i, r in enumerate(rows) for s in r["train_starts"][::4]
                    if s + min(0, min(offsets)) >= 0 and s + max(offsets) + 40 <= r["val_boundary"]]
    scorer = ev.import_file(ev.SCORER, "offset_diagnostic_scorer")
    results = {}
    for offset in (0, *offsets):
        windows = [repro.Window(w.trajectory, w.start + offset) for w in base_windows]
        raw = {r["file"]: [] for r in rows}
        efficiency_sum, element_count = 0., 0
        loader = DataLoader(repro.WindowDataset(trajectories, windows), batch_size=16, shuffle=False)
        with torch.inference_mode():
            for batch, (history, truth_tensor) in enumerate(loader):
                assert time.monotonic() - start < 120
                truth = truth_tensor.numpy()
                x, _ = normalizer.preprocess(history.cuda(), truth_tensor.cuda())
                point, lo, hi = model.forward_distribution(x)
                p, lower, upper = [v.cpu().numpy() for v in held.absolute_distribution(point, lo, hi, normalizer)]
                mask = truth[..., :2] != 0
                inside = (truth[..., :2] >= lower[..., :2]) & (truth[..., :2] <= upper[..., :2])
                efficiency_sum += float((inside * np.exp(-(upper[..., :2] - lower[..., :2]) / scorer.SIGMA_GLOBAL))[mask].sum(dtype=np.float64))
                element_count += int(mask.sum())
                labels = [rows[w.trajectory]["file"] for w in windows[batch * 16: batch * 16 + len(history)]]
                for filename in dict.fromkeys(labels):
                    select = np.asarray([v == filename for v in labels])
                    raw[filename].append(ev.metric_state(scorer, p[select], truth[select], lower[select], upper[select]))
        raw = {k: ev.merge_states(v) for k, v in raw.items()}
        results[str(offset)] = {"metrics": ev.finalize(scorer, ev.merge_states(list(raw.values()))),
            "efficiency": efficiency_sum / element_count, "by_trajectory": raw}
    base = results["0"]["metrics"]
    shifted = (ev.finalize(scorer, ev.merge_states([v for o in offsets for v in results[str(o)]['by_trajectory'].values()]))
               if point_grid else {k: sum(results[str(o)]["metrics"][k] for o in offsets) / len(offsets) for k in base})
    delta = {k: shifted[k] - base[k] for k in base}
    point = ev.fixed_time_final({**base, **{k: shifted[k] for k in ev.METRICS[:3]}}) - base["fixed_time_fitted_final"]
    efficiency_delta = sum(results[str(o)]["efficiency"] for o in offsets) / len(offsets) - results["0"]["efficiency"]
    worse = sum(sum(ev.finalize(scorer, results[str(o)]["by_trajectory"][r["file"]])["sps_score"] for o in offsets) / len(offsets) <
        ev.finalize(scorer, results["0"]["by_trajectory"][r["file"]])["sps_score"] for r in rows)
    supported = delta["sps_score"] <= -.20 and efficiency_delta <= -.003 and abs(point) <= .03 and worse >= 5
    if point_grid:
        differences = []
        for row in rows:
            original = ev.finalize(scorer, results['0']['by_trajectory'][row['file']])
            changed = ev.finalize(scorer, ev.merge_states([results[str(o)]['by_trajectory'][row['file']] for o in offsets]))
            differences.append(ev.fixed_time_final({**original, **{k: changed[k] for k in ev.METRICS[:3]}}) - original['fixed_time_fitted_final'])
        worse = sum(v < 0 for v in differences)
        supported = point <= -.02 and worse >= 6
    result = {"results": results, "shifted_minus_grid": delta, "efficiency_delta": efficiency_delta,
        "point_only_fitted_delta": point, "worse_trajectories": worse, "supports_training": supported,
        "access": audit.report(), "base_windows": [{"file": rows[w.trajectory]["file"], "start": w.start} for w in base_windows],
        "source_checkpoint_sha256": source["checkpoint_sha256"], "source_sha256": held.sha256(Path(__file__)),
        "scope": "8 training-prefix trajectories, matched nearby starts; not independent validation", "point_grid": point_grid,
        "gate_target": 'point' if point_grid else 'UQ', "seconds": time.monotonic() - start}
    held.write_json(Path('/root/roy_route_20260910') / ('point_time_offsets.json' if point_grid else 'uq_time_offsets.json'), result)
    print(json.dumps({k: result[k] for k in ("shifted_minus_grid", "efficiency_delta", "point_only_fitted_delta", "worse_trajectories", "supports_training", "seconds")}), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--point-grid', action='store_true')
    main(parser.parse_args().point_grid)
