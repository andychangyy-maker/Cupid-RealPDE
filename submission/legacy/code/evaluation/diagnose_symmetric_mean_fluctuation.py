"""A fixed, target-free role composition diagnostic on the held15 screen."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/training")]
import train_b210_held_residual_uq as held


def main():
    torch, dense, native, prefix = held.runtime()
    import numpy as np
    from models.ensemble_unet2d import EnsembleUNet2d
    from models.reflection_point_uq import reflect_normalized
    import evaluate_b210_transfer as ev
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    ev.TIME_SCORE = 92.03266035232576
    start = time.monotonic()
    root = Path("/root/roy_route_20260910/held15")
    control = held.read_bound(Path("/root/autodl-tmp/track1_reflection_exposure_stage_20260909/held15/control/uq/uq_endpoint.json"),
        "e965bca6eb35468be17c1cf5d0cf6cb7d6c61cd81ca1ca161819d97da01fad08")
    candidate = json.loads((root / "point/point_endpoint.json").read_text())
    assert candidate["checkpoint_sha256"] == "534dd706135377186867c1941d7415fdc2649ddfa9d59aa53f77e1295d546403"
    models = []
    for receipt in (control, candidate):
        assert held.sha256(receipt["checkpoint_path"]) == receipt["checkpoint_sha256"]
        payload = torch.load(receipt["checkpoint_path"], map_location="cpu", weights_only=False)
        model = EnsembleUNet2d(dense.ft12.validate_b210_model_config(payload["model_config"])).float()
        model.load_state_dict(payload["model_state_dict"], strict=True)
        normalizer, report = prefix.normalizer_from_point(payload, torch.device("cuda:0"))
        assert report["sha256"] == control["normalizer_sha256"]
        models.append(model.cuda().eval())
    manifest = Path("/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json")
    assert held.sha256(manifest) == ev.MANIFEST_SHA
    rows = [r for r in ev.clean_rows(manifest) if r["aoa"] == 15]
    scorer = ev.import_file(ev.SCORER, "role_diagnostic_scorer")
    names = ("baseline", "single", "tta", "role_single", "role_tta")
    states = {k: {r["file"]: [] for r in rows} for k in names}
    errors = {k: {"mean_squared_sum": 0., "fluctuation_squared_sum": 0.} for k in names}
    hashes = {k: hashlib.sha256() for k in ("input", "truth")}
    with torch.inference_mode():
        for history, truth, labels in ev.tail_batches(rows, Path("/root/autodl-tmp/realpde/data/real")):
            assert time.monotonic() - start < 180
            for k, v in (("input", history), ("truth", truth)):
                hashes[k].update(v.tobytes())
            x, _ = normalizer.preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
            baseline, lo, hi = models[0].forward_distribution(x)
            single = models[1](x)
            reflected = models[1](reflect_normalized(x, normalizer.mean_input, normalizer.std_input))
            tta = .5 * (single + reflect_normalized(reflected, normalizer.mean_target, normalizer.std_target))
            points = {"baseline": baseline, "single": single, "tta": tta,
                "role_single": single - single.mean(1, keepdim=True) + baseline.mean(1, keepdim=True),
                "role_tta": tta - tta.mean(1, keepdim=True) + baseline.mean(1, keepdim=True)}
            for name, point in points.items():
                p, lower, upper = [v.cpu().numpy() for v in held.absolute_distribution(point, lo, hi, normalizer)]
                error = (p[..., :2] - truth[..., :2]).astype(np.float64)
                mean = error.mean(1, keepdims=True)
                errors[name]["mean_squared_sum"] += float((mean ** 2).sum() * error.shape[1])
                errors[name]["fluctuation_squared_sum"] += float(((error - mean) ** 2).sum())
                for filename in dict.fromkeys(labels):
                    mask = np.asarray([v == filename for v in labels])
                    states[name][filename].append(ev.metric_state(scorer, p[mask], truth[mask], lower[mask], upper[mask]))
    states = {k: {f: ev.merge_states(v) for f, v in groups.items()} for k, groups in states.items()}
    scores = {k: ev.finalize(scorer, ev.merge_states(list(v.values()))) for k, v in states.items()}
    delta = {k: {m: v[m] - scores["baseline"][m] for m in v} for k, v in scores.items()}
    positive = {k: sum(ev.finalize(scorer, v)["fixed_time_fitted_final"] >=
        ev.finalize(scorer, states["baseline"][f])["fixed_time_fitted_final"] for f, v in states[k].items())
        for k in ("role_single", "role_tta")}
    passed = [k for k in positive if delta[k]["fixed_time_fitted_final"] >= .15 and
        delta[k]["sps_score"] >= .30 and positive[k] >= .6 * len(rows)]
    held.write_json(root / "role_diagnostic.json", {"scores": scores, "delta": delta, "raw_states": states,
        "error_decomposition": errors, "positive_trajectories": positive, "trajectory_count": len(rows),
        "passed_modes": passed, "hashes": {k: v.hexdigest() for k, v in hashes.items()},
        "checkpoint_sha256": [control["checkpoint_sha256"], candidate["checkpoint_sha256"]],
        "scope": "Stage held15 teacher diagnostic, baseline intervals recentered, no deployable or official gain",
        "seconds": time.monotonic() - start, "source_sha256": held.sha256(Path(__file__))})
    print(json.dumps({"delta": delta, "positive": positive, "passed": passed}), flush=True)


if __name__ == "__main__":
    main()
