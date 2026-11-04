"""One fixed deterministic joint fine-tune using a soft fitted-score objective."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import train_b210_held_residual_uq as held

CONTROL_SHAS = {
    15: "e965bca6eb35468be17c1cf5d0cf6cb7d6c61cd81ca1ca161819d97da01fad08",
    20: "96ea76a6504b2b36b31515ccf36642d9450824b2600486f14f9664d38a7044d1",
}


def metric_errors(torch, pred, truth):
    assert pred.shape[1:] == (20, 32, 64, 2)
    norm = lambda x: torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1)
    relative = norm(pred - truth) / norm(truth).clamp_min(1e-8)
    energy = lambda x: .5 * (x - x.mean(1, keepdim=True)).square().mean(1).sum(-1)
    target_energy = energy(truth)
    tke = norm(energy(pred) - target_energy) / norm(target_energy).clamp_min(1e-8)
    probes = list(range(8, 25, 2))
    profiles = []
    for column in (13, 21, 29, 37):
        p = pred[:, :, probes, column].mean(1)
        t = truth[:, :, probes, column].mean(1)
        profiles.append(norm(p - t) / norm(t).clamp_min(1e-8))
    return torch.stack((relative, tke, torch.stack(profiles).mean(0)), 1)


def main(angle, root):
    torch, dense, native, prefix = held.runtime()
    import numpy as np
    from torch.utils.data import DataLoader
    from models.ensemble_unet2d import EnsembleUNet2d
    import evaluate_b210_transfer as ev
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    ev.TIME_SCORE = 92.03266035232576
    begun = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    control_path = Path(f"/root/autodl-tmp/track1_reflection_exposure_stage_20260909/held{angle}/control/uq/uq_endpoint.json")
    source = held.read_bound(control_path, CONTROL_SHAS[angle])
    assert held.sha256(source["checkpoint_path"]) == source["checkpoint_sha256"]
    payload = torch.load(source["checkpoint_path"], map_location="cpu", weights_only=False)
    manifest = Path("/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json")
    assert held.sha256(manifest) == ev.MANIFEST_SHA
    rows = ev.clean_rows(manifest)
    fit, test = [r for r in rows if r["aoa"] != angle], [r for r in rows if r["aoa"] == angle]
    assert [r["file"] for r in fit] == source["point_receipt"]["fit_files"]
    assert source["sealed"] and source["tail_arrays_read"] == 0
    real_dir = Path("/root/autodl-tmp/realpde/data/real")
    audit = held.RoleAudit(fit)
    trajectories, metadata, windows = native.dr1.load_prefix_rows(real_dir, fit, "fit", audit)
    tagged = native.ensemble_train.TrajectoryTaggedDataset(
        native.ensemble_train.common.repro.WindowDataset(trajectories, windows))
    native.ensemble_train.seed_everything(4)
    loader = DataLoader(tagged, batch_size=8, shuffle=True, num_workers=0, pin_memory=True,
                        generator=torch.Generator().manual_seed(4))
    batches = native.ensemble_train.common.repro.cycling(loader)
    scorer = ev.import_file(ev.SCORER, "deterministic_metric_scorer")
    common = native.ensemble_train.common

    def make(state):
        model = EnsembleUNet2d(dense.ft12.validate_b210_model_config(payload["model_config"])).float()
        model.load_state_dict(state, strict=True)
        normalizer, report = prefix.normalizer_from_point(payload, torch.device("cuda:0"))
        assert report["sha256"] == source["normalizer_sha256"]
        return model.cuda().eval(), normalizer

    model, normalizer = make(payload["model_state_dict"])
    parameters = tuple(model.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1600)
    stream = hashlib.sha256()
    for step in range(1, 1601):
        if time.monotonic() - begun > 850:
            raise TimeoutError("deterministic metric training budget")
        history, target, _, indices = next(batches)
        stream.update(indices.numpy().astype("<i8", copy=False).tobytes())
        x, y = normalizer.preprocess(history.cuda(), target.cuda())
        optimizer.zero_grad(set_to_none=True)
        pred, _, quantiles, lower, upper, _, _ = model.forward_uq_training(x)
        physical = common.to_physical_target(pred, normalizer)[..., :2]
        truth = common.to_physical_target(y, normalizer)[..., :2]
        errors = metric_errors(torch, physical, truth)
        scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
        lo, hi = lower * scale, upper * scale
        residual = truth - physical
        tau = lo.new_tensor([.01075, .00550]).view(1, 1, 1, 1, 2) * .035
        inside = torch.sigmoid((residual + lo) / tau) * torch.sigmoid((hi - residual) / tau)
        mask = truth != 0
        factor = ((.5 / (.5 + errors)) * errors.new_tensor([.5, .3, .2])).sum(1)
        efficiency = factor.view(-1, 1, 1, 1, 1) * torch.exp(-(lo + hi) / common.SPS_SIGMA)
        soft_sps = (efficiency * inside)[mask].mean()
        point_scores = 1 / (1 + .5 * errors.mean(0))
        scores = torch.cat((point_scores, point_scores.new_tensor([ev.TIME_SCORE / 100]), soft_sps.view(1)))
        utility = (scores.pow(.6854) * scores.new_tensor([.5, .1, .1, .1, .2])).sum().pow(1 / .6854)
        pinball = native.ensemble_train.residual_quantile_loss(model, pred, quantiles, y, normalizer,
            physical_target=common.to_physical_target(y, normalizer))
        loss = -utility + .005 * pinball
        if step == 1:
            p, t = physical.detach().cpu().numpy(), truth.detach().cpu().numpy()
            reference = np.stack((scorer.rel_l2_per_sample(p, t, 2), scorer.tke_rel_l2_per_sample(p, t, 2),
                                  scorer.mvpe_rel_l2_per_sample(p, t)), 1)
            error_difference = float(np.abs(reference - errors.detach().cpu().numpy()).max())
            hard = (((residual >= -lo) & (residual <= hi)) * efficiency)[mask].mean()
            reference_sps = scorer.aggregate_sps(p, t, 2, (physical-lo).detach().cpu().numpy(),
                                                (physical+hi).detach().cpu().numpy())[0]
            hard_difference = abs(float(hard) - reference_sps)
            gradients = torch.autograd.grad(loss, (pred, lower, upper), retain_graph=True)
            assert max(error_difference, hard_difference) <= 5e-6
            assert all(bool(torch.isfinite(v).all()) and float(v.norm()) > 0 for v in gradients)
            assert all(not module.training for module in model.modules())
            held.write_json(root / "preflight.json", {"error_max_abs": error_difference,
                "hard_sps_abs": hard_difference, "gradient_norms": [float(v.norm()) for v in gradients],
                "dropout_disabled": True, "normalizer_sha256": source["normalizer_sha256"]})
        assert bool(torch.isfinite(loss))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        assert bool(torch.isfinite(norm)) and float(norm) > 0
        optimizer.step(); scheduler.step()
        if step % 200 == 0:
            progress = {"angle": angle, "step": step, "loss": float(loss), "soft_sps": float(soft_sps),
                        "seconds": time.monotonic() - begun}
            held.write_json(root / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    checkpoint = root / "joint_metric1600.pth"
    torch.save({"model_state_dict": state}, checkpoint)
    receipt = {"checkpoint_path": str(checkpoint), "checkpoint_sha256": held.sha256(checkpoint),
        "source_receipt_sha256": CONTROL_SHAS[angle], "source_checkpoint_sha256": source["checkpoint_sha256"],
        "normalizer_sha256": source["normalizer_sha256"], "updates": 1600, "seed": 4,
        "sample_stream_sha256": stream.hexdigest(), "access": audit.report(), "held_aoa": angle,
        "sealed": True, "source_sha256": held.sha256(Path(__file__))}
    held.write_json(root / "receipt.json", receipt)
    del model, optimizer, parameters, pred, quantiles, lower, upper, physical, errors, efficiency, loss
    gc.collect(); torch.cuda.empty_cache()
    results = {}
    for label, weights in (("baseline", payload["model_state_dict"]), ("candidate", state)):
        model, normalizer = make(weights)
        raw = {r["file"]: [] for r in test}
        digests = {k: hashlib.sha256() for k in ("input", "truth")}
        with torch.inference_mode():
            for history, truth, labels in ev.tail_batches(test, real_dir):
                if time.monotonic() - begun > 900:
                    raise TimeoutError("deterministic metric evaluation budget")
                for key, array in (("input", history), ("truth", truth)):
                    digests[key].update(array.tobytes())
                x, _ = normalizer.preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
                point, lo, hi = model.forward_distribution(x)
                arrays = [v.cpu().numpy() for v in held.absolute_distribution(point, lo, hi, normalizer)]
                for filename in dict.fromkeys(labels):
                    select = np.asarray([v == filename for v in labels])
                    raw[filename].append(ev.metric_state(scorer, arrays[0][select], truth[select], arrays[1][select], arrays[2][select]))
        raw = {k: ev.merge_states(v) for k, v in raw.items()}
        results[label] = {"metrics": ev.finalize(scorer, ev.merge_states(list(raw.values()))),
                          "by_trajectory": raw, "hashes": {k: v.hexdigest() for k, v in digests.items()}}
        del model
        gc.collect(); torch.cuda.empty_cache()
    assert results["baseline"]["hashes"] == results["candidate"]["hashes"]
    before, after = [results[k]["metrics"] for k in ("baseline", "candidate")]
    delta = {k: after[k] - before[k] for k in before}
    point_delta = ev.fixed_time_final({**before, **{k: after[k] for k in ev.METRICS[:3]}}) - before["fixed_time_fitted_final"]
    positive = sum(ev.finalize(scorer, results["candidate"]["by_trajectory"][r["file"]])["fixed_time_fitted_final"] >=
        ev.finalize(scorer, results["baseline"]["by_trajectory"][r["file"]])["fixed_time_fitted_final"] for r in test)
    passed = delta["fixed_time_fitted_final"] >= .10 and delta["sps_score"] >= .30 and point_delta >= -.03 and positive >= .6 * len(test)
    result = {"decision": "PASS_NEXT_STAGE_FOLD" if passed else "STOP_DETERMINISTIC_METRIC1600_CONFIG",
        "held_aoa": angle, "delta": delta, "point_only_fitted_delta": point_delta,
        "positive_trajectories": positive, "trajectory_count": len(test), "results": results, "receipt": receipt,
        "seconds": time.monotonic() - begun, "scope": "Real/UQ stage LOAO with shared Real-teacher Sim ancestry; not official gain"}
    held.write_json(root / "RESULT.json", result)
    print(json.dumps({k: result[k] for k in ("decision", "delta", "positive_trajectories", "seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=int, choices=(15, 20), required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.angle, args.root)
    except Exception:
        held.write_json(args.root / "FAILED.json", {"traceback": traceback.format_exc()})
        raise
