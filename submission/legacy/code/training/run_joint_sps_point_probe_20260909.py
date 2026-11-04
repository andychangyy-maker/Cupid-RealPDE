"""Fixed joint continuation with versus without a direct SPS point gradient."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import train_b210_held_residual_uq as held


def main(root, stage_held=None):
    torch, dense, native, prefix = held.runtime()
    import numpy as np
    from torch.utils.data import DataLoader
    import evaluate_b210_transfer as ev
    import train_c208_real_sim_rehearsal as real_recipe

    begun = time.monotonic()
    root.mkdir()
    parent = root.parent
    args = argparse.Namespace(
        point_receipt=parent / "assets/point_endpoint.json",
        point_receipt_sha256="38bf52aa61f3190914a3d257b99fb0c05a5662b9b02a8118c044e7177e66db6c",
        point_checkpoint=parent / "assets/point.pth", device="cuda:0")
    manifest = Path("/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json")
    real_dir = Path("/root/autodl-tmp/realpde/data/real")
    rows, split = held.load_split(manifest)
    if stage_held is None:
        uq_path = parent / "inherit_control/uq3200.pth"
        assert held.sha256(uq_path) == "b5c2c769084e1db3b0b690deb96072381ea2d25f86eb97dd6d295841acd6ca85"
        uq = torch.load(uq_path, map_location="cpu", weights_only=False)
        assert uq["point_checkpoint_sha256"] == held.sha256(args.point_checkpoint)
        held.validate_pure_start(held.read_bound(args.point_receipt, args.point_receipt_sha256)["sim_receipt"])
        fit_rows = held.filter_rows(rows, split, "point_fit")
    else:
        receipt_shas = {15: "e965bca6eb35468be17c1cf5d0cf6cb7d6c61cd81ca1ca161819d97da01fad08",
                        20: "96ea76a6504b2b36b31515ccf36642d9450824b2600486f14f9664d38a7044d1"}
        path = Path(f"/root/autodl-tmp/track1_reflection_exposure_stage_20260909/held{stage_held}/control/uq/uq_endpoint.json")
        source_receipt = held.read_bound(path, receipt_shas[stage_held])
        assert source_receipt["held_aoa"] == stage_held and source_receipt["sealed"]
        assert source_receipt["tail_arrays_read"] == 0
        assert source_receipt["access_audit"]["held_arrays_read"] == 0
        assert held.sha256(source_receipt["checkpoint_path"]) == source_receipt["checkpoint_sha256"]
        stage_payload = torch.load(source_receipt["checkpoint_path"], map_location="cpu", weights_only=False)
        fit_rows = [row for row in rows if row["aoa"] != stage_held]
        assert [row["file"] for row in fit_rows] == source_receipt["point_receipt"]["fit_files"]
        assert source_receipt["access_audit"]["expected"] == [row["file"] for row in fit_rows]
        held.write_json(root / "source_receipt.json", source_receipt)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    common = native.ensemble_train.common
    initial_hashes, stream_hashes, seals = {}, {}, {}
    audit = held.RoleAudit(fit_rows)
    trajectories, metadata, windows = native.dr1.load_prefix_rows(real_dir, fit_rows, "fit", audit)
    dataset = native.ensemble_train.common.repro.WindowDataset(trajectories, windows)
    tagged = native.ensemble_train.TrajectoryTaggedDataset(dataset)
    objective = native.sharp2_args()
    point_args = real_recipe.real_loss_args()

    def load_start():
        if stage_held is not None:
            from models.ensemble_unet2d import EnsembleUNet2d
            config = dense.ft12.validate_b210_model_config(stage_payload["model_config"])
            model = EnsembleUNet2d(config).float()
            model.load_state_dict(stage_payload["model_state_dict"], strict=True)
            normalizer, report = prefix.normalizer_from_point(stage_payload, torch.device(args.device))
            assert report["sha256"] == source_receipt["normalizer_sha256"]
            return model.to(args.device), normalizer
        receipt, payload, model, normalizer, norm_report = held.load_point(args, split)
        assert norm_report["sha256"] == uq["normalizer_sha256"]
        model.load_state_dict(uq["uq_state_dict"], strict=False)
        assert set(uq["uq_state_dict"]) == set(native.partition_state(model, uq=True))
        return model, normalizer

    def point_sps(pred, lo, hi, physical_truth, normalizer):
        physical_pred = common.to_physical_target(pred, normalizer)[..., :2]
        truth = physical_truth[..., :2]
        scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
        lower, upper = lo.detach()*scale, hi.detach()*scale
        residual = truth - physical_pred
        tau = pred.new_tensor([.01075, .00550]).view(1, 1, 1, 1, 2) * .035
        inside = torch.sigmoid((residual+lower)/tau)*torch.sigmoid((upper-residual)/tau)
        mask = truth != 0
        efficiency = (inside * torch.exp(-(lower+upper)/common.SPS_SIGMA)*mask).sum((1, 2, 3, 4))
        efficiency = efficiency / mask.sum((1, 2, 3, 4)).clamp_min(1)
        factor = common.sample_sps_factor(physical_pred, truth, detach=False)
        default = common.sample_sps_factor(physical_pred, truth)
        assert torch.equal(factor.detach(), default) and not default.requires_grad
        return -(factor*efficiency).mean()

    training_arms = ("control", "candidate") if stage_held is None else ("continuation",)
    for arm in training_arms:
        phase_start = time.monotonic()
        held.write_json(root / "progress.json", {"phase": arm, "step": 0})
        model, normalizer = load_start()
        initial_hashes[arm] = native.dr1.model_state_sha256(model.state_dict())
        parameters = dense.ft12.configure_trainable(model)
        native.ensemble_train.seed_everything(4)
        loader = DataLoader(tagged, batch_size=8, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(4), pin_memory=True)
        batches = native.ensemble_train.common.repro.cycling(loader)
        optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800)
        stream = hashlib.sha256()
        for step in range(1, 801):
            if time.monotonic()-phase_start > 800 or time.monotonic()-begun > 1950:
                raise TimeoutError("joint probe budget")
            model.train()
            history, target, trajectory_ids, indices = next(batches)
            stream.update(indices.numpy().astype("<i8", copy=False).tobytes())
            target = target.to(args.device)
            x, y = normalizer.preprocess(history.to(args.device), target)
            optimizer.zero_grad(set_to_none=True)
            pred, members, quantiles, lo, hi, features, logits = model.forward_uq_training(x)
            point_loss, _ = native.ensemble_train.point_loss(
                point_args, pred, y, normalizer, (5., 3., 2.), 4000+step)
            member_loss, _ = native.ensemble_train.bootstrap_member_point_loss(
                point_args, members, y, trajectory_ids, normalizer, (5., 3., 2.), 4000+step)
            uq_loss, _ = native.uq_train.uq_objective(
                objective, model, pred, quantiles, y, normalizer,
                interval_extents=(lo, hi), coverage_logits=logits)
            loss = .25*point_loss + member_loss + uq_loss
            if arm == "candidate":
                direct = point_sps(pred, lo, hi, target, normalizer)
                if step == 1:
                    gradient = torch.autograd.grad(direct, pred, retain_graph=True)[0]
                    assert torch.isfinite(gradient).all() and float(gradient.norm()) > 0
                    held.write_json(root / "gradient_preflight.json", {
                        "direct_point_gradient_norm": float(gradient.norm()),
                        "default_factor_values_identical_and_detached": True, "weight": .05})
                loss = loss + .05*direct
            assert bool(torch.isfinite(loss))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            assert bool(torch.isfinite(norm))
            optimizer.step(); scheduler.step()
            if step % 200 == 0:
                progress = {"phase": arm, "step": step, "loss": float(loss),
                            "seconds": time.monotonic()-phase_start}
                held.write_json(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        stream_hashes[arm] = stream.hexdigest()
        checkpoint = root / (arm+"_u800.pth")
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"model_state_dict": state}, checkpoint)
        seals[arm] = {"checkpoint_path": str(checkpoint), "checkpoint_sha256": held.sha256(checkpoint),
                      "initial_state_sha256": initial_hashes[arm], "sample_stream_sha256": stream_hashes[arm],
                      "updates": 800, "access": audit.report(), "seconds": time.monotonic()-phase_start}
        held.write_json(root / (arm+"_receipt.json"), seals[arm])
        del model, parameters, optimizer, state, pred, members, quantiles, lo, hi, features, logits, loss
        gc.collect(); torch.cuda.empty_cache()
    assert len(set(initial_hashes.values())) == 1 and len(set(stream_hashes.values())) == 1
    # Both endpoints are sealed before this study opens its held trajectory tails.
    held.write_json(root / "progress.json", {"phase": "evaluation"})
    official = ev.import_file(ev.SCORER, "joint_probe_scorer")
    ev.TIME_SCORE = 92.03266035232576
    held_files = (set(split["roles"]["uq_calibration"]["files"]) if stage_held is None else
                  {row["file"] for row in rows if row["aoa"] == stage_held})
    eval_rows = [r for r in ev.clean_rows(manifest) if r["file"] in held_files]
    results = {}
    for arm in ("start", *training_arms):
        model, normalizer = load_start()
        if arm != "start":
            path = Path(seals[arm]["checkpoint_path"])
            assert held.sha256(path) == seals[arm]["checkpoint_sha256"]
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
        model.eval()
        by_file = {row["file"]: [] for row in eval_rows}
        hashes = {key: hashlib.sha256() for key in ("input", "truth")}
        with torch.inference_mode():
            for inputs, truth, labels in ev.tail_batches(eval_rows, real_dir):
                if time.monotonic()-begun > 1980:
                    raise TimeoutError("joint evaluation budget")
                hashes["input"].update(inputs.tobytes()); hashes["truth"].update(truth.tobytes())
                x, _ = normalizer.preprocess(torch.from_numpy(inputs).to(args.device),
                                            torch.from_numpy(truth).to(args.device))
                pred, lo, hi = model.forward_distribution(x)
                pred, lower, upper = held.absolute_distribution(pred, lo, hi, normalizer)
                pred, lower, upper = (v.cpu().numpy() for v in (pred, lower, upper))
                for name in dict.fromkeys(labels):
                    select = np.asarray([label == name for label in labels])
                    by_file[name].append(ev.metric_state(official, pred[select], truth[select], lower[select], upper[select]))
        by_file = {k: ev.merge_states(v) for k, v in by_file.items()}
        by_aoa = {str(aoa): ev.finalize(official, ev.merge_states([
            by_file[r["file"]] for r in eval_rows if r["aoa"] == aoa])) for aoa in sorted({r["aoa"] for r in eval_rows})}
        results[arm] = {"metrics": ev.finalize(official, ev.merge_states(list(by_file.values()))),
                        "by_trajectory": by_file, "by_aoa": by_aoa,
                        "hashes": {k: v.hexdigest() for k, v in hashes.items()}}
        del model; gc.collect(); torch.cuda.empty_cache()
    assert all(results["start"]["hashes"] == results[arm]["hashes"] for arm in training_arms)
    if stage_held is not None:
        start, after = results["start"]["metrics"], results["continuation"]["metrics"]
        delta = {k: after[k]-start[k] for k in start}
        point_only = ev.fixed_time_final({**start, **{k: after[k] for k in ev.METRICS[:3]}})-start["fixed_time_fitted_final"]
        positive = delta["sps_score"] > 0 and delta["fixed_time_fitted_final"] > 0 and point_only >= -.05
        result = {"decision": "CONTINUE_NEXT_STAGE_FOLD" if positive else "STOP_JOINT_CONTINUATION_STAGE",
                  "held_aoa": stage_held, "continuation_minus_start": delta,
                  "point_only_fitted_final_delta": point_only, "results": results, "seals": seals,
                  "seconds": time.monotonic()-begun,
                  "scope": "Real/UQ stage LOAO only; shared Sim ancestry is not held out."}
        held.write_json(root / "RESULT.json", result)
        held.write_json(root / "progress.json", {"phase": "complete", "decision": result["decision"]})
        print(result["decision"], flush=True)
        return
    deltas = {arm: {k: results["candidate"]["metrics"][k]-results[arm]["metrics"][k]
                    for k in results[arm]["metrics"]} for arm in ("start", "control")}
    angles = {a: results["candidate"]["by_aoa"][a]["sps_score"]-results["control"]["by_aoa"][a]["sps_score"]
              for a in results["control"]["by_aoa"]}
    gate = {"sps_vs_control": deltas["control"]["sps_score"] >= .30,
            "final_vs_control": deltas["control"]["fixed_time_fitted_final"] >= .05,
            "four_positive_aoa": sum(v > 0 for v in angles.values()) >= 4,
            "worst_aoa": min(angles.values()) >= -.10,
            "point_guard": all(deltas["control"][k] >= -.03 for k in ev.METRICS[:3]),
            "sps_vs_start": deltas["start"]["sps_score"] >= .30,
            "final_vs_start": deltas["start"]["fixed_time_fitted_final"] > 0}
    result = {"decision": "PASS_TO_STAGE_LOAO" if all(gate.values()) else "STOP_JOINT_SPS_POINT_CONFIG",
              "gate": gate, "candidate_minus": deltas, "aoa_sps_deltas_vs_control": angles,
              "results": results, "seals": seals, "seconds": time.monotonic()-begun,
              "scope": "Lineage-disjoint trajectory discovery set; no official score or release claim."}
    held.write_json(root / "RESULT.json", result)
    held.write_json(root / "progress.json", {"phase": "complete", "decision": result["decision"]})
    print(result["decision"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage-held", type=int, choices=(15, 20))
    a = p.parse_args()
    try:
        main(a.root, a.stage_held)
    except Exception:
        if a.root.exists():
            held.write_json(a.root / "FAILED.json", {"traceback": traceback.format_exc()})
        raise
