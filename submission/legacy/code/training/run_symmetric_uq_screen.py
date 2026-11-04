"""Sealed Real8000 -> fixed single/paired-point UQ -> stage holdout decision."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import train_b210_held_residual_uq as held

CONTROL_RECEIPTS = {
    15: "e965bca6eb35468be17c1cf5d0cf6cb7d6c61cd81ca1ca161819d97da01fad08",
    20: "96ea76a6504b2b36b31515ccf36642d9450824b2600486f14f9664d38a7044d1",
}


def main(args):
    torch, dense, native, prefix = held.runtime()
    import numpy as np
    from torch.utils.data import DataLoader
    from models.ensemble_unet2d import EnsembleUNet2d
    from models.reflection_point_uq import attach_reflected_point
    import evaluate_b210_transfer as ev
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    ev.TIME_SCORE = 92.03266035232576
    root = args.root
    begun = time.monotonic()
    deadline = begun + args.wall_cap
    manifest = Path("/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json")
    real_dir = Path("/root/autodl-tmp/realpde/data/real")
    assert held.sha256(manifest) == "efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e"
    rows = ev.clean_rows(manifest)
    fit = [r for r in rows if r["aoa"] != args.angle]
    test = [r for r in rows if r["aoa"] == args.angle]
    control_path = Path(f"/root/autodl-tmp/track1_reflection_exposure_stage_20260909/held{args.angle}/control/uq/uq_endpoint.json")
    control_receipt = held.read_bound(control_path, CONTROL_RECEIPTS[args.angle])
    point_path = root / "point/point_endpoint.json"
    point_receipt = json.loads(point_path.read_text())
    expected_updates = 4000 if (args.coordinate_reflection or args.uq_feature_dropout or args.same_start_midpoint or args.batch_mean_score or args.raw_grid_phase) else 8000
    assert point_receipt["updates"] == expected_updates
    assert point_receipt["held_aoa"] == args.angle
    assert point_receipt["fit_files"] == [r["file"] for r in fit]
    if args.raw_grid_phase:
        assert point_receipt['seed'] == 4 and point_receipt.get('vertical_reflection') is None
        assert point_receipt['raw_grid_phase'] == {'source_hw': [64, 128], 'model_hw': [32, 64],
                                                  'nonzero_probability': .25, 'cpu_seed': 29011}
        assert point_receipt['sample_stream_sha256'] == control_receipt['point_receipt']['sample_stream_sha256']
        assert min(point_receipt['raw_grid_phase_sampling']['phase_counts']) > 0
    elif args.batch_mean_score:
        assert point_receipt['seed'] == 4 and point_receipt.get('vertical_reflection') is None
        assert point_receipt['point_score_aggregation'] == 'score_of_batch_mean_errors_point_and_members'
    elif args.same_start_midpoint:
        assert not args.uq_feature_dropout and not args.coordinate_reflection and args.modes == ['single']
        assert point_receipt['seed'] == 5 and point_receipt.get('vertical_reflection') is None
    elif args.uq_feature_dropout:
        assert not args.coordinate_reflection and args.modes == ["single"]
        assert point_receipt.get("vertical_reflection") is None
        assert held.sha256(point_path) == control_receipt["point_receipt_sha256"]
    else:
        assert point_receipt["vertical_reflection"]["probability"] == (.25 if args.coordinate_reflection else .5)
    if args.coordinate_reflection:
        assert args.modes == ["single"]
        assert point_receipt["vertical_reflection"]["velocity_sign"] == "unchanged"
    assert control_receipt["point_receipt"]["fit_files"] == point_receipt["fit_files"]
    for receipt in (control_receipt, point_receipt):
        assert held.sha256(receipt["checkpoint_path"]) == receipt["checkpoint_sha256"]
    base = torch.load(control_receipt["checkpoint_path"], map_location="cpu", weights_only=False)
    candidate = torch.load(point_receipt["checkpoint_path"], map_location="cpu", weights_only=False)
    assert candidate["iteration"] == expected_updates and candidate["phase"] == "real"
    assert base["model_config"] == candidate["model_config"]
    scorer = ev.import_file(ev.SCORER, "symmetric_screen_scorer")

    def make(payload, mode, uq_state=None):
        model = EnsembleUNet2d(dense.ft12.validate_b210_model_config(payload["model_config"])).float()
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if uq_state is not None:
            assert set(uq_state) == set(native.partition_state(model, uq=True))
            model.load_state_dict(uq_state, strict=False)
        normalizer, report = prefix.normalizer_from_point(payload, torch.device("cuda:0"))
        assert report["sha256"] == control_receipt["normalizer_sha256"]
        model = model.cuda().eval()
        if args.uq_feature_dropout:
            model.uq_fusion.register_forward_hook(
                lambda module, inputs, output: torch.nn.functional.dropout2d(
                    output, p=.20, training=module.training))
        if mode == "tta":
            attach_reflected_point(model, normalizer)
        return model, normalizer

    def evaluate(payload, mode, uq_state=None):
        model, normalizer = make(payload, mode, uq_state)
        states = {r["file"]: [] for r in test}
        digests = {k: hashlib.sha256() for k in ("input", "truth")}
        with torch.inference_mode():
            for history, truth, labels in ev.tail_batches(test, real_dir):
                if time.monotonic() > deadline:
                    raise TimeoutError("symmetric screen evaluation budget")
                for key, array in (("input", history), ("truth", truth)):
                    digests[key].update(array.tobytes())
                x, _ = normalizer.preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
                prediction, lower, upper = model.forward_distribution(x)
                prediction, lower, upper = [v.cpu().numpy() for v in held.absolute_distribution(
                    prediction, lower, upper, normalizer)]
                for name in dict.fromkeys(labels):
                    mask = np.asarray([v == name for v in labels])
                    states[name].append(ev.metric_state(scorer, prediction[mask], truth[mask], lower[mask], upper[mask]))
        states = {k: ev.merge_states(v) for k, v in states.items()}
        result = {"metrics": ev.finalize(scorer, ev.merge_states(list(states.values()))),
                  "by_trajectory": states, "hashes": {k: v.hexdigest() for k, v in digests.items()}}
        del model
        gc.collect(); torch.cuda.empty_cache()
        return result

    def point_delta(result, control):
        substituted = {**control, **{k: result[k] for k in ev.METRICS[:3]}}
        return ev.fixed_time_final(substituted) - control["fixed_time_fitted_final"]

    baseline = evaluate(base, "single")
    replica_diagnostic = None
    if args.same_start_midpoint:
        replica_diagnostic = evaluate(candidate, 'single')
        assert replica_diagnostic['hashes'] == baseline['hashes']
        from shared_sim_real_midpoint import materialize
        candidate, point_receipt, point_path = materialize(root, candidate, point_receipt, control_receipt)
    point_results, active = {}, []
    for mode in args.modes:
        result = evaluate(candidate, mode)
        assert result["hashes"] == baseline["hashes"]
        change = point_delta(result["metrics"], baseline["metrics"])
        point_results[mode] = {"point_only_fitted_delta": change,
                              "point_scores": {k: result["metrics"][k] for k in ev.METRICS[:3]}}
        point_passed = change >= -.03
        if args.same_start_midpoint or args.batch_mean_score or args.raw_grid_phase:
            differences = {k: result['metrics'][k] - baseline['metrics'][k] for k in ev.METRICS[:3]}
            point_passed = (change >= .05 and differences['tke_rel_l2_score'] >= .20
                            and differences['rel_l2_score'] >= 0 and differences['mvpe_rel_l2_score'] >= 0)
        if point_passed:
            active.append(mode)
    held.write_json(root/"point_screen.json", {"baseline": baseline, "candidates": point_results,
        "active_modes": active, "scope": "Stage LOAO development screen; point gate only, no new UQ gain claimed"})
    results, seals = {}, {}
    if active:
        audit = held.RoleAudit(fit)
        trajectories, metadata, windows = native.dr1.load_prefix_rows(real_dir, fit, "fit", audit)
        dataset = native.ensemble_train.common.repro.WindowDataset(trajectories, windows)
        tagged = native.ensemble_train.TrajectoryTaggedDataset(dataset)
        for mode in active:
            started = time.monotonic()
            model, normalizer = make(candidate, mode)
            native.uq_train.seed_everything(18)
            frozen = native.dr1.model_state_sha256(native.partition_state(model, uq=False))
            parameters = native.configure_uq_only(model)
            loader = DataLoader(tagged, batch_size=8, shuffle=True, num_workers=0,
                                pin_memory=True, generator=torch.Generator().manual_seed(18))
            batches = native.ensemble_train.common.repro.cycling(loader)
            optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3200)
            stream = hashlib.sha256()
            for step in range(1, 3201):
                if time.monotonic() > min(deadline, started+900):
                    raise TimeoutError("symmetric UQ budget")
                native.set_uq_training_mode(model)
                history, target, _, indices = next(batches)
                stream.update(indices.numpy().astype("<i8", copy=False).tobytes())
                x, y = normalizer.preprocess(history.cuda(), target.cuda())
                optimizer.zero_grad(set_to_none=True)
                prediction, quantiles, extents, logits = native.uq_train.quantile_forward(model, x)
                loss, _ = native.uq_train.uq_objective(native.sharp2_args(), model, prediction, quantiles, y,
                    normalizer, interval_extents=extents, coverage_logits=logits)
                assert bool(torch.isfinite(loss))
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                assert bool(torch.isfinite(norm)) and (step != 1 or float(norm) > 0)
                optimizer.step(); scheduler.step()
                if step % 400 == 0:
                    progress = {"angle": args.angle, "mode": mode, "step": step,
                                "loss": float(loss), "seconds": time.monotonic()-started}
                    held.write_json(root/"uq_progress.json", progress)
                    print(json.dumps(progress), flush=True)
            assert frozen == native.dr1.model_state_sha256(native.partition_state(model, uq=False))
            state = {k: v.detach().cpu().clone() for k, v in native.partition_state(model, uq=True).items()}
            path = root/(mode+"_uq3200.pth")
            torch.save(state, path)
            seals[mode] = {"point_sha256": point_receipt["checkpoint_sha256"], "checkpoint_path": str(path),
                "checkpoint_sha256": held.sha256(path), "frozen_point_sha256": frozen,
                "sample_stream_sha256": stream.hexdigest(), "access": audit.report(),
                "point_receipt_sha256": held.sha256(point_path), "updates": 3200, "seed": 18,
                "mode": mode, "uq_feature_dropout": .20 if args.uq_feature_dropout else 0.,
                "seconds": time.monotonic()-started, "sealed": True}
            if args.uq_feature_dropout:
                assert stream.hexdigest() == control_receipt["sample_stream_sha256"]
            held.write_json(root/(mode+"_uq_receipt.json"), seals[mode])
            del model, parameters, optimizer, prediction, quantiles, extents, logits, loss
            gc.collect(); torch.cuda.empty_cache()
            results[mode] = evaluate(candidate, mode, state)
            del state
    if len(seals) == 2:
        assert seals["single"]["sample_stream_sha256"] == seals["tta"]["sample_stream_sha256"]
    passed, changes = [], {}
    for mode, result in results.items():
        assert result["hashes"] == baseline["hashes"]
        delta = {k: result["metrics"][k]-baseline["metrics"][k] for k in baseline["metrics"]}
        point = point_delta(result["metrics"], baseline["metrics"])
        changes[mode] = {"delta": delta, "point_only_fitted_delta": point}
        final_floor = .09 if args.uq_feature_dropout else (.10 if mode == "single" else .45)
        sps_floor = .40 if args.uq_feature_dropout else .30
        positive = sum(ev.finalize(scorer, result["by_trajectory"][row["file"]])["sps_score"] >=
                       ev.finalize(scorer, baseline["by_trajectory"][row["file"]])["sps_score"] for row in test)
        changes[mode]["positive_trajectories"] = positive
        if args.uq_feature_dropout:
            assert max(abs(delta[k]) for k in ev.METRICS[:3]) <= 1e-5
        passed_gate = delta["fixed_time_fitted_final"] >= final_floor and delta["sps_score"] >= sps_floor and point >= -.03 and (not args.uq_feature_dropout or positive >= .6 * len(test))
        if args.same_start_midpoint or args.batch_mean_score or args.raw_grid_phase:
            positive = sum(ev.finalize(scorer, result['by_trajectory'][row['file']])['fixed_time_fitted_final'] >=
                           ev.finalize(scorer, baseline['by_trajectory'][row['file']])['fixed_time_fitted_final'] for row in test)
            changes[mode]['positive_trajectories'] = positive
            passed_gate = delta['fixed_time_fitted_final'] >= .10 and delta['sps_score'] >= 0 and point >= .05 and positive >= .6 * len(test)
        if passed_gate:
            passed.append(mode)
    stop = ("STOP_RAW_GRID_PHASE_POINT_SCORE" if args.raw_grid_phase else
            "STOP_BATCH_MEAN_POINT_SCORE" if args.batch_mean_score else
            "STOP_SHARED_SIM_REAL_MIDPOINT" if args.same_start_midpoint else
            "STOP_UQ_FEATURE_DROPOUT_CONFIG" if args.uq_feature_dropout else
            "STOP_COORDINATE_REFLECTION_CONFIG" if args.coordinate_reflection else "STOP_SYMMETRIC_REAL8000_CONFIG")
    result = {"decision": "PASS_NEXT_STAGE_FOLD" if passed else stop,
        "coordinate_reflection": args.coordinate_reflection,
        "uq_feature_dropout": .20 if args.uq_feature_dropout else 0.,
        "same_start_midpoint": args.same_start_midpoint, 'replica_diagnostic': replica_diagnostic,
        "batch_mean_score": args.batch_mean_score,
        "raw_grid_phase": args.raw_grid_phase,
        "held_aoa": args.angle, "passed_modes": passed, "point_screen": point_results,
        "baseline": baseline, "results": results, "changes": changes, "seals": seals,
        "point_receipt_sha256": held.sha256(point_path), "seconds": time.monotonic()-begun,
        "scope": "Real/UQ stage LOAO; development fold, shared Real-teacher Sim ancestry; not official gain",
        "source_sha256": held.sha256(Path(__file__))}
    held.write_json(root/"RESULT.json", result)
    print(json.dumps({k: result[k] for k in ("decision", "passed_modes", "changes", "seconds")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--angle", type=int, choices=(15, 20), required=True)
    p.add_argument("--modes", nargs="+", choices=("single", "tta"), default=["single", "tta"])
    p.add_argument("--wall-cap", type=int, default=2100)
    p.add_argument("--coordinate-reflection", action="store_true")
    p.add_argument("--uq-feature-dropout", action="store_true")
    p.add_argument('--same-start-midpoint', action='store_true')
    p.add_argument('--batch-mean-score', action='store_true')
    p.add_argument('--raw-grid-phase', action='store_true')
    args = p.parse_args()
    if args.raw_grid_phase and (args.modes != ['single'] or args.uq_feature_dropout
            or args.coordinate_reflection or args.same_start_midpoint or args.batch_mean_score or args.wall_cap != 1100):
        p.error('raw grid phase is the fixed single-forward 1100-second screen')
    if args.batch_mean_score and (args.modes != ['single'] or args.uq_feature_dropout
            or args.coordinate_reflection or args.same_start_midpoint or args.wall_cap != 1100):
        p.error('batch-mean score is the fixed single-forward 1100-second screen')
    if args.same_start_midpoint and (args.angle != 15 or args.modes != ['single']
            or args.uq_feature_dropout or args.coordinate_reflection or args.wall_cap != 1100):
        p.error('midpoint is the fixed held15 single-forward 1100-second screen')
    try:
        main(args)
    except Exception:
        held.write_json(args.root/"UQ_FAILED.json", {"traceback": traceback.format_exc()})
        raise
