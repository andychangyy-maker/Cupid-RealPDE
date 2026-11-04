"""Run and score one preregistered point-frozen UQ time-jitter stage fold."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import train_b210_held_residual_uq as held

CONTROL_SHAS = {
    0: "0bc806a6f80a3abc666a7bd7d4ebd3afe1bad3fc8cdb1e1f62400635ceed347e",
    5: "9b8d42dc7e09c0de872106d5fcf54864c78bffdc883535da52c5d590fc5da53f",
    10: "3baea2b929f2d339cbc95a14f2b2a09e604f7fab153724048bf6840609deb8d7",
    15: "e965bca6eb35468be17c1cf5d0cf6cb7d6c61cd81ca1ca161819d97da01fad08",
    20: "96ea76a6504b2b36b31515ccf36642d9450824b2600486f14f9664d38a7044d1",
}


def main(args):
    torch, dense, native, prefix = held.runtime()
    import numpy as np
    from models.ensemble_unet2d import EnsembleUNet2d
    import train_b210_transfer_uq as transfer
    import evaluate_b210_transfer as ev
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    ev.TIME_SCORE = 92.03266035232576
    begun = time.monotonic()
    args.root.mkdir(parents=True, exist_ok=False)
    path = Path(f"/root/autodl-tmp/track1_reflection_exposure_stage_20260909/held{args.angle}/control/uq/uq_endpoint.json")
    source = held.read_bound(path, CONTROL_SHAS[args.angle])
    batch8 = None
    if args.batch_size == 64 or args.uq_wide320:
        batch8 = held.read_bound(args.root.parent / 'fulltime_jitter_stage15/uq/uq_endpoint.json',
            '5934dfcb248851fa1dd3a562463bcffcf19bd3653c586bea2f2bd3b056d14673')
    jitter = None
    if args.uq_reflection_combination and not args.confirmation_cohort:
        jitter = held.read_bound(Path('/root/roy_route_20260910/fulltime_jitter_stage15/uq/uq_endpoint.json'),
            '5934dfcb248851fa1dd3a562463bcffcf19bd3653c586bea2f2bd3b056d14673')
        reflection_path = Path('/root/track1_quality_20260910/assets/uq_reflection_uq3200.pth')
        assert held.sha256(reflection_path) == 'e75ea764e0f1cbbd78b31a9b93a2d850600b8260a71ef5529651feee0e8c89fb'
    manifest = Path("/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json")
    real_dir = Path("/root/autodl-tmp/realpde/data/real")
    command = ["--execute", "--label", "full", "--point-label", "C", "--initialization", "point",
        "--full-real-point", "--held-aoa", str(args.angle), "--uq-phase-jitter",
        "--point-receipt", str(path.parent.parent / "point/point_endpoint.json"),
        "--point-receipt-sha256", source["point_receipt_sha256"], "--clean-manifest", str(manifest),
        "--real-dir", str(real_dir), "--output-dir", str(args.root / "uq"),
        "--wall-budget-seconds", str(args.train_cap_seconds), "--source-commit", args.source_commit]
    command.extend(('--batch-size', str(args.batch_size)))
    if args.uq_wide320:
        command.append('--uq-wide320')
    if args.full_time:
        command.append('--uq-full-time')
    if args.uq_reflection_combination:
        command.append('--uq-physical-reflection')
        if args.confirmation_cohort:
            command.append('--compact-uq')
    held.write_json(args.root / "training_arguments.json", command)
    candidate = transfer.execute(transfer.parse_args(command))
    if args.batch_size == 64:
        assert candidate['samples_seen'] == 200164 and candidate['batch_size'] == 64
        assert candidate['sample_stream_sha256'] == '8f8c3dfd05bc6355c12ebf22144cb67a6f0e7f8b1badadfe84a03a4852f3971e'
        assert candidate['sampling_intervention']['actual_sampling_sha256'] == 'a0ecf83b238f864f071cde671ae1eee0cceddaa27f65f6a3598ce1c00837afdc'
    common = ['point_receipt_sha256', 'normalizer_sha256', 'point_frozen_sha256']
    if args.uq_wide320:
        assert candidate['source_initial_uq_sha256'] == source['initial_uq_sha256'] == batch8['initial_uq_sha256']
        assert candidate['sample_stream_sha256'] == batch8['sample_stream_sha256']
        assert candidate['sampling_intervention']['actual_sampling_sha256'] == batch8['sampling_intervention']['actual_sampling_sha256']
        assert candidate['widening']['new_width'] == 320 and candidate['widening']['initial_function']['point_bitwise']
    else:
        common += ['initial_uq_sha256', 'model_config_sha256']
    for key in common:
        assert candidate[key] == source[key], key
        if batch8 is not None:
            assert candidate[key] == batch8[key], key
        if jitter is not None:
            assert candidate[key] == jitter[key], key
    if jitter is not None:
        assert candidate['sample_stream_sha256'] == jitter['sample_stream_sha256']
        assert candidate['sampling_intervention'] == jitter['sampling_intervention']
        assert candidate['physical_reflection']['draws'] == candidate['samples_seen']
    if args.full_time:
        fit_rows = [r for r in ev.clean_rows(manifest) if r['aoa'] != args.angle]
        assert candidate['fit_windows'] == sum((r['frames'] - 40) // 20 + 1 for r in fit_rows)
        assert candidate['tail_arrays_read'] == len(fit_rows)
        assert candidate['sampling_intervention']['support'] == 'full_fit_trajectories'
    else:
        assert candidate['sample_stream_sha256'] == source['sample_stream_sha256']
        assert candidate['tail_arrays_read'] == 0
    assert candidate['access_audit']['held_arrays_read'] == 0 and candidate["point_state_bitwise"]
    assert candidate["access_audit"]["optimizer_stride"] is None
    gc.collect(); torch.cuda.empty_cache()
    rows = [r for r in ev.clean_rows(manifest) if r["aoa"] == args.angle]
    scorer = ev.import_file(ev.SCORER, "uq_jitter_stage_scorer")
    results = {}
    receipts = [('baseline', source), *([('batch8', batch8)] if batch8 is not None else []), ('candidate', candidate)]
    if jitter is not None:
        receipts = [('baseline', source), ('jitter_only', jitter), ('reflection_only', source), ('candidate', candidate)]
    for label, receipt in receipts:
        assert held.sha256(receipt["checkpoint_path"]) == receipt["checkpoint_sha256"]
        payload = torch.load(receipt["checkpoint_path"], map_location="cpu", weights_only=False)
        expected = replace(dense.ft12.expected_b210_config(), uq_hidden_channels=320) if args.uq_wide320 and label == 'candidate' else None
        model = EnsembleUNet2d(dense.ft12.validate_b210_model_config(payload["model_config"], expected)).float()
        if payload.get('compact_uq', False):
            parent = payload['point_receipt']
            assert held.sha256(parent['checkpoint_path']) == parent['checkpoint_sha256']
            point_payload = torch.load(parent['checkpoint_path'], map_location='cpu', weights_only=False)
            assert point_payload['model_config'] == payload['model_config']
            assert set(payload['model_state_dict']) == {k for k in model.state_dict() if k.startswith(native.UQ_PREFIXES)}
            model.load_state_dict({**point_payload['model_state_dict'], **payload['model_state_dict']}, strict=True)
            del point_payload
        else:
            model.load_state_dict(payload["model_state_dict"], strict=True)
        if label == 'reflection_only':
            patch = torch.load(reflection_path, map_location='cpu', weights_only=False)
            assert patch['point_checkpoint_sha256'] == source['point_receipt']['checkpoint_sha256']
            assert patch['normalizer_sha256'] == source['normalizer_sha256']
            assert set(patch['uq_state_dict']) == {k for k in model.state_dict() if k.startswith(native.UQ_PREFIXES)}
            merged = {**payload['model_state_dict'], **patch['uq_state_dict']}
            model.load_state_dict(merged, strict=True)
            del patch, merged
        normalizer, report = prefix.normalizer_from_point(payload, torch.device("cuda:0"))
        assert report["sha256"] == source["normalizer_sha256"]
        model.cuda().eval()
        raw = {r["file"]: [] for r in rows}
        hashes = {k: hashlib.sha256() for k in ("input", "truth", "prediction")}
        with torch.inference_mode():
            for history, truth, labels in ev.tail_batches(rows, real_dir):
                if time.monotonic() - begun > args.train_cap_seconds + (300 if jitter is not None else 240 if batch8 is not None else 180):
                    raise TimeoutError("UQ phase jitter stage budget")
                for key, array in (("input", history), ("truth", truth)):
                    hashes[key].update(array.tobytes())
                x, _ = normalizer.preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
                point, lower, upper = model.forward_distribution(x)
                p, lo, hi = [v.cpu().numpy() for v in held.absolute_distribution(point, lower, upper, normalizer)]
                hashes["prediction"].update(p.tobytes())
                for filename in dict.fromkeys(labels):
                    select = np.asarray([v == filename for v in labels])
                    raw[filename].append(ev.metric_state(scorer, p[select], truth[select], lo[select], hi[select]))
        raw = {k: ev.merge_states(v) for k, v in raw.items()}
        results[label] = {"metrics": ev.finalize(scorer, ev.merge_states(list(raw.values()))),
            "by_trajectory": raw, "hashes": {k: v.hexdigest() for k, v in hashes.items()}}
        del model, payload
        gc.collect(); torch.cuda.empty_cache()
    assert all(results["baseline"]["hashes"][k] == results["candidate"]["hashes"][k] for k in ("input", "truth"))
    if args.uq_reflection_combination:
        assert results['baseline']['hashes'] == results['candidate']['hashes']
    before, after = [results[k]["metrics"] for k in ("baseline", "candidate")]
    delta = {k: after[k] - before[k] for k in before}
    point_difference = max(abs(delta[k]) for k in ev.METRICS[:3])
    assert point_difference <= 1e-5
    positive = sum(ev.finalize(scorer, results["candidate"]["by_trajectory"][r["file"]])["fixed_time_fitted_final"] >=
        ev.finalize(scorer, results["baseline"]["by_trajectory"][r["file"]])["fixed_time_fitted_final"] for r in rows)
    passed = delta["sps_score"] >= .30 and delta["fixed_time_fitted_final"] >= .07 and positive >= .6 * len(rows)
    stop = 'STOP_UQ_FULLTIME_JITTER_CONFIG' if args.full_time else 'STOP_UQ_PHASE_JITTER_CONFIG'
    decision = 'COHORT_FOLD_RECORDED' if args.confirmation_cohort else ('PASS_NEXT_STAGE_FOLD' if passed else stop)
    exposure = {}
    if jitter is not None:
        assert all(v['hashes'] == results['baseline']['hashes'] for v in results.values())
        better_single = max(results[k]['metrics']['sps_score'] for k in ('jitter_only', 'reflection_only'))
        over_single = after['sps_score'] - better_single
        passed = (delta['sps_score'] >= .40 and delta['fixed_time_fitted_final'] >= .09
                  and over_single >= .10 and positive >= .7 * len(rows))
        decision = 'PASS_COMBINED_UQ_NEXT_STAGE' if passed else 'STOP_COMBINED_UQ_QUALITY_CONFIG'
        exposure = {'uq_reflection_combination': True, 'sps_over_better_single': over_single,
                    'gate': {'sps_delta_min': .40, 'fixed_time_final_delta_min': .09,
                             'sps_over_better_single_min': .10, 'positive_fraction_min': .7}}
    if batch8 is not None:
        assert results['batch8']['hashes'] == results['candidate']['hashes']
        relative = {k: after[k] - results['batch8']['metrics'][k] for k in before}
        batch_positive = sum(ev.finalize(scorer, results['candidate']['by_trajectory'][r['file']])['fixed_time_fitted_final'] >=
            ev.finalize(scorer, results['batch8']['by_trajectory'][r['file']])['fixed_time_fitted_final'] for r in rows)
        passed = relative['fixed_time_fitted_final'] >= .04 and delta['fixed_time_fitted_final'] >= .09 and batch_positive >= .6 * len(rows)
        decision = 'PASS_B64_NEXT_STAGE_FOLD' if passed else 'STOP_B64_EXPOSURE_CONFIG'
        if args.uq_wide320:
            passed = relative['fixed_time_fitted_final'] >= .20 and relative['sps_score'] >= .60 and batch_positive >= .6 * len(rows)
            decision = 'PASS_WIDE320_TIMING_SCREEN' if passed else 'STOP_WIDE320_CAPACITY_CONFIG'
        exposure = {'batch_size': args.batch_size, 'uq_wide320': args.uq_wide320,
                    'delta_vs_batch8': relative, 'positive_vs_batch8': batch_positive}
    result = {"decision": decision, "held_aoa": args.angle, 'full_time': args.full_time,
        'uq_reflection_combination': args.uq_reflection_combination,
        'confirmation_cohort': args.confirmation_cohort,
        **exposure,
        "delta": delta, "positive_trajectories": positive, "trajectory_count": len(rows), "point_score_max_abs": point_difference,
        "results": results, "candidate_receipt_sha256": held.sha256(args.root / "uq/uq_endpoint.json"),
        "control_receipt_sha256": CONTROL_SHAS[args.angle], "seconds": time.monotonic() - begun,
        "scope": "Real/UQ stage LOAO, shared Sim has Real-teacher ancestry; no official-gain claim",
        "source_sha256": held.sha256(Path(__file__))}
    held.write_json(args.root / "RESULT.json", result)
    shown = ('decision', 'seconds') if args.confirmation_cohort else ('decision', 'delta', 'positive_trajectories', 'seconds')
    print(json.dumps({k: result[k] for k in shown}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=int, choices=(0, 5, 10, 15, 20), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument('--full-time', action='store_true')
    parser.add_argument('--confirmation-cohort', action='store_true')
    parser.add_argument('--train-cap-seconds', type=int, choices=(600, 900, 1800, 2400), default=600)
    parser.add_argument('--batch-size', type=int, choices=(8, 64), default=8)
    parser.add_argument('--uq-wide320', action='store_true')
    parser.add_argument('--uq-reflection-combination', action='store_true')
    args = parser.parse_args()
    if args.uq_reflection_combination:
        if not args.full_time or args.batch_size != 8 or args.uq_wide320 or args.train_cap_seconds != 900:
            parser.error('reflection combination requires fixed B8 full-time 900-second training')
        if (args.confirmation_cohort and args.angle == 15) or (not args.confirmation_cohort and args.angle != 15):
            parser.error('held15 is discovery only; confirmation uses the separate 0/5/10/20 cohort')
    if args.batch_size == 64 and (args.angle != 15 or not args.full_time or args.confirmation_cohort or args.train_cap_seconds != 2400):
        parser.error('B64 is fixed to the preregistered held15 full-time 2400-second screen')
    if args.batch_size == 8 and args.train_cap_seconds == 2400:
        parser.error('the longer budget is reserved for B64')
    if args.uq_wide320 and (args.angle != 15 or not args.full_time or args.confirmation_cohort or args.batch_size != 8 or args.train_cap_seconds != 1800):
        parser.error('wide320 is fixed to the held15 full-time B8 1800-second screen')
    if not args.uq_wide320 and args.train_cap_seconds == 1800:
        parser.error('1800 seconds is reserved for wide320')
    if args.angle in (0, 5, 10) and not (args.confirmation_cohort and args.full_time):
        parser.error('new 0/5/10 folds require fixed full-time confirmation cohort')
    try:
        main(args)
    except Exception:
        held.write_json(args.root / "FAILED.json", {"traceback": traceback.format_exc()})
        raise
