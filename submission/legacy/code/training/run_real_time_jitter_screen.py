"""Score the sealed Real phase intervention, then conditionally fit matching UQ."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import train_b210_held_residual_uq as held
from run_uq_phase_jitter_stage import CONTROL_SHAS


def main(args):
    torch, dense, native, prefix = held.runtime()
    import numpy as np
    from models.ensemble_unet2d import EnsembleUNet2d
    import evaluate_b210_transfer as ev
    import train_b210_transfer_uq as transfer
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    ev.TIME_SCORE = 92.03266035232576
    begun = time.monotonic()
    manifest = Path('/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json')
    assert held.sha256(manifest) == held.MANIFEST_SHA
    real_dir = Path('/root/autodl-tmp/realpde/data/real')
    rows = ev.clean_rows(manifest)
    fit, test = [r for r in rows if r['aoa'] != args.angle], [r for r in rows if r['aoa'] == args.angle]
    control_path = Path(f'/root/autodl-tmp/track1_reflection_exposure_stage_20260909/held{args.angle}/control/uq/uq_endpoint.json')
    control = held.read_bound(control_path, CONTROL_SHAS[args.angle])
    point_path = args.root / 'point/point_endpoint.json'
    point = json.loads(point_path.read_text())
    transfer.validate_point_receipt(point, 'C', full_real_point=True)
    assert point['held_aoa'] == args.angle and point['fit_files'] == [r['file'] for r in fit]
    prefix_uq_intervention = args.no_input_corruption or args.sam_real
    if args.sam_real:
        assert point['real_optimizer'] == {'name': 'SAM_AdamW', 'rho': .05, 'norm': 'l2',
                                          'matched_torch_rng': True, 'passes_per_update': 2}
        assert not point.get('real_time_jitter') and not point.get('vertical_reflection') and not point.get('real_input_corruption')
    elif args.no_input_corruption:
        assert point['real_input_corruption'] == 'off_after_matching_rng_draws'
        assert point.get('real_time_jitter') is None and point.get('vertical_reflection') is None
    else:
        assert point['real_time_jitter']['max_shift'] == 4
        assert not point['real_time_jitter_sampling']['max_support_overrun']
        assert min(point['real_time_jitter_sampling']['shift_histogram']) > 0
    old_point = control['point_receipt']
    for key in ('start_sim_sha256', 'start_metadata_sha256', 'sample_stream_sha256', 'fit_files', 'optimizer_windows'):
        assert point[key] == old_point[key], key
    scorer = ev.import_file(ev.SCORER, 'real_time_jitter_scorer')

    def evaluate(receipt):
        assert held.sha256(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
        payload = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)
        model = EnsembleUNet2d(dense.ft12.validate_b210_model_config(payload['model_config'])).float()
        model.load_state_dict(payload['model_state_dict'], strict=True)
        normalizer, report = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
        assert report['sha256'] == control['normalizer_sha256']
        model.cuda().eval()
        raw = {r['file']: [] for r in test}
        hashes = {k: hashlib.sha256() for k in ('input', 'truth', 'prediction')}
        with torch.inference_mode():
            for history, truth, labels in ev.tail_batches(test, real_dir):
                if time.monotonic() - begun > 900:
                    raise TimeoutError('sealed point evaluation and UQ budget')
                for key, array in (('input', history), ('truth', truth)):
                    hashes[key].update(array.tobytes())
                x, _ = normalizer.preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
                p, lo, hi = [v.cpu().numpy() for v in held.absolute_distribution(*model.forward_distribution(x), normalizer)]
                hashes['prediction'].update(p.tobytes())
                for name in dict.fromkeys(labels):
                    take = np.asarray([v == name for v in labels])
                    raw[name].append(ev.metric_state(scorer, p[take], truth[take], lo[take], hi[take]))
        raw = {k: ev.merge_states(v) for k, v in raw.items()}
        result = {'metrics': ev.finalize(scorer, ev.merge_states(list(raw.values()))),
                  'by_trajectory': raw, 'hashes': {k: v.hexdigest() for k, v in hashes.items()}}
        del model, payload
        gc.collect(); torch.cuda.empty_cache()
        return result

    def point_delta(result, baseline):
        return ev.fixed_time_final({**baseline, **{k: result[k] for k in ev.METRICS[:3]}}) - baseline['fixed_time_fitted_final']

    baseline, point_result = evaluate(control), evaluate(point)
    assert all(baseline['hashes'][k] == point_result['hashes'][k] for k in ('input', 'truth'))
    point_change = point_delta(point_result['metrics'], baseline['metrics'])
    result = {'held_aoa': args.angle, 'point_only_fitted_delta': point_change, 'baseline': baseline,
              'point_screen': point_result, 'point_receipt_sha256': held.sha256(point_path),
              'control_receipt_sha256': CONTROL_SHAS[args.angle],
              'scope': 'Real/UQ stage LOAO; shared Sim has Real-teacher ancestry; not official gain',
              'source_sha256': held.sha256(Path(__file__))}
    held.write_json(args.root / 'point_screen.json', result)
    print(json.dumps({'point_only_fitted_delta': point_change}), flush=True)
    if point_change < -.03:
        decision = ('STOP_SAM_REAL_POINT_SCREEN' if args.sam_real else
                    'STOP_REAL_CLEAN_INPUT_POINT_SCREEN' if args.no_input_corruption else 'STOP_REAL_TIME_JITTER_POINT_SCREEN')
        result.update(decision=decision, seconds=time.monotonic() - begun)
        held.write_json(args.root / 'RESULT.json', result)
        return
    command = ['--execute', '--label', 'full', '--point-label', 'C', '--initialization', 'point',
        '--full-real-point', '--held-aoa', str(args.angle),
        '--point-receipt', str(point_path), '--point-receipt-sha256', held.sha256(point_path),
        '--clean-manifest', str(manifest), '--real-dir', str(real_dir), '--output-dir', str(args.root / 'uq'),
        '--wall-budget-seconds', '600', '--source-commit', args.source_commit]
    if not prefix_uq_intervention:
        command.extend(('--uq-full-time', '--uq-phase-jitter'))
    held.write_json(args.root / 'uq_arguments.json', command)
    candidate = transfer.execute(transfer.parse_args(command))
    for key in ('normalizer_sha256', 'model_config_sha256'):
        assert candidate[key] == control[key], key
    assert candidate['point_state_bitwise'] and candidate['access_audit']['held_arrays_read'] == 0
    if prefix_uq_intervention:
        assert candidate['tail_arrays_read'] == 0
        assert candidate['fit_windows'] == sum(len(r['train_starts']) for r in fit)
        assert candidate['sample_stream_sha256'] == control['sample_stream_sha256']
    else:
        assert candidate['tail_arrays_read'] == len(fit)
        assert candidate['fit_windows'] == sum((r['frames'] - 40) // 20 + 1 for r in fit)
    after = evaluate(candidate)
    assert after['hashes'] == point_result['hashes']
    delta = {k: after['metrics'][k] - baseline['metrics'][k] for k in baseline['metrics']}
    assert abs(point_delta(after['metrics'], baseline['metrics']) - point_change) < 1e-5
    positive = sum(ev.finalize(scorer, after['by_trajectory'][r['file']])['fixed_time_fitted_final'] >=
        ev.finalize(scorer, baseline['by_trajectory'][r['file']])['fixed_time_fitted_final'] for r in test)
    passed = delta['fixed_time_fitted_final'] >= .075 and point_change >= 0 and delta['sps_score'] >= 0 and positive >= .6 * len(test)
    stop = 'STOP_REAL_UQ_TIME_JITTER_CONFIG'
    if prefix_uq_intervention:
        passed = delta['fixed_time_fitted_final'] >= .10 and point_change >= 0 and delta['sps_score'] >= .20 and positive >= .6 * len(test)
        stop = 'STOP_SAM_REAL_CONFIG' if args.sam_real else 'STOP_REAL_CLEAN_INPUT_CONFIG'
    result.update(decision='PASS_NEXT_STAGE_FOLD' if passed else stop, no_input_corruption=args.no_input_corruption,
        sam_real=args.sam_real,
        candidate=after, delta=delta, positive_trajectories=positive, trajectory_count=len(test),
        candidate_receipt_sha256=held.sha256(args.root / 'uq/uq_endpoint.json'), seconds=time.monotonic() - begun)
    if args.angle == 15 and not prefix_uq_intervention:
        reference_path = args.root.parent / 'fulltime_jitter_stage15/RESULT.json'
        reference = json.loads(reference_path.read_text())['results']
        assert all(reference['baseline']['hashes'][k] == baseline['hashes'][k] for k in ('input', 'truth'))
        result['delta_vs_uq_only_fulltime_jitter'] = {k: after['metrics'][k] - reference['candidate']['metrics'][k] for k in delta}
    held.write_json(args.root / 'RESULT.json', result)
    print(json.dumps({k: result[k] for k in ('decision', 'point_only_fitted_delta', 'delta', 'positive_trajectories', 'seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--angle', type=int, choices=(15, 20), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--no-input-corruption', action='store_true')
    parser.add_argument('--sam-real', action='store_true')
    args = parser.parse_args()
    if args.no_input_corruption and args.angle != 15:
        parser.error('clean-input screen is fixed to held15')
    if args.sam_real and (args.angle != 15 or args.no_input_corruption):
        parser.error('SAM screen is fixed to held15 without clean-input intervention')
    try:
        main(args)
    except Exception:
        held.write_json(args.root / 'SCREEN_FAILED.json', {'traceback': traceback.format_exc()})
        raise
