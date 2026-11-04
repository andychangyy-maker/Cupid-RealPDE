"""Fixed held10 test of five spaced UQ history frames against the sealed 0910q fold."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

CONTROL = Path('/root/autodl-tmp/track1_combined_uq_confirm_20260910/held10/uq/uq_endpoint.json')


def main(args):
    sys.path.insert(0, str(args.source_root / 'code/training'))
    import train_b210_held_residual_uq as held
    torch, dense, native, prefix = held.runtime()
    import numpy as np
    import evaluate_b210_transfer as ev
    from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    ev.TIME_SCORE = 92.0326603523
    assert held.sha256(CONTROL) == 'fa835fd611a4ca8fe626de2e9dce4331b83f02456f433f1690950089a69d8234'
    receipts = {'control': json.loads(CONTROL.read_text()),
                'candidate': json.loads((args.root / 'uq/uq_endpoint.json').read_text())}
    control, candidate = receipts.values()
    assert control['checkpoint_sha256'] == '1937033ab6c25806b3960654d3914d3c142371e4665eb73bbcc8e44900dee0df'
    for key in ('point_receipt_sha256', 'point_frozen_sha256', 'normalizer_sha256', 'initial_uq_sha256',
                'sample_stream_sha256', 'sampling_intervention', 'physical_reflection', 'access_audit',
                'updates', 'seed', 'batch_size', 'fit_windows', 'samples_seen'):
        assert candidate[key] == control[key], key
    assert candidate['updates'] == 3200 and candidate['seed'] == 18 and candidate['held_aoa'] == 10
    assert candidate['point_state_bitwise'] and candidate['access_audit']['held_arrays_read'] == 0
    assert candidate['sparse_history']['indices'] == [0, 4, 8, 12, 16]
    rows = ev.clean_rows(args.manifest)
    assert candidate['point_receipt']['fit_files'] == [r['file'] for r in rows if r['aoa'] != 10]
    test = [r for r in rows if r['aoa'] == 10]
    assert len(test) == 18 and sum(len(r['val_starts']) for r in test) == 162
    scorer = ev.import_file(ev.SCORER, 'sparse_history_scorer')
    results = {}
    for label, receipt in receipts.items():
        assert held.sha256(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
        payload = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)
        assert payload['compact_uq'] and payload['iteration'] == 3200
        parent = payload['point_receipt']
        assert parent['checkpoint_sha256'] == '0997ff5851e4ba198c62599e3202c3e56a67095d1b0e6e58219ae5d3501f9bc3'
        assert held.sha256(parent['checkpoint_path']) == parent['checkpoint_sha256']
        point = torch.load(parent['checkpoint_path'], map_location='cpu', weights_only=False)
        expected = {**point['model_config'], **({'uq_sparse_history_context': True} if label == 'candidate' else {})}
        assert payload['model_config'] == expected
        config = dict(payload['model_config']); assert config.pop('architecture') == 'ensemble_unet'
        model = EnsembleUNet2d(EnsembleUNet2dConfig(**config)).float()
        assert set(payload['model_state_dict']) == {k for k in model.state_dict() if k.startswith(native.UQ_PREFIXES)}
        model.load_state_dict({**point['model_state_dict'], **payload['model_state_dict']}, strict=True)
        assert native.dr1.model_state_sha256(native.partition_state(model, uq=False)) == receipt['point_frozen_sha256']
        norm, report = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
        assert report['sha256'] == receipt['normalizer_sha256']
        del point, payload
        model.cuda().eval()
        states = {r['file']: [] for r in test}
        hashes = {k: hashlib.sha256() for k in ('input', 'truth', 'prediction')}
        with torch.inference_mode():
            for history, truth, labels in ev.tail_batches(test, args.real_dir):
                hashes['input'].update(history.tobytes()); hashes['truth'].update(truth.tobytes())
                x, _ = norm.preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
                point, lower, upper = model.forward_distribution(x)
                p, lo, hi = [v.cpu().numpy() for v in held.absolute_distribution(point, lower, upper, norm)]
                hashes['prediction'].update(p.tobytes())
                for file in dict.fromkeys(labels):
                    select = np.asarray([v == file for v in labels])
                    states[file].append(ev.metric_state(scorer, p[select], truth[select], lo[select], hi[select]))
        states = {k: ev.merge_states(v) for k, v in states.items()}
        results[label] = dict(metrics=ev.finalize(scorer, ev.merge_states(list(states.values()))),
                              by_trajectory=states, hashes={k: v.hexdigest() for k, v in hashes.items()})
        del model, norm
        gc.collect(); torch.cuda.empty_cache()
    assert all(results['control']['hashes'][k] == results['candidate']['hashes'][k] for k in ('input', 'truth'))
    old = json.loads((CONTROL.parent.parent / 'RESULT.json').read_text())['results']['candidate']['metrics']
    assert max(abs(results['control']['metrics'][k]-old[k]) for k in (*ev.METRICS[:3], 'sps_score')) <= 1e-5
    before, after = [results[k]['metrics'] for k in ('control', 'candidate')]
    delta = {k: after[k]-before[k] for k in before}
    assert max(abs(delta[k]) for k in ev.METRICS[:3]) <= 1e-5
    trajectories = {f: ev.finalize(scorer, value)['sps_score']-ev.finalize(scorer, results['control']['by_trajectory'][f])['sps_score']
                    for f, value in results['candidate']['by_trajectory'].items()}
    positive = sum(v >= 0 for v in trajectories.values())
    passed = delta['sps_score'] >= .20 and delta['fixed_time_fitted_final'] >= .045 and positive/18 >= .70
    result = dict(decision='PASS_UQ_SPARSE_HISTORY_HELD10' if passed else 'STOP_UQ_SPARSE_HISTORY_HELD10',
                  delta=delta, nonnegative_trajectories=positive, trajectory_deltas=trajectories,
                  results=results, receipts=receipts, source_sha256=held.sha256(__file__),
                  scope='Stage held10 with shared Sim ancestors and reused research fold; fixed Time, no official gain.')
    (args.root / 'RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('decision', 'delta', 'nonnegative_trajectories')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'manifest', 'real-dir', 'source-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
