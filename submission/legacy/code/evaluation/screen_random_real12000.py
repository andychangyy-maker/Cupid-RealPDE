"""Fixed held15 comparison of sealed C/Real4000 and random Real12000."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/training'))
import train_b210_held_residual_uq as held
from train_random_real12000 import BASELINE_SHA, NORMALIZER_SHA


def main(args, *, rebalanced=False, folded_eight=False):
    assert not (rebalanced and folded_eight)
    torch, dense, _, prefix = held.runtime()
    import numpy as np
    import evaluate_b210_transfer as ev
    from models.ensemble_unet2d import EnsembleUNet2d
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    ev.TIME_SCORE = 92.0326603523
    rows = ev.clean_rows(args.manifest)
    fit_files = [row['file'] for row in rows if row['aoa'] != 15]
    test = [row for row in rows if row['aoa'] == 15]
    scorer = ev.import_file(ev.SCORER, 'random_real_scorer')
    keys = ev.METRICS[:3]
    functions = (lambda p, y: scorer.rel_l2_per_sample(p, y, 2),
                 lambda p, y: scorer.tke_rel_l2_per_sample(p, y, 2), scorer.mvpe_rel_l2_per_sample)
    results, receipts = {}, {}
    for label, directory, updates in (('control', args.control_dir, 4000), ('candidate', args.root, 4000 if rebalanced or folded_eight else 12000)):
        receipt = json.loads((directory / 'point_endpoint.json').read_text())
        assert receipt['held_aoa'] == 15 and receipt['fit_files'] == fit_files
        assert receipt['updates'] == updates
        assert held.sha256(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
        if label == 'control':
            assert receipt['checkpoint_sha256'] == BASELINE_SHA
        elif not rebalanced and not folded_eight:
            assert receipt['initialization'] == 'random'
        payload = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)
        assert payload['iteration'] == updates and payload['held_aoa'] == 15
        assert payload['normalizer_sha256'] == NORMALIZER_SHA
        if folded_eight and label == 'candidate':
            assert payload['schema'] == 'eight_head_affine_fold_point_v1'
            assert receipt['schema'] == 'eight_head_affine_fold_receipt_v1'
            assert receipt['sample_stream_sha256'] == '7e7a4e6a384afd0a6edacb7607449a9070b1cbc5e655cdaaa4912bf1a2dd2682'
            assert payload['fold'] == receipt['fold']
            assert receipt['start_sim_sha256'] == '628860c7d82239d942895f08a6fd72ede81feb6a7cad4de4e32e110ed2742b9b'
            assert receipt['updates'] == 4000 and receipt['seed'] == 4
        if rebalanced:
            from train_b210_transfer_uq import validate_point_receipt, validate_point_payload
            validate_point_receipt(receipt, 'C', full_real_point=True)
            validate_point_payload(payload, receipt)
            if label == 'candidate':
                from train_rebalanced_real4000 import INTERVENTION
                sampling = receipt['re_balanced_sampling']
                assert all(sampling[key] == value for key, value in INTERVENTION.items())
                assert receipt['sample_stream_sha256'] == '6aea6c185d7796ccb25f7f84d6543e5a28ae085909b91298d9cc411b5230a296'
                # JSON receipts stringify integer Re keys; preserve all values when comparing.
                assert json.loads(json.dumps(payload['start']['re_balanced_sampling'])) == sampling
        config = dense.ft12.validate_b210_model_config(payload['model_config'])
        model = EnsembleUNet2d(config).float()
        model.load_state_dict(payload['model_state_dict'], strict=True)
        norm, _ = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
        model.cuda().eval()
        hashes = {key: hashlib.sha256() for key in ('input', 'truth')}
        errors = {row['file']: [] for row in test}
        with torch.inference_mode():
            for history, truth, labels in ev.tail_batches(test, args.real_dir):
                hashes['input'].update(history.tobytes())
                hashes['truth'].update(truth.tobytes())
                x, _ = norm.preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
                prediction = model(x)
                zeros = torch.zeros_like(prediction[..., :2])
                prediction = held.absolute_distribution(prediction, zeros, zeros, norm)[0].cpu().numpy()
                assert np.isfinite(prediction).all() and np.all(prediction[..., 2] == 0)
                values = np.stack([fn(prediction, truth) for fn in functions], axis=-1)
                for file, value in zip(labels, values):
                    errors[file].append(value.tolist())
        def metrics(values):
            means = np.asarray(values, dtype=np.float64).mean(axis=0)
            return {key: scorer.score_error(value) for key, value in zip(keys, means)}
        results[label] = dict(point_metrics=metrics([v for values in errors.values() for v in values]),
                              trajectory_metrics={file: metrics(values) for file, values in errors.items()},
                              errors=errors, hashes={key: value.hexdigest() for key, value in hashes.items()})
        receipts[label] = receipt
        del payload, model, norm
        gc.collect()
        torch.cuda.empty_cache()
    assert results['control']['hashes'] == results['candidate']['hashes']
    assert sum(map(len, results['control']['errors'].values())) == 153 and len(test) == 17
    expected = (95.00859412090676, 80.16879759178723, 95.617979435169)
    assert max(abs(results['control']['point_metrics'][key]-value) for key, value in zip(keys, expected)) < 1e-5
    def contribution(point):
        return ev.fixed_time_final({**point, 'sps_score': 43.2953289552})
    changes = {key: results['candidate']['point_metrics'][key]-results['control']['point_metrics'][key] for key in keys}
    point_delta = contribution(results['candidate']['point_metrics'])-contribution(results['control']['point_metrics'])
    trajectory_deltas = {file: contribution(results['candidate']['trajectory_metrics'][file])-contribution(value)
                         for file, value in results['control']['trajectory_metrics'].items()}
    positive = sum(value >= 0 for value in trajectory_deltas.values())
    gate = .05 if rebalanced or folded_eight else .08
    experiment = 'EIGHT_HEAD_FOLDED_REAL4000' if folded_eight else 'REBALANCED_REAL4000' if rebalanced else 'RANDOM_REAL12000'
    scope = ('Both arms exclude held15 during Real training; shared Sim saw held prefixes. Recipe selection reused held15. '
             if rebalanced or folded_eight else 'Candidate training excludes held15; baseline shared Sim saw held prefixes. Recipe selection reused held15. ')
    passed = point_delta >= gate and min(changes.values()) >= -.03 and positive/17 >= .70
    result = dict(decision=('PASS_' if passed else 'STOP_')+experiment+'_HELD15',
                  changes=changes, point_fitted_delta=point_delta, nonnegative_trajectories=positive,
                  trajectory_count=17, trajectory_deltas=trajectory_deltas, results=results, receipts=receipts,
                  source_sha256=held.sha256(__file__),
                  gate=dict(point_fitted_delta=gate, minimum_point_subscore_delta=-.03, nonnegative_trajectory_fraction=.70),
                  scope=scope+'SPS/Time are fixed, not measured gains.')
    dense.dr1.write_json_atomic(args.root / 'RESULT.json', result)
    print(json.dumps({key: result[key] for key in ('decision', 'changes', 'point_fitted_delta', 'nonnegative_trajectories')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'control-dir', 'manifest', 'real-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
