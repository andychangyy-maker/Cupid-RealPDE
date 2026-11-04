"""Fixed C temporal mean plus the frozen reversible teacher's fluctuations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/training'))
import train_b210_held_residual_uq as held


def main(args):
    torch, _, _, prefix = held.runtime()
    import numpy as np
    import evaluate_b210_transfer as ev
    from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
    from models.reversible_history_coordinates import install
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    ev.TIME_SCORE = 92.0326603523
    reference = held.read_bound(args.reference, args.reference_sha)
    assert reference['decision'] == 'STOP_REVERSIBLE_HISTORY_HELD15'
    assert held.sha256(args.manifest) == held.MANIFEST_SHA
    rows = ev.clean_rows(args.manifest)
    fit = [r['file'] for r in rows if r['aoa'] != 15]
    test = [r for r in rows if r['aoa'] == 15]
    models, norms = {}, {}
    for label in ('control', 'candidate'):
        receipt = reference['receipts'][label]
        assert held.sha256(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
        assert receipt['fit_files'] == fit and receipt['held_aoa'] == 15 and receipt['updates'] == 4000
        payload = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)
        assert payload['normalizer_sha256'] == '641e7fdfa05af5fefc195a8d7906e0cd44796e66b7d3cc4fac42ca99eabf7b1d'
        config = dict(payload['model_config'])
        assert config.pop('architecture') == 'ensemble_unet'
        model = EnsembleUNet2d(EnsembleUNet2dConfig(**config)).float()
        norms[label] = prefix.normalizer_from_point(payload, torch.device('cuda:0'))[0]
        if label == 'candidate':
            install(model, norms[label])
        model.load_state_dict(payload['model_state_dict'], strict=True)
        models[label] = model.cuda().eval()
        del payload
    scorer = ev.import_file(ev.SCORER, 'reversible_splice_scorer')
    keys = ev.METRICS[:3]
    functions = (lambda p, y: scorer.rel_l2_per_sample(p, y, 2),
                 lambda p, y: scorer.tke_rel_l2_per_sample(p, y, 2), scorer.mvpe_rel_l2_per_sample)
    errors = {label: {r['file']: [] for r in test} for label in ('control', 'candidate', 'splice')}
    hashes = {key: hashlib.sha256() for key in ('input', 'truth')}
    with torch.inference_mode():
        for history, truth, labels in ev.tail_batches(test, args.real_dir):
            hashes['input'].update(history.tobytes())
            hashes['truth'].update(truth.tobytes())
            predictions = {}
            for label, model in models.items():
                x, _ = norms[label].preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
                point = model(x)
                zero = torch.zeros_like(point[..., :2])
                predictions[label] = held.absolute_distribution(point, zero, zero, norms[label])[0].cpu().numpy()
            c = predictions['control'][..., :2].astype(np.float64)
            t = predictions['candidate'][..., :2].astype(np.float64)
            splice = np.zeros_like(predictions['control'])
            splice[..., :2] = c.mean(axis=1, keepdims=True) + t - t.mean(axis=1, keepdims=True)
            predictions['splice'] = splice
            for label, point in predictions.items():
                assert np.isfinite(point).all() and np.all(point[..., 2] == 0)
                values = np.stack([fn(point, truth) for fn in functions], axis=-1)
                for file, value in zip(labels, values):
                    errors[label][file].append(value.tolist())
    def metrics(values):
        return {key: scorer.score_error(value) for key, value in zip(keys, np.asarray(values, dtype=np.float64).mean(axis=0))}
    def contribution(point):
        return ev.fixed_time_final({**point, 'sps_score': 43.2953289552})
    results = {label: dict(point_metrics=metrics([v for values in files.values() for v in values]),
                           trajectory_metrics={file: metrics(values) for file, values in files.items()}, errors=files)
               for label, files in errors.items()}
    hashes = {key: value.hexdigest() for key, value in hashes.items()}
    assert len(test) == 17 and sum(map(len, errors['control'].values())) == 153
    for label in ('control', 'candidate'):
        assert hashes == reference['results'][label]['hashes']
        assert max(abs(results[label]['point_metrics'][key] - reference['results'][label]['point_metrics'][key]) for key in keys) <= 1e-5
    assert abs(results['splice']['point_metrics'][keys[2]] - results['control']['point_metrics'][keys[2]]) <= 1e-5
    assert abs(results['splice']['point_metrics'][keys[1]] - results['candidate']['point_metrics'][keys[1]]) <= 1e-5
    changes = {key: results['splice']['point_metrics'][key] - results['control']['point_metrics'][key] for key in keys}
    delta = contribution(results['splice']['point_metrics']) - contribution(results['control']['point_metrics'])
    trajectory = {file: contribution(value) - contribution(results['control']['trajectory_metrics'][file])
                  for file, value in results['splice']['trajectory_metrics'].items()}
    positive = sum(value >= 0 for value in trajectory.values())
    passed = delta >= .08 and min(changes.values()) >= -.03 and positive / 17 >= .70
    result = dict(decision=('PASS_' if passed else 'STOP_') + 'REVERSIBLE_FLUCTUATION_SPLICE',
                  changes=changes, point_fitted_delta=delta, nonnegative_trajectories=positive,
                  trajectory_count=17, trajectory_deltas=trajectory, results=results, hashes=hashes,
                  receipts=reference['receipts'], reference_sha256=args.reference_sha, source_sha256=held.sha256(__file__),
                  scope='Fixed offline splice, reused stage-held15; no new training, student, UQ, timing or official gain.')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('decision', 'changes', 'point_fitted_delta', 'nonnegative_trajectories')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('reference', 'manifest', 'real-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reference-sha', required=True)
    main(parser.parse_args())
