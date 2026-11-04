"""Fixed C mean plus energy-preserved C/random-Real fluctuation teacher."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from screen_energy_ensemble_teacher import energy_average


def combine(control, random):
    prediction, energy = energy_average(control, random)
    prediction[..., :2] += .5 * (control[..., :2].mean(axis=1, keepdims=True)
                                 - random[..., :2].mean(axis=1, keepdims=True))
    return prediction, energy


def main(args):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'code/training'))
    import train_b210_held_residual_uq as held
    torch, dense, _, prefix = held.runtime()
    import evaluate_b210_transfer as ev
    from models.ensemble_unet2d import EnsembleUNet2d
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    ev.TIME_SCORE = 92.0326603523
    prior = json.loads((args.random_dir / 'RESULT.json').read_text())
    assert prior['decision'] == 'STOP_RANDOM_REAL12000_HELD15'
    rows = ev.clean_rows(args.manifest)
    fit = [r['file'] for r in rows if r['aoa'] != 15]
    test = [r for r in rows if r['aoa'] == 15]
    labels = ('control', 'random', 'teacher')
    models, norms, receipts, configs = [], [], [], []
    identities = ('ce84999755b685bfc84061cec3367ee324f64d71661ee5fcfd33320182c5b5fa',
                  'a3a2dc56dfcb8e7a32c247331cbb2daec786d88e52de7de5a60cf4de0ba0964b')
    for i, directory in enumerate((args.control_dir, args.random_dir)):
        receipt = json.loads((directory / 'point_endpoint.json').read_text())
        assert receipt['checkpoint_sha256'] == identities[i] == held.sha256(receipt['checkpoint_path'])
        assert receipt['held_aoa'] == 15 and receipt['fit_files'] == fit and receipt['seed'] == 4
        assert receipt['updates'] == (4000, 12000)[i]
        payload = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)
        assert payload['normalizer_sha256'] == '641e7fdfa05af5fefc195a8d7906e0cd44796e66b7d3cc4fac42ca99eabf7b1d'
        config = dense.ft12.validate_b210_model_config(payload['model_config'])
        configs.append(config)
        model = EnsembleUNet2d(config).float()
        model.load_state_dict(payload['model_state_dict'], strict=True)
        norms.append(prefix.normalizer_from_point(payload, torch.device('cuda:0'))[0])
        models.append(model.cuda().eval())
        receipts.append(receipt)
        del payload
    assert configs[0] == configs[1] and configs[0].point_feature_refiner_channels == 0
    scorer = ev.import_file(ev.SCORER, 'cross_source_scorer')
    keys = ev.METRICS[:3]
    functions = (lambda p, y: scorer.rel_l2_per_sample(p, y, 2),
                 lambda p, y: scorer.tke_rel_l2_per_sample(p, y, 2), scorer.mvpe_rel_l2_per_sample)
    errors = {label: {r['file']: [] for r in test} for label in labels}
    hashes = {key: hashlib.sha256() for key in ('input', 'truth')}
    energy = []
    with torch.inference_mode():
        for history, truth, files in ev.tail_batches(test, args.real_dir):
            hashes['input'].update(history.tobytes()); hashes['truth'].update(truth.tobytes())
            predictions = []
            for model, norm in zip(models, norms):
                x, _ = norm.preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
                point = model(x)
                zero = torch.zeros_like(point[..., :2])
                predictions.append(held.absolute_distribution(point, zero, zero, norm)[0].cpu().numpy())
            teacher, diagnostic = combine(*predictions)
            energy.append(diagnostic)
            predictions.append(teacher)
            for label, prediction in zip(labels, predictions):
                assert np.isfinite(prediction).all() and np.all(prediction[..., 2] == 0)
                values = np.stack([fn(prediction, truth) for fn in functions], axis=-1)
                for file, value in zip(files, values):
                    errors[label][file].append(value.tolist())
    def metrics(values):
        return {k: scorer.score_error(v) for k, v in zip(keys, np.asarray(values, dtype=np.float64).mean(axis=0))}
    def contribution(point):
        return ev.fixed_time_final({**point, 'sps_score': 43.2953289552})
    results = {label: dict(point_metrics=metrics([v for values in files.values() for v in values]),
                           trajectory_metrics={file: metrics(values) for file, values in files.items()}, errors=files)
               for label, files in errors.items()}
    assert len(test) == 17 and sum(map(len, errors['control'].values())) == 153
    assert {k: h.hexdigest() for k, h in hashes.items()} == prior['results']['control']['hashes']
    for label, old in (('control', 'control'), ('random', 'candidate')):
        assert max(abs(results[label]['point_metrics'][k]-prior['results'][old]['point_metrics'][k]) for k in keys) < 1e-5
    changes = {k: results['teacher']['point_metrics'][k]-results['control']['point_metrics'][k] for k in keys}
    assert abs(changes['mvpe_rel_l2_score']) < 1e-5
    delta = contribution(results['teacher']['point_metrics'])-contribution(results['control']['point_metrics'])
    trajectory = {f: contribution(v)-contribution(results['control']['trajectory_metrics'][f])
                  for f, v in results['teacher']['trajectory_metrics'].items()}
    positive = sum(v >= 0 for v in trajectory.values())
    passed = delta >= .08 and min(changes.values()) >= -.03 and positive/17 >= .70
    result = dict(decision=('PASS_' if passed else 'STOP_')+'CROSS_SOURCE_FLUCT_TEACHER',
                  point_fitted_delta=delta, changes=changes, nonnegative_trajectories=positive,
                  trajectory_deltas=trajectory, results=results, energy=energy, receipts=receipts,
                  hashes={k: h.hexdigest() for k, h in hashes.items()}, source_sha256=held.sha256(__file__),
                  scope='Shared Sim ancestors and reused held15 selection; two-model teacher only, no student, SPS, Time or official gain.')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('decision', 'point_fitted_delta', 'changes', 'nonnegative_trajectories')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('control-dir', 'random-dir', 'manifest', 'real-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
