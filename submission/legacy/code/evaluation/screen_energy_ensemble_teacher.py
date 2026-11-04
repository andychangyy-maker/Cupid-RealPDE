"""One fixed two-seed teacher diagnostic; never use a two-model submission."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def energy_average(first, second):
    a, b = first[..., :2], second[..., :2]
    mean = .5 * (a.mean(axis=1, keepdims=True) + b.mean(axis=1, keepdims=True))
    da = a - a.mean(axis=1, keepdims=True)
    db = b - b.mean(axis=1, keepdims=True)
    fluctuation = .5 * (da + db)
    target = .5 * ((da * da).sum(axis=-1).mean(axis=1)
                   + (db * db).sum(axis=-1).mean(axis=1))
    before = (fluctuation * fluctuation).sum(axis=-1).mean(axis=1)
    gain = np.minimum(np.sqrt(target / (before + 1e-12)), 2.)
    result = np.zeros_like(first)
    result[..., :2] = mean + fluctuation * gain[:, None, ..., None]
    after = ((result[..., :2] - result[..., :2].mean(axis=1, keepdims=True)) ** 2).sum(axis=-1).mean(axis=1)
    return result, dict(pixels=int(target.size), capped_pixels=int(np.count_nonzero(target > 4 * (before + 1e-12))),
                        target_energy_sum=float(target.sum()), before_energy_sum=float(before.sum()),
                        after_energy_sum=float(after.sum()))


def main(args):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'code/training'))
    import train_b210_held_residual_uq as held
    torch, dense, native, prefix = held.runtime()
    import evaluate_b210_transfer as ev
    from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    ev.TIME_SCORE = 92.0326603523
    rows = ev.clean_rows(args.manifest)
    fit_files = [r['file'] for r in rows if r['aoa'] != 15]
    test = [r for r in rows if r['aoa'] == 15]
    receipts = [json.loads(args.seed4_receipt.read_text()), json.loads(args.seed5_receipt.read_text())]
    assert receipts[1]['checkpoint_sha256'] == '3b7f60e2eb888a365249d095eabef8e81a6a2970bd19e050fcbb467ac55d54e0'
    models, normalizers, configs = [], [], []
    for i, receipt in enumerate(receipts):
        assert receipt['held_aoa'] == 15 and receipt['updates'] == 4000 and receipt['seed'] == 4+i
        assert receipt['fit_files'] == fit_files and receipt['optimizer_windows'] == 10380
        assert receipt['start_sim_sha256'] == '628860c7d82239d942895f08a6fd72ede81feb6a7cad4de4e32e110ed2742b9b'
        checkpoint = args.seed5_checkpoint if i == 1 else Path(receipt['checkpoint_path'])
        assert held.sha256(checkpoint) == receipt['checkpoint_sha256']
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert payload['normalizer_sha256'] == '641e7fdfa05af5fefc195a8d7906e0cd44796e66b7d3cc4fac42ca99eabf7b1d'
        config = dict(payload['model_config']); assert config.pop('architecture') == 'ensemble_unet'
        config = EnsembleUNet2dConfig(**config); assert config.point_feature_refiner_channels == 0
        configs.append(config)
        model = EnsembleUNet2d(config).float()
        model.load_state_dict(payload['model_state_dict'], strict=True)
        normalizers.append(prefix.normalizer_from_point(payload, torch.device('cuda:0'))[0])
        models.append(model.cuda().eval())
        del payload
    assert configs[0] == configs[1]
    scorer = ev.import_file(ev.SCORER, 'energy_teacher_scorer')
    keys = ev.METRICS[:3]
    functions = (lambda p, y: scorer.rel_l2_per_sample(p, y, 2),
                 lambda p, y: scorer.tke_rel_l2_per_sample(p, y, 2), scorer.mvpe_rel_l2_per_sample)
    labels = ('seed4', 'seed5', 'ordinary_average', 'energy_average')
    errors = {label: {r['file']: [] for r in test} for label in labels}
    hashes = {k: hashlib.sha256() for k in ('input', 'truth')}
    energy = []
    with torch.inference_mode():
        for history, truth, files in ev.tail_batches(test, args.real_dir):
            hashes['input'].update(history.tobytes()); hashes['truth'].update(truth.tobytes())
            predictions = []
            for model, norm in zip(models, normalizers):
                x, _ = norm.preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
                point = model(x)
                zeros = torch.zeros_like(point[..., :2])
                predictions.append(held.absolute_distribution(point, zeros, zeros, norm)[0].cpu().numpy())
            candidate, diagnostic = energy_average(*predictions); energy.append(diagnostic)
            predictions += [.5*(predictions[0]+predictions[1]), candidate]
            for label, point in zip(labels, predictions):
                assert np.isfinite(point).all() and np.all(point[..., 2] == 0)
                values = np.stack([fn(point, truth) for fn in functions], axis=-1)
                for file, value in zip(files, values):
                    errors[label][file].append(value.tolist())
    def metrics(values):
        return {k: scorer.score_error(v) for k,v in zip(keys, np.asarray(values,dtype=np.float64).mean(axis=0))}
    def contribution(point):
        return ev.fixed_time_final({**point,'sps_score':43.2953289552})
    results = {label: dict(point_metrics=metrics([v for values in files.values() for v in values]),
                           trajectory_metrics={file:metrics(values) for file,values in files.items()}, errors=files)
               for label, files in errors.items()}
    assert sum(map(len, errors['seed4'].values())) == 153
    changes = {}
    for label in ('seed4','seed5'):
        candidate, baseline = results['energy_average'], results[label]
        trajectory = {file: contribution(value)-contribution(baseline['trajectory_metrics'][file])
                      for file,value in candidate['trajectory_metrics'].items()}
        changes[label] = dict(point_fitted_delta=contribution(candidate['point_metrics'])-contribution(baseline['point_metrics']),
                             metrics={k:candidate['point_metrics'][k]-baseline['point_metrics'][k] for k in keys},
                             nonnegative_trajectories=sum(v>=0 for v in trajectory.values()), trajectory_deltas=trajectory)
    passed = all(c['point_fitted_delta'] >= .05 and c['metrics'][keys[1]] >= .20
                 and c['metrics'][keys[0]] >= -.03 and c['metrics'][keys[2]] >= -.03
                 and c['nonnegative_trajectories']/len(test) >= .70 for c in changes.values())
    result = dict(decision='PASS_ENERGY_TEACHER_TO_STUDENT_RESEARCH' if passed else 'STOP_ENERGY_ENSEMBLE_TEACHER',
                  changes=changes,results=results,receipts=receipts,energy=energy,
                  hashes={k:v.hexdigest() for k,v in hashes.items()},script_sha256=held.sha256(__file__),
                  scope='Real-stage held15 diagnostic with shared Sim ancestors; teacher is two forwards, no student or official gain.')
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'decision':result['decision'],'changes':changes}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('seed4-receipt','seed5-receipt','seed5-checkpoint','manifest','real-dir','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
