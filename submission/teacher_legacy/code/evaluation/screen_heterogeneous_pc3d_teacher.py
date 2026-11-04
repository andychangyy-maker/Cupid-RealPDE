"""One fixed C-mean, energy-preserved C/PC3D teacher on held15."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from screen_cross_source_fluct_teacher import combine


def main(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
    import train_b210_held_residual_uq as held
    import train_heterogeneous_pc3d_real as protocol
    import evaluate_b210_transfer as ev
    from models.ensemble_unet2d import EnsembleUNet2d

    torch, dense, _, prefix = held.runtime()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    ev.TIME_SCORE = 92.0326603523
    angle = getattr(args, 'held_aoa', 15)
    assert angle in (0, 15, 20)
    assert held.sha256(args.manifest) == protocol.MANIFEST_SHA
    rows = ev.clean_rows(args.manifest)
    fit = [r['file'] for r in rows if r['aoa'] != angle]
    test = [r for r in rows if r['aoa'] == angle]
    reference = json.loads(args.control_result.read_text())['results']['control'] if angle == 15 else None
    control = json.loads(args.control_receipt.read_text())
    assert control['checkpoint_sha256'] == held.sha256(control['checkpoint_path'])
    if angle == 15: assert control['checkpoint_sha256'] == protocol.BASELINE_SHA
    assert control['fit_files'] == fit and control['held_aoa'] == angle and control['updates'] == 4000
    assert control['seed'] == 4 and control['arm'] == 'C' and not control['held_data_read']
    payload = torch.load(control['checkpoint_path'], map_location='cpu', weights_only=False)
    normalizer_sha = payload['normalizer_sha256']
    if angle == 15: assert normalizer_sha == protocol.REFERENCE_NORMALIZER_SHA
    config = dense.ft12.validate_b210_model_config(payload['model_config'])
    model = EnsembleUNet2d(config).float()
    model.load_state_dict(payload['model_state_dict'], strict=True)
    norm = prefix.normalizer_from_point(payload, torch.device('cuda:0'))[0]
    models = [model.cuda().eval()]
    del payload
    candidate = json.loads((args.root / 'candidate' / 'point_endpoint.json').read_text())
    source = candidate['source']
    assert source['experiment'] == ('heterogeneous_pc3d_teacher_v1' if angle == 15 else 'heterogeneous_pc3d_confirm_v1')
    assert source['baseline_point_sha256'] == control['checkpoint_sha256']
    assert source['fit_files'] == fit and source['held_aoa'] == angle and not source['held_data_read']
    assert source['normalizer_windows'] == control['fit_windows'] and source['optimizer_windows'] == control['optimizer_windows']
    assert source['original_c_sample_and_bootstrap_sha256'] == control['sample_stream_sha256']
    assert source['normalizer_reference_semantic_sha256'] == normalizer_sha
    if angle != 15: assert source['source_commit'] == control['source_commit']
    assert candidate['training']['updates'] == 4000 and candidate['training']['sample_cursor'] == 4000
    payload = torch.load(candidate['endpoint']['path'], map_location='cpu', weights_only=False)
    assert payload['source'] == source and payload['update'] == 4000 and payload['ema_decay'] == .999
    assert all(torch.equal(v.cpu(), getattr(norm, k).cpu()) for k, v in payload['normalizer'].items())
    models.append(protocol.stage_r.load_ema_endpoint(candidate['endpoint'], torch.device('cuda:0')).eval())
    del payload
    scorer = ev.import_file(ev.SCORER, 'heterogeneous_pc3d_scorer')
    keys = ev.METRICS[:3]
    functions = (lambda p, y: scorer.rel_l2_per_sample(p, y, 2),
                 lambda p, y: scorer.tke_rel_l2_per_sample(p, y, 2), scorer.mvpe_rel_l2_per_sample)
    labels = ('control', 'pc3d_diagnostic_only', 'teacher')
    errors = {label: {r['file']: [] for r in test} for label in labels}
    hashes = {k: hashlib.sha256() for k in ('input', 'truth')}
    energy = []
    with torch.inference_mode():
        for history, truth, files in ev.tail_batches(test, args.real_dir):
            hashes['input'].update(history.tobytes())
            hashes['truth'].update(truth.tobytes())
            tensor = torch.from_numpy(history).cuda()
            x, _ = norm.preprocess(tensor, tensor)
            predictions = []
            for i, model in enumerate(models):
                point = model(x)
                if i == 1:
                    point = protocol.stage_r.zero_unmeasured_pressure(point)
                zero = torch.zeros_like(point[..., :2])
                predictions.append(held.absolute_distribution(point, zero, zero, norm)[0].cpu().numpy())
            teacher, diagnostic = combine(*predictions)
            energy.append(diagnostic)
            predictions.append(teacher)
            for label, point in zip(labels, predictions):
                assert np.isfinite(point).all() and np.all(point[..., 2] == 0)
                values = np.stack([fn(point, truth) for fn in functions], axis=-1)
                for file, value in zip(files, values):
                    errors[label][file].append(value.tolist())
    def metrics(values):
        return {k: scorer.score_error(v) for k, v in zip(keys, np.asarray(values, dtype=np.float64).mean(axis=0))}
    def contribution(value):
        return ev.fixed_time_final({**value, 'sps_score': 43.2953289552})
    results = {label: dict(point_metrics=metrics([v for values in files.values() for v in values]),
                           trajectory_metrics={f: metrics(values) for f, values in files.items()}, errors=files)
               for label, files in errors.items()}
    assert sum(map(len, errors['control'].values())) == sum(len(r['val_starts']) for r in test)
    if angle == 15:
        assert len(test) == 17 and sum(map(len, errors['control'].values())) == 153
        assert {k: v.hexdigest() for k, v in hashes.items()} == reference['hashes']
        assert max(abs(results['control']['point_metrics'][k] - reference['point_metrics'][k]) for k in keys) <= 1e-5
    changes = {k: results['teacher']['point_metrics'][k] - results['control']['point_metrics'][k] for k in keys}
    assert abs(changes['mvpe_rel_l2_score']) <= 1e-5
    delta = contribution(results['teacher']['point_metrics']) - contribution(results['control']['point_metrics'])
    trajectory = {f: contribution(v) - contribution(results['control']['trajectory_metrics'][f])
                  for f, v in results['teacher']['trajectory_metrics'].items()}
    positive = sum(v >= 0 for v in trajectory.values())
    passed = delta >= .08 and min(changes.values()) >= -.03 and positive / 17 >= .70
    decision = ('PASS_' if passed else 'STOP_') + 'HETEROGENEOUS_PC3D_TEACHER_HELD15' if angle == 15 else f'SEALED_PC3D_CONFIRM_HELD{angle}'
    result = dict(decision=decision, held_aoa=angle,
                  changes=changes, point_fitted_delta=delta, nonnegative_trajectories=positive,
                  trajectory_deltas=trajectory, trajectory_count=len(test), results=results, energy=energy,
                  receipts=dict(control=control, pc3d=candidate), source_sha256=held.sha256(__file__),
                  hashes={k: v.hexdigest() for k, v in hashes.items()},
                  scope='Real-stage angle holdout with shared C Sim ancestors; held15 reused for development. Teacher only; no student/UQ/Time/official gain.')
    (args.root / 'RESULT.json').write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k: result[k] for k in ('decision', 'changes', 'point_fitted_delta', 'nonnegative_trajectories')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'manifest', 'real-dir', 'control-receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--control-result', type=Path)
    parser.add_argument('--held-aoa', type=int, choices=(0, 15, 20), default=15)
    main(parser.parse_args())
