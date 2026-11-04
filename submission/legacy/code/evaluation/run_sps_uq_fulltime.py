"""One sealed prefix/full-time UQ pair per held angle, using the release runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
for folder in ('code/evaluation', 'code/training', 'code/submissions', 'code'):
    sys.path.insert(0, str(ROOT / folder))
import evaluate_b210_transfer as replay

ZIP_SHA = 'b8938d06c100cc68bb4b7e310de112754a9053b4d2834134cb2b4e6f4b5c2a35'
POINT_SHA = {5: '8de05d5f98e6f99c78e51f3bf531601ad2d2f38e2bd4df4ec0598c7945011dc9',
             20: 'beac5c0f5e5ec6b05b5486d6ae25c72e3b6ca55862425ae75ac5b12ee943ac29'}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def loss_parts(official, prediction, truth, lower, upper):
    """Exact additive decomposition of 100-SPS on the scorer's element set."""
    import numpy as np
    errors = [official.rel_l2_per_sample(prediction, truth, 2),
              official.tke_rel_l2_per_sample(prediction, truth, 2),
              official.mvpe_rel_l2_per_sample(prediction, truth)]
    factor = sum(w * .5 / (.5 + error) for w, error in zip((.5, .3, .2), errors))
    factor = factor.reshape(-1, 1, 1, 1, 1)
    measured = truth[..., :2] != 0
    inside = ((truth[..., :2] >= lower[..., :2]) & (truth[..., :2] <= upper[..., :2]))
    penalty = np.exp(-(upper[..., :2] - lower[..., :2]) / official.SIGMA_GLOBAL)
    arrays = {'accuracy_loss': np.broadcast_to(1-factor, measured.shape),
              'miss_loss': factor * ~inside,
              'width_loss': factor * inside * (1-penalty)}
    result = {key: float(np.sum(value, where=measured, dtype=np.float64)) for key, value in arrays.items()}
    result['count'] = int(measured.sum())
    return result


def worker(args):
    import dataclasses
    import io
    import numpy as np
    import torch
    from build_track1_blended_ensemble_unet2d_package import compact_state_dict
    from models.ensemble_unet2d import EnsembleUNet2dConfig
    runtime = args.output / ('runtime_' + args.label)
    runtime.mkdir()
    with zipfile.ZipFile(args.baseline_zip) as archive:
        for name in replay.RUNTIME_FILES:
            target = runtime / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
        template = torch.load(io.BytesIO(archive.read('ensemble_unet2d.pth')), map_location='cpu', weights_only=False)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    configs = []
    for item in (template, payload):
        config = dict(item['model_config'])
        if config.pop('architecture') != 'ensemble_unet':
            raise ValueError('Unexpected runtime architecture')
        configs.append(dataclasses.asdict(EnsembleUNet2dConfig(**config)))
    if configs[0] != configs[1]:
        raise ValueError('Training architecture differs from release runtime')
    compact, _ = compact_state_dict(payload['model_state_dict'], 1024)
    torch.save({**template, 'quantized_state_dict': compact, 'normalizer': payload['normalizer']},
               runtime / 'ensemble_unet2d.pth')
    del payload, compact, template
    # Import the ZIP's model modules after checking the current trainer config.
    for name in list(sys.modules):
        if name == 'models' or name.startswith('models.'):
            del sys.modules[name]
    sys.path.insert(0, str(runtime))
    submission = replay.import_file(runtime / 'submission.py', 'fulltime_submission')
    official = replay.import_file(replay.SCORER, 'fulltime_official')
    rows = [r for r in replay.clean_rows(args.clean_manifest) if r['aoa'] == args.held_aoa]
    states = {r['file']: [] for r in rows}
    hashes = {k: hashlib.sha256() for k in ('inputs', 'truth', 'prediction')}
    parts = []
    for inputs, truth, labels in replay.tail_batches(rows, args.real_dir):
        output = submission.predict(inputs)
        for key, value in [('inputs', inputs), ('truth', truth), ('prediction', output['prediction'])]:
            hashes[key].update(value.tobytes())
        for label in dict.fromkeys(labels):
            chosen = np.asarray([v == label for v in labels])
            states[label].append(replay.metric_state(official, output['prediction'][chosen], truth[chosen],
                                                     output['lower'][chosen], output['upper'][chosen]))
        parts.append(loss_parts(official, output['prediction'], truth, output['lower'], output['upper']))
    states = {key: replay.merge_states(values) for key, values in states.items()}
    pooled = replay.merge_states(list(states.values()))
    decomposition = replay.merge_states(parts)
    result = {'metrics': replay.finalize(official, pooled), 'pooled_state': pooled,
              'trajectory_states': states, 'loss_parts': decomposition,
              'hashes': {k: v.hexdigest() for k, v in hashes.items()},
              'checkpoint_sha256': replay.sha256(args.checkpoint), 'runtime_zip_sha256': ZIP_SHA,
              'evaluator_sha256': replay.sha256(Path(__file__)),
              'scorer_sha256': replay.sha256(replay.SCORER), 'held_aoa': args.held_aoa}
    loss = 100 * sum(decomposition[k] for k in ('accuracy_loss', 'miss_loss', 'width_loss')) / decomposition['count']
    if abs(loss + result['metrics']['sps_score'] - 100) > 2e-5:
        raise ValueError('SPS loss decomposition disagrees with official scorer')
    save(args.output / (args.label + '_evaluation.json'), result)


def run(args):
    if (args.output.exists() and not args.resume_sealed) or replay.sha256(args.baseline_zip) != ZIP_SHA:
        raise ValueError('New output directory and exact 918343 ZIP required')
    args.output.mkdir(parents=True, exist_ok=args.resume_sealed)
    deadline = time.monotonic() + 7200
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}
    identity = {'source_commit': args.source_commit, 'baseline_zip_sha256': ZIP_SHA,
                'folds': [5, 20], 'point_sha256': POINT_SHA, 'started_epoch': time.time(),
                'queue_cap_seconds': 7200, 'scope': 'Real/UQ-stage LOAO, shared Sim has historical Real ancestry',
                'gate': 'both SPS deltas > 0 and pooled SPS delta >= .20; identical point',
                'hardware': 'A800; both arms retrained', 'submission_authorized': False}
    if args.resume_sealed:
        identity = json.loads((args.output / 'identity.json').read_text())
        if identity['source_commit'] != args.source_commit or identity['baseline_zip_sha256'] != ZIP_SHA:
            raise ValueError('Resume must preserve the original training and runtime bindings')
        deadline -= time.time()-identity['started_epoch']
        save(args.output / 'evaluation_repair.json', {'resumed_epoch': time.time(),
             'evaluator_sha256': replay.sha256(Path(__file__)),
             'reason': 'Use release config after canonical architecture equality check; sealed weights unchanged'})
    else:
        save(args.output / 'identity.json', identity)

    def execute(script, options, log, cap):
        if (args.output / 'STOP').exists():
            raise RuntimeError('External STOP requested')
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen([sys.executable, str(ROOT / script), *map(str, options)],
                                       stdout=stream, stderr=subprocess.STDOUT, env=env)
            save(args.output / 'status.json', {'status': 'RUNNING', 'pid': process.pid,
                                              'log': str(log), 'started_epoch': time.time()})
            try:
                code = process.wait(timeout=max(.01, min(cap, deadline-time.monotonic())))
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                raise
            if code:
                raise RuntimeError(f'{script} failed with exit {code}; {log}')

    completed = {}
    for aoa in (5, 20):
        fold = args.output / f'held{aoa}'
        fold.mkdir(exist_ok=args.resume_sealed)
        receipt_path = args.point_root / f'fold{aoa}/held{aoa}/point_full/point_endpoint.json'
        point = json.loads(receipt_path.read_text())
        if point['checkpoint_sha256'] != POINT_SHA[aoa] or replay.sha256(point['checkpoint_path']) != POINT_SHA[aoa]:
            raise ValueError('Sealed full-Real point identity changed')
        for label in ('prefix', 'fulltime'):
            if args.resume_sealed and (fold / label / 'uq_endpoint.json').exists():
                continue
            options = ['--execute', '--held-aoa', aoa, '--label', 'full', '--point-label', 'C',
                       '--full-real-point', '--initialization', 'point', '--seed', 18,
                       '--point-receipt', receipt_path, '--point-receipt-sha256', replay.sha256(receipt_path),
                       '--clean-manifest', args.clean_manifest, '--real-dir', args.real_dir,
                       '--output-dir', fold / label, '--source-commit', args.source_commit,
                       '--wall-budget-seconds', 1800]
            if label == 'fulltime':
                options += ['--uq-full-time']
            execute('code/training/train_b210_transfer_uq.py', options, fold / (label + '_training.log'), 1900)
        receipts = {label: json.loads((fold / label / 'uq_endpoint.json').read_text())
                    for label in ('prefix', 'fulltime')}
        for key in ('initial_uq_sha256', 'point_frozen_sha256', 'point_receipt_sha256',
                    'normalizer_sha256', 'model_config_sha256', 'updates', 'seed'):
            if receipts['prefix'][key] != receipts['fulltime'][key]:
                raise ValueError('UQ pair identity mismatch: ' + key)
        fit = [r for r in replay.clean_rows(args.clean_manifest) if r['aoa'] != aoa]
        for label, receipt in receipts.items():
            expected = (sum(len(r['train_starts']) for r in fit) if label == 'prefix'
                        else sum((r['frames']-40)//20+1 for r in fit))
            if (receipt['fit_windows'] != expected or receipt['access_audit']['opened'] != [r['file'] for r in fit]
                    or not receipt['point_state_bitwise'] or replay.sha256(receipt['checkpoint_path']) != receipt['checkpoint_sha256']):
                raise ValueError('UQ support, frozen point or endpoint seal mismatch')
        # Both endpoints are sealed before the first held trajectory is opened.
        for label, receipt in receipts.items():
            if args.resume_sealed and (fold / (label + '_evaluation.json')).exists():
                continue
            execute('code/evaluation/run_sps_uq_fulltime.py', ['--worker', '--label', label,
                    '--held-aoa', aoa, '--checkpoint', receipt['checkpoint_path'], '--output', fold,
                    '--baseline-zip', args.baseline_zip, '--clean-manifest', args.clean_manifest,
                    '--real-dir', args.real_dir], fold / (label + '_evaluation.log'), 600)
        results = {label: json.loads((fold / (label + '_evaluation.json')).read_text())
                   for label in receipts}
        if any(results['prefix']['hashes'][k] != results['fulltime']['hashes'][k] for k in ('inputs', 'truth')):
            raise ValueError('Inputs or truth differed between UQ arms')
        point_delta = max(abs(results['fulltime']['metrics'][k]-results['prefix']['metrics'][k])
                          for k in replay.METRICS[:3])
        if point_delta > 1e-5:
            raise ValueError('Frozen point score difference exceeds A800 numerical tolerance')
        for label, result in results.items():
            if result['checkpoint_sha256'] != receipts[label]['checkpoint_sha256'] or result['runtime_zip_sha256'] != ZIP_SHA:
                raise ValueError('Evaluation does not bind the sealed checkpoint and release runtime')
        delta = {key: results['fulltime']['metrics'][key] - results['prefix']['metrics'][key]
                 for key in results['prefix']['metrics']}
        completed[str(aoa)] = {'delta': delta, 'arms': results,
                              'point_predictions_bitwise': results['prefix']['hashes']['prediction'] == results['fulltime']['hashes']['prediction'],
                              'point_score_max_delta': point_delta, 'point_score_atol': 1e-5}
        save(fold / 'comparison.json', completed[str(aoa)])
        print(json.dumps({'held_aoa': aoa, 'delta': delta}), flush=True)
        if delta['sps_score'] <= 0:
            break
    official = replay.import_file(replay.SCORER, 'fulltime_aggregate')
    pooled = {label: replay.finalize(official, replay.merge_states(
        [r['arms'][label]['pooled_state'] for r in completed.values()])) for label in ('prefix', 'fulltime')}
    delta = {key: pooled['fulltime'][key]-pooled['prefix'][key] for key in pooled['prefix']}
    passed = len(completed) == 2 and all(r['delta']['sps_score'] > 0 for r in completed.values()) and delta['sps_score'] >= .20
    save(args.output / 'decision.json', {'decision': 'PASS_TWO_STAGE_FOLDS' if passed else 'STOP_UQ_FULLTIME_CONFIG',
         'pooled': pooled, 'delta': delta, 'folds': completed, 'official_gain': None,
         'elapsed_seconds': 7200-(deadline-time.monotonic())})
    save(args.output / 'status.json', {'status': 'TERMINAL', 'passed': passed, 'finished_epoch': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--resume-sealed', action='store_true')
    parser.add_argument('--label')
    parser.add_argument('--held-aoa', type=int)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--source-commit')
    for key in ('output', 'baseline-zip', 'clean-manifest', 'real-dir', 'point-root'):
        parser.add_argument('--'+key, type=Path, required=key not in ('point-root',))
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        try:
            run(args)
        except Exception as error:
            if args.output.is_dir():
                save(args.output / 'status.json', {'status': 'ERROR', 'error': str(error), 'finished_epoch': time.time()})
            raise
