"""Frozen two-fold point/absolute-interval crossover; no training or packaging."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import evaluate_b210_transfer as ev


def mean_role_point(baseline, q4):
    import numpy as np
    return (q4.astype(np.float64) - q4.mean(axis=1, keepdims=True, dtype=np.float64)
            + baseline.mean(axis=1, keepdims=True, dtype=np.float64)).astype(np.float32)


def main(output, source_commit, mean_role=False):
    import numpy as np
    import torch
    started = time.monotonic()
    torch.set_num_threads(4)
    ev.TIME_SCORE = 92.03266035232576
    original = Path('/root/autodl-tmp/track1_reflection_exposure_stage_20260909')
    manifest = Path('/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json')
    real_dir = Path('/root/autodl-tmp/realpde/data/real')
    receipt_hashes = {
        15: ('e965bca6eb35468be17c1cf5d0cf6cb7d6c61cd81ca1ca161819d97da01fad08',
             '7fe6dcc90eb8f7b73445a2db2aa8f750f0e0128f5375d604fafdb46c906219da'),
        20: ('96ea76a6504b2b36b31515ccf36642d9450824b2600486f14f9664d38a7044d1',
             '095dd2762ec02ed4744f69d1127969f9f144aa84ccddd4e63fa5973b3cfc4c13'),
    }
    arms = {'base': ('control', 'control'), 'q4': ('reflection', 'reflection'),
            'base_point_q4_bounds': ('control', 'reflection'),
            'q4_point_base_bounds': ('reflection', 'control')}
    if mean_role:
        arms['base_mean_q4_fluctuation_bounds'] = ('mean_role', 'reflection')
    scorer = ev.import_file(ev.SCORER, 'q4_function_scorer')
    all_rows = ev.clean_rows(manifest)
    results, pooled, bindings = {}, {name: [] for name in arms}, {}
    runtime_files = None
    for angle in (15, 20):
        folder = original / f'held{angle}'
        sealed = json.loads((folder / 'evaluation.json').read_text())
        models, receipts = {}, {}
        for label, expected in zip(('control', 'reflection'), receipt_hashes[angle]):
            receipt_path = folder / label / 'uq/uq_endpoint.json'
            assert ev.sha256(receipt_path) == expected
            receipt = receipts[label] = json.loads(receipt_path.read_text())
            assert ev.sha256(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
            assert receipt['held_aoa'] == angle and receipt['point_receipt']['held_aoa'] == angle
            fit = [r['file'] for r in all_rows if r['aoa'] != angle]
            assert receipt['point_receipt']['fit_files'] == fit
            runtime = folder / label / 'runtime'
            assert ev.sha256(runtime / 'ensemble_unet2d.pth') == sealed['bindings'][label]['checkpoint_sha256']
            hashes = {name: ev.sha256(runtime / name) for name in ev.RUNTIME_FILES}
            if runtime_files is None:
                runtime_files = hashes
                sys.path.insert(0, str(runtime))
            assert hashes == runtime_files
            models[label] = ev.import_file(runtime / 'submission.py', f'q4_function_{angle}_{label}')
        assert receipts['control']['normalizer_sha256'] == receipts['reflection']['normalizer_sha256']
        rows = [r for r in all_rows if r['aoa'] == angle]
        states = {name: {r['file']: [] for r in rows} for name in arms}
        hashes = {name: hashlib.sha256() for name in ('input', 'truth')}
        for history, truth, labels in ev.tail_batches(rows, real_dir):
            hashes['input'].update(history.tobytes()); hashes['truth'].update(truth.tobytes())
            prediction = {label: model.predict(history) for label, model in models.items()}
            if mean_role:
                prediction['mean_role'] = {'prediction': mean_role_point(
                    prediction['control']['prediction'], prediction['reflection']['prediction'])}
            for name, (point_source, bounds_source) in arms.items():
                point, bounds = prediction[point_source], prediction[bounds_source]
                for trajectory in dict.fromkeys(labels):
                    take = np.asarray([trajectory == label for label in labels])
                    states[name][trajectory].append(ev.metric_state(
                        scorer, point['prediction'][take], truth[take], bounds['lower'][take], bounds['upper'][take]))
            if time.monotonic() - started > 300:
                raise TimeoutError('Q4 function crossover budget')
        states = {name: {key: ev.merge_states(value) for key, value in items.items()} for name, items in states.items()}
        metrics = {name: ev.finalize(scorer, ev.merge_states(list(value.values()))) for name, value in states.items()}
        for name, old in [('base', 'control'), ('q4', 'reflection')]:
            assert hashes['input'].hexdigest() == sealed['arms'][old]['input_sha256']
            assert hashes['truth'].hexdigest() == sealed['arms'][old]['target_sha256']
            assert max(abs(metrics[name][key] - sealed['arms'][old]['metrics'][key]) for key in ev.METRICS) <= 1e-4
        assert all(metrics['base'][key] == metrics['base_point_q4_bounds'][key] for key in ev.METRICS[:3])
        assert all(metrics['q4'][key] == metrics['q4_point_base_bounds'][key] for key in ev.METRICS[:3])
        if mean_role:
            role = metrics['base_mean_q4_fluctuation_bounds']
            assert abs(role['tke_rel_l2_score'] - metrics['q4']['tke_rel_l2_score']) <= 1e-4
            assert abs(role['mvpe_rel_l2_score'] - metrics['base']['mvpe_rel_l2_score']) <= 1e-4
        delta = {name: {key: values[key] - metrics['base'][key] for key in (*ev.METRICS, 'fixed_time_fitted_final')}
                 for name, values in metrics.items()}
        results[str(angle)] = {'metrics': metrics, 'delta': delta, 'states': states,
                               'hashes': {key: value.hexdigest() for key, value in hashes.items()}}
        for name, value in states.items(): pooled[name].extend(value.values())
        bindings[str(angle)] = sealed['bindings']
        print(json.dumps({'angle': angle, 'delta': delta}), flush=True)
        del models
    metrics = {name: ev.finalize(scorer, ev.merge_states(value)) for name, value in pooled.items()}
    delta = {key: metrics['base_point_q4_bounds'][key] - metrics['base'][key]
             for key in (*ev.METRICS, 'fixed_time_fitted_final')}
    passed = (delta['sps_score'] >= .20 and delta['fixed_time_fitted_final'] >= .045
              and all(value['delta']['base_point_q4_bounds']['sps_score'] >= 0 for value in results.values()))
    decision = 'PASS_Q4_FUNCTION_TEACHER_SCREEN' if passed else 'STOP_Q4_INTERVAL_FUNCTION_TRANSFER'
    if mean_role:
        candidate = 'base_mean_q4_fluctuation_bounds'
        delta = {key: metrics[candidate][key] - metrics['base'][key]
                 for key in (*ev.METRICS, 'fixed_time_fitted_final')}
        positive = sum(ev.finalize(scorer, fold['states'][candidate][name])['fixed_time_fitted_final']
                       >= ev.finalize(scorer, base)['fixed_time_fitted_final']
                       for fold in results.values() for name, base in fold['states']['base'].items())
        count = sum(len(fold['states']['base']) for fold in results.values())
        passed = (delta['fixed_time_fitted_final'] >= .10
                  and metrics[candidate]['fixed_time_fitted_final'] >= metrics['q4']['fixed_time_fitted_final']
                  and all(fold['delta'][candidate]['fixed_time_fitted_final'] >= 0 for fold in results.values())
                  and positive / count >= .60)
        decision = 'PASS_Q4_MEAN_ROLE_TEACHER_SCREEN' if passed else 'STOP_Q4_MEAN_ROLE_TEACHER'
        print(json.dumps({'nonnegative_trajectories': positive, 'trajectories': count}), flush=True)
    result = {'decision': decision, 'mean_role': mean_role,
              'delta': delta, 'metrics': metrics, 'folds': results, 'bindings': bindings,
              'runtime_files': runtime_files, 'source_commit': source_commit, 'seconds': time.monotonic() - started,
              'scope': 'Selection-reused stage-LOAO mechanism diagnostic; shared Sim has Real ancestry',
              'training_performed': False, 'official_gain': None, 'release_verified': False}
    (output / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('decision', 'delta', 'seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--mean-role', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    try:
        main(args.output, args.source_commit, args.mean_role)
    except Exception:
        (args.output / 'FAILED.json').write_text(json.dumps({'traceback': traceback.format_exc()}, indent=2))
        raise
