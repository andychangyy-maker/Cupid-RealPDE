"""One fixed five-fold Real/UQ-stage confirmation; no automatic new route."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/evaluation'))
import evaluate_b210_transfer as replay


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def run(args):
    if args.output_dir.exists() or not re.fullmatch('[0-9a-f]{40}', args.source_commit):
        raise ValueError('new output directory and committed source identity required')
    args.output_dir.mkdir(parents=True)
    started = time.time()
    deadline = time.perf_counter() + 9000
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}
    save(args.output_dir / 'queue.identity.json', {
        'pid': os.getpid(), 'started_epoch': started, 'source_commit': args.source_commit,
        'scope': 'Real/UQ-stage LOAO; shared Sim with historical Real ancestry, not full-pipeline holdout',
        'fold_order': [0, 5, 10, 15, 20], 'wall_cap_seconds': 9000,
        'gate': {'pooled_fitted_final_min': .05, 'positive_folds_min': 4, 'worst_fold_floor': -.03},
        'old_discovery_gate_unchanged': True, 'submission_authorized': False,
    })

    def command(script, options, log, cap):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError('stage confirmation budget exhausted')
        with log.open('w', encoding='utf-8') as output:
            process = subprocess.Popen([sys.executable, str(ROOT / script), *map(str, options)],
                                       stdout=output, stderr=subprocess.STDOUT, env=env)
            save(args.output_dir / 'queue.status.json', {
                'status': 'RUNNING', 'child_pid': process.pid, 'script': script,
                'log': str(log), 'source_commit': args.source_commit,
            })
            try:
                result = process.wait(timeout=min(cap, remaining))
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                raise
            if result:
                raise RuntimeError(f'{script} exited {result}; see {log}')

    for aoa in (0, 5, 10, 15, 20):
        fold = args.output_dir / f'held{aoa}'
        fold.mkdir()
        common = ['--execute', '--held-aoa', aoa, '--clean-manifest', args.clean_manifest,
                  '--real-dir', args.real_dir, '--device', 'cuda:0']
        command('code/training/train_b210_transfer_real4k.py', [*common, '--arm', 'C',
            '--source-commit', args.source_commit, '--start-sim', args.start_sim,
            '--start-sim-sha256', '628860c7d82239d942895f08a6fd72ede81feb6a7cad4de4e32e110ed2742b9b',
            '--start-metadata', args.start_metadata,
            '--start-metadata-sha256', '282307574088299de967a7d64597a417150dd81cb18ea7b8d2897ee29b00e96f',
            '--output-dir', fold / 'point', '--wall-budget-seconds', 1800], fold / 'point.log', 1900)
        point_receipt = fold / 'point/point_endpoint.json'
        for label, initialization in (('fresh', 'fresh'), ('inherit', 'point')):
            command('code/training/train_b210_transfer_uq.py', [*common, '--label', label,
                '--point-label', 'C', '--initialization', initialization,
                '--point-receipt', point_receipt, '--point-receipt-sha256', replay.sha256(point_receipt),
                '--output-dir', fold / label, '--wall-budget-seconds', 900], fold / f'{label}.log', 1000)
        scratch = fold / 'scratch'; scratch.mkdir()
        command('code/evaluation/evaluate_b210_transfer.py', [*common,
            '--baseline-zip', args.baseline_zip, '--expected-labels', 'fresh', 'inherit',
            '--study', 'uq-initialization', '--compare', 'inherit:fresh',
            '--uq-receipt', f'fresh={fold / "fresh/uq_endpoint.json"}',
            '--uq-receipt', f'inherit={fold / "inherit/uq_endpoint.json"}',
            '--output', fold / 'evaluation.json', '--scratch-dir', scratch,
            '--wall-budget-seconds', 600], fold / 'evaluation.log', 650)
        result = json.loads((fold / 'evaluation.json').read_text())
        print(json.dumps({'held_aoa': aoa, 'delta': result['contrasts']['inherit-fresh']}), flush=True)

    official = replay.import_file(replay.SCORER, 'stage_confirmation_scoring')
    results = {str(a): json.loads((args.output_dir / f'held{a}/evaluation.json').read_text())
               for a in (0, 5, 10, 15, 20)}
    pooled = {label: replay.finalize(official, replay.merge_states(
        [r['arms'][label]['pooled_state'] for r in results.values()])) for label in ('fresh', 'inherit')}
    deltas = {a: r['contrasts']['inherit-fresh']['fixed_time_fitted_final'] for a, r in results.items()}
    delta = pooled['inherit']['fixed_time_fitted_final'] - pooled['fresh']['fixed_time_fitted_final']
    point_equal = all(r['arms']['fresh']['prediction_sha256'] == r['arms']['inherit']['prediction_sha256']
                      for r in results.values())
    passed = point_equal and delta >= .05 and sum(v > 0 for v in deltas.values()) >= 4 and min(deltas.values()) >= -.03
    decision = {'decision': 'PASS_STAGE_CONFIRMATION_NOT_RELEASE' if passed else 'STOP_UQ_INHERIT_STAGE_CONFIRMATION',
        'pooled_metrics': pooled, 'pooled_fitted_final_delta': delta, 'by_aoa_fitted_final_delta': deltas,
        'point_predictions_bitwise_within_each_fold': point_equal,
        'scope': 'Real/UQ-stage holdout only; shared Sim has historical Real access',
        'independent_hidden_data': False, 'official_improvement_claimed': False,
        'release_verified': False, 'elapsed_seconds': time.time() - started}
    save(args.output_dir / 'decision.json', decision)
    save(args.output_dir / 'queue.status.json', {'status': 'TERMINAL', **decision})
    return decision


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output-dir', 'start-sim', 'start-metadata', 'clean-manifest', 'real-dir', 'baseline-zip'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2))
    except Exception as error:
        if args.output_dir.is_dir():
            save(args.output_dir / 'queue.failure.json', {'error': repr(error), 'status': 'EXECUTION_FAILURE_NOT_NEGATIVE_SCORE'})
        raise
