"""Run the fixed H800 Real-controller-TKE experiment once, without automatic retry."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[2]
ROOT = SOURCE.parent
ASSETS = Path('/root/autodl-tmp/b210_transfer_20260905')
REAL = Path('/root/autodl-tmp/realpde/data/real')
MANIFEST = Path('/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json')
CONTROL = Path('/root/autodl-tmp/uq_ema_20260906/uq/uq_endpoint.json')


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + '\n')


def main(commit):
    with (ROOT / 'queue.pid').open('x') as handle:
        handle.write(str(os.getpid()))
    deadline = time.time() + 3000
    save('queue.identity.json', {'pid': os.getpid(), 'source_commit': commit,
        'started_epoch': time.time(), 'deadline_epoch': deadline,
        'queue_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def stage(label, entry, arguments, cap):
        remaining = min(cap, deadline - time.time())
        if remaining <= 0:
            raise TimeoutError('Experiment budget exhausted')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU occupied before ' + label)
        command = [sys.executable, '-u', str(SOURCE / entry), '--execute',
                   *map(str, arguments), '--wall-budget-seconds', str(remaining)]
        save('queue.status.json', {'status': 'RUNNING', 'phase': label, 'argv': command})
        with (ROOT / (label + '.log')).open('x') as handle:
            result = subprocess.run(command, cwd=SOURCE, stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=min(remaining + 15, deadline - time.time()))
        (ROOT / (label + '.exit_code.txt')).write_text(str(result.returncode))
        if result.returncode:
            raise RuntimeError(label + ' failed; no automatic retry')

    common = ['--clean-manifest', MANIFEST, '--real-dir', REAL]
    stage('real', 'code/training/train_b210_transfer_real4k.py', [*common,
        '--arm', 'C', '--constant-controller-tke', '--source-commit', commit,
        '--start-sim', ASSETS / 'runs/causal_rebuild/causal_standard20k/model_10000.pth',
        '--start-sim-sha256', '628860c7d82239d942895f08a6fd72ede81feb6a7cad4de4e32e110ed2742b9b',
        '--start-metadata', ASSETS / 'runs/causal_rebuild/sim_binding.json',
        '--start-metadata-sha256', '282307574088299de967a7d64597a417150dd81cb18ea7b8d2897ee29b00e96f',
        '--output-dir', ROOT / 'real'], 1500)
    point = ROOT / 'real/point_endpoint.json'
    stage('uq', 'code/training/train_b210_transfer_uq.py', [*common,
        '--label', 'tke', '--point-label', 'C', '--initialization', 'point', '--objective', 'sharp2',
        '--point-receipt', point, '--point-receipt-sha256', hashlib.sha256(point.read_bytes()).hexdigest(),
        '--output-dir', ROOT / 'uq'], 900)
    (ROOT / 'evaluation_scratch').mkdir()
    stage('evaluation', 'code/evaluation/evaluate_b210_transfer.py', [*common,
        '--study', 'real-tke-constant', '--expected-labels', 'raw', 'tke', '--compare', 'tke:raw',
        '--uq-receipt', 'raw=' + str(CONTROL), '--uq-receipt', 'tke=' + str(ROOT / 'uq/uq_endpoint.json'),
        '--baseline-zip', ASSETS / 'assets/track1_v3.zip', '--output', ROOT / 'evaluation.json',
        '--scratch-dir', ROOT / 'evaluation_scratch'], 600)
    result = json.loads((ROOT / 'evaluation.json').read_text())
    arms = result['arms']
    delta = result['contrasts']['tke-raw']['fixed_time_fitted_final']
    aoas = {a: arms['tke']['by_aoa'][a]['fixed_time_fitted_final']
            - arms['raw']['by_aoa'][a]['fixed_time_fitted_final'] for a in ('0', '5', '10', '15', '20')}
    passed = delta >= .10 and sum(v > 0 for v in aoas.values()) >= 3
    decision = {'decision': ('PASS_EXPLORATORY_NEEDS_INDEPENDENT_VALIDATION' if passed
                             else 'STOP_REAL_TKE_CONSTANT_CONFIG'),
                'candidate_minus_control_fixed_time_fitted_final': delta, 'by_aoa_final': aoas,
                'pass': passed, 'official_improvement_claimed': False, 'release_verified': False}
    save('decision.json', decision)
    save('queue.status.json', {'status': 'TERMINAL', **decision, 'completed_epoch': time.time()})


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except Exception as error:
        save('queue.failure.json', {'type': type(error).__name__, 'message': str(error)})
        (ROOT / 'queue.exit_code.txt').write_text('1')
        raise
    else:
        (ROOT / 'queue.exit_code.txt').write_text('0')
