"""Single bounded queue for the preregistered teacher-from-start 2x2 study."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/dev/shm/b210_teacher_from_start_20260906')
SOURCE = ROOT / 'source'
PYTHON = '/root/autodl-tmp/envs/realpde/bin/python'
ASSETS = Path('/root/autodl-tmp/b210_transfer_20260905/assets')
MANIFEST = '/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json'
DATA = Path('/root/autodl-tmp/realpde/data')
OFF = Path('/dev/shm/b210_held_residual_20260906/pure_sim/sim_endpoint.json')
ARMS = 'ABCD'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(name, value):
    path = ROOT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def matched(receipts, keys):
    for key in keys:
        if any(item[key] != receipts[0][key] for item in receipts[1:]):
            raise ValueError('paired field changed: ' + key)


def main():
    with (ROOT / 'queue.pid').open('x') as handle:
        handle.write(str(os.getpid()))
    identity = read(ROOT / 'launch_identity.json')
    deadline = time.time() + 12600
    save('queue.identity.json', {**identity, 'pid': os.getpid(), 'started_epoch': time.time(),
                               'deadline_epoch': deadline, 'launcher_sha256': sha(__file__)})
    sys.path[:0] = [str(SOURCE / 'code/training'), str(SOURCE / 'code/evaluation')]
    import train_b210_teacher_real as real
    import evaluate_b210_transfer as evaluation
    receipts = {}

    def stage(label, entry, args, cap, receipt):
        if deadline - time.time() < cap:
            raise TimeoutError('insufficient fixed whole-queue budget before ' + label)
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU occupied before ' + label)
        command = [PYTHON, '-u', str(SOURCE / entry), '--execute', *map(str, args),
                   '--wall-budget-seconds', str(cap)]
        save('queue.status.json', {'status': 'RUNNING', 'phase': label, 'argv': command,
                                   'started_epoch': time.time(), 'deadline_epoch': deadline})
        with (ROOT / (label + '.log')).open('x') as handle:
            result = subprocess.run(command, cwd=SOURCE, stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=min(cap + 15, deadline - time.time()))
        (ROOT / (label + '.exit_code.txt')).write_text(str(result.returncode))
        if result.returncode:
            raise RuntimeError(label + ' failed; no automatic retry')
        receipts[label] = {'path': str(receipt), 'sha256': sha(receipt)}
        save('completed_receipts.json', receipts)

    if sha(OFF) != '778430483e6e400c06f4de3c3a610b325dc998bbdf2117c2f45a8adaa0cef7c8':
        raise ValueError('sealed OFF receipt changed')
    off = read(OFF)
    real.validate_sim_receipt(off, 'C')
    if sha(off['checkpoint_path']) != off['checkpoint_sha256']:
        raise ValueError('sealed OFF checkpoint changed')
    on_path = ROOT / 'sim_on/sim_endpoint.json'
    stage('sim_on', 'code/training/train_b210_pure_sim_start.py', [
        '--sim-dir', DATA / 'sim', '--output-dir', ROOT / 'sim_on',
        '--source-commit', identity['source_commit'], '--teacher-cache', ASSETS / 'p1_teacher_cache.pt',
        '--teacher-cache-sha256', real.sim_start.TEACHER_CACHE_SHA256], 4200, on_path)
    on = read(on_path)
    real.validate_sim_receipt(on, 'A')
    matched([on, off], ['initial_model_state_sha256', 'sample_stream_sha256',
                        'normalizer_sha256', 'model_config_sha256', 'hgrad_scale_contract',
                        'optimizer', 'native_loss_config', 'ema_updates'])
    save('sim_pair.json', {'on_receipt_sha256': sha(on_path), 'off_receipt_sha256': sha(OFF),
                           'paired_fields_match': True, 'teacher_ancestry': 'unknown'})
    for arm in ARMS:
        sim_receipt = on_path if arm in 'AB' else OFF
        output = ROOT / ('point_' + arm)
        stage('point_' + arm, 'code/training/train_b210_teacher_real.py', [
            '--arm', arm, '--sim-receipt', sim_receipt, '--sim-receipt-sha256', sha(sim_receipt),
            '--clean-manifest', MANIFEST, '--real-dir', DATA / 'real', '--output-dir', output],
            1800, output / 'point_endpoint.json')
    points = [read(ROOT / ('point_' + arm) / 'point_endpoint.json') for arm in ARMS]
    matched(points, ['sample_stream_sha256', 'seed', 'updates', 'ema_decay', 'tail_read'])
    for arm in ARMS:
        point_receipt = ROOT / ('point_' + arm) / 'point_endpoint.json'
        output = ROOT / ('uq_' + arm)
        stage('uq_' + arm, 'code/training/train_b210_transfer_uq.py', [
            '--label', arm, '--initialization', 'fresh', '--point-receipt', point_receipt,
            '--point-receipt-sha256', sha(point_receipt), '--clean-manifest', MANIFEST,
            '--real-dir', DATA / 'real', '--output-dir', output], 900, output / 'uq_endpoint.json')
    (ROOT / 'scratch').mkdir()
    eval_args = ['--expected-labels', *ARMS, '--baseline-zip', ASSETS / 'track1_v3.zip',
                 '--clean-manifest', MANIFEST, '--real-dir', DATA / 'real',
                 '--output', ROOT / 'evaluation.json', '--scratch-dir', ROOT / 'scratch']
    for arm in ARMS:
        eval_args += ['--uq-receipt', arm + '=' + str(ROOT / ('uq_' + arm) / 'uq_endpoint.json')]
    stage('eval', 'code/evaluation/evaluate_b210_transfer.py', eval_args, 600, ROOT / 'evaluation.json')
    result = read(ROOT / 'evaluation.json')['arms']
    gates = {}
    metric = 'fixed_time_fitted_final'
    for arm in ARMS:
        delta = result[arm]['metrics'][metric] - result['v3']['metrics'][metric]
        aoas = {aoa: row[metric] - result['v3']['by_aoa'][aoa][metric]
                for aoa, row in result[arm]['by_aoa'].items()}
        gates[arm] = {'final_delta': delta, 'aoa_deltas': aoas,
                      'pass': delta >= .10 and sum(value > 0 for value in aoas.values()) >= 3}
    qualified = [arm for arm in ARMS if gates[arm]['pass']]
    point_signals = {}
    for left, right in [('C', 'A'), ('D', 'B'), ('B', 'A'), ('D', 'C')]:
        def point_delta(group):
            rows = {arm: result[arm]['metrics'] if group is None else result[arm]['by_aoa'][group]
                    for arm in [left, right, 'v3']}
            return (evaluation.fixed_time_final({**rows[left], 'sps_score': rows['v3']['sps_score']})
                    - evaluation.fixed_time_final({**rows[right], 'sps_score': rows['v3']['sps_score']}))
        delta = point_delta(None)
        aoas = {aoa: point_delta(aoa) for aoa in result['v3']['by_aoa']}
        point_signals[left + '-' + right] = {'point_only_final_delta': delta, 'aoa_deltas': aoas,
            'direction_signal': delta >= .05 and sum(value > 0 for value in aoas.values()) >= 3}
    save('decision.json', {'decision': 'CANDIDATE_FOR_CONFIRMATION' if qualified else 'STOP_RELEASE_THIS_CONFIG',
        'qualified': qualified, 'selected': max(qualified, key=lambda arm: gates[arm]['final_delta']) if qualified else None,
        'gates': gates, 'point_signals': point_signals, 'evaluation_sha256': sha(ROOT / 'evaluation.json'),
        'evidence': 'Exploratory clean716 only; independent confirmation and actual ZIP/A800 pending',
        'official_improvement_claimed': False, 'release_verified': False})
    save('queue.status.json', {'status': 'TERMINAL', 'completed_epoch': time.time(), 'receipts': receipts,
                               'decision_sha256': sha(ROOT / 'decision.json')})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save('queue.failure.json', {'type': type(error).__name__, 'message': str(error), 'epoch': time.time()})
        (ROOT / 'queue.exit_code.txt').write_text('1')
        raise
    else:
        (ROOT / 'queue.exit_code.txt').write_text('0')
