"""One Sim -> Real -> point comparison after an admitted hardware preflight."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(args):
    root = args.run_dir
    preflight = json.loads(args.preflight_receipt.read_text())
    if preflight['decision'] != 'PASS_HARDWARE_PREFLIGHT':
        raise ValueError('Hardware preflight has not passed')
    with (root / 'queue.pid').open('x') as handle:
        handle.write(str(os.getpid()))
    deadline = time.time() + 7800 - preflight['wall_seconds']
    completed = {}

    def save(name, value):
        path = root / name
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)

    def stage(label, entry, argv, cap, receipt):
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError('Whole-queue time budget exhausted')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU occupied before ' + label)
        command = [sys.executable, '-u', str(SOURCE / entry), '--execute',
                   *map(str, argv), '--wall-budget-seconds', str(min(cap, remaining))]
        save('queue.status.json', {'status': 'RUNNING', 'phase': label, 'argv': command,
                                   'deadline_epoch': deadline, 'completed': completed})
        with (root / (label + '.log')).open('x') as handle:
            result = subprocess.run(command, cwd=SOURCE, stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=min(cap + 15, remaining))
        (root / (label + '.exit_code.txt')).write_text(str(result.returncode))
        if result.returncode:
            raise RuntimeError(label + ' failed; no automatic retry')
        completed[label] = {'receipt': str(receipt), 'sha256': sha(receipt)}
        save('completed_receipts.json', completed)

    save('queue.identity.json', {'source_commit': args.source_commit, 'pid': os.getpid(),
        'started_epoch': time.time(), 'deadline_epoch': deadline, 'launcher_sha256': sha(__file__),
        'preflight_receipt_sha256': sha(args.preflight_receipt)})
    trainer = 'code/training/train_b256_offline_teacher.py'
    common = ['--source-commit', args.source_commit]
    sim_receipt = root / 'sim/sim_endpoint.json'
    stage('sim', trainer, [*common, '--phase', 'sim', '--sim-dir', args.data_dir / 'sim',
        '--teacher-cache', args.teacher_cache, '--output-dir', root / 'sim',
        '--preflight-receipt', args.preflight_receipt,
        '--preflight-receipt-sha256', sha(args.preflight_receipt)], 4800, sim_receipt)
    point_receipt = root / 'real/point_endpoint.json'
    stage('real', trainer, [*common, '--phase', 'real', '--real-dir', args.data_dir / 'real',
        '--clean-manifest', args.clean_manifest, '--output-dir', root / 'real',
        '--sim-receipt', sim_receipt, '--sim-receipt-sha256', sha(sim_receipt)], 2400, point_receipt)
    controls = []
    for name in ('reference_a', 'reference_old_c'):
        if getattr(args, name):
            controls += ['--' + name.replace('_', '-'), getattr(args, name)]
    evaluation = root / 'evaluation.json'
    stage('evaluation', 'code/evaluation/evaluate_b256_offline_teacher.py', [
        '--candidate-receipt', point_receipt, '--clean-manifest', args.clean_manifest,
        '--real-dir', args.data_dir / 'real', '--output', evaluation, *controls], 300, evaluation)
    decision = json.loads(evaluation.read_text())['decision']
    save('queue.status.json', {'status': 'TERMINAL', 'decision': decision,
                               'completed_epoch': time.time(), 'completed': completed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'preflight-receipt', 'data-dir', 'teacher-cache', 'clean-manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--reference-a', type=Path)
    parser.add_argument('--reference-old-c', type=Path)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        (args.run_dir / 'queue.failure.json').write_text(json.dumps({
            'type': type(error).__name__, 'message': str(error), 'epoch': time.time()}, indent=2))
        (args.run_dir / 'queue.exit_code.txt').write_text('1')
        raise
    else:
        (args.run_dir / 'queue.exit_code.txt').write_text('0')
