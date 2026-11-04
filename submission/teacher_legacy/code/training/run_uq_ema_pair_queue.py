"""One UQ3200 raw/EMA999 trajectory and the preregistered clean716 comparison."""
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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    root = args.run_dir
    preflight = json.loads(args.preflight_receipt.read_text())
    if preflight['decision'] != 'PASS_HARDWARE_PREFLIGHT':
        raise ValueError('Hardware preflight has not passed')
    with (root / 'queue.pid').open('x') as handle:
        handle.write(str(os.getpid()))
    deadline = time.time() + 1800

    def save(name, value):
        path = root / name
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)

    def stage(label, entry, argv, cap):
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError('Whole-queue time budget exhausted')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU occupied before ' + label)
        command = [sys.executable, '-u', str(SOURCE / entry), '--execute',
                   *map(str, argv), '--wall-budget-seconds', str(min(cap, remaining))]
        save('queue.status.json', {'status': 'RUNNING', 'phase': label,
                                   'argv': command, 'deadline_epoch': deadline})
        with (root / (label + '.log')).open('x') as handle:
            result = subprocess.run(command, cwd=SOURCE, stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=min(cap + 15, remaining))
        (root / (label + '.exit_code.txt')).write_text(str(result.returncode))
        if result.returncode:
            raise RuntimeError(label + ' failed; no automatic retry')

    save('queue.identity.json', {'source_commit': args.source_commit, 'pid': os.getpid(),
        'started_epoch': time.time(), 'deadline_epoch': deadline, 'launcher_sha256': sha(__file__),
        'preflight_receipt_sha256': sha(args.preflight_receipt)})
    stage('uq', 'code/training/train_b210_transfer_uq.py', [
        '--uq-ema999-pair', '--label', 'raw', '--point-label', 'C', '--initialization', 'point',
        '--objective', 'sharp2', '--source-commit', args.source_commit,
        '--point-receipt', args.point_receipt, '--point-receipt-sha256', sha(args.point_receipt),
        '--clean-manifest', args.clean_manifest, '--real-dir', args.real_dir,
        '--output-dir', root / 'uq'], 1200)
    (root / 'evaluation_scratch').mkdir()
    stage('evaluation', 'code/evaluation/evaluate_b210_transfer.py', [
        '--study', 'uq-ema', '--expected-labels', 'raw', 'ema', '--compare', 'ema:raw',
        '--uq-receipt', 'raw=' + str(root / 'uq/uq_endpoint.json'),
        '--uq-receipt', 'ema=' + str(root / 'uq/uq_ema999_endpoint.json'),
        '--baseline-zip', args.baseline_zip, '--clean-manifest', args.clean_manifest,
        '--real-dir', args.real_dir, '--output', root / 'evaluation.json',
        '--scratch-dir', root / 'evaluation_scratch'], 600)
    result = json.loads((root / 'evaluation.json').read_text())
    arms = result['arms']
    sps = arms['ema']['metrics']['sps_score'] - arms['raw']['metrics']['sps_score']
    final = (arms['ema']['metrics']['fixed_time_fitted_final']
             - arms['v3']['metrics']['fixed_time_fitted_final'])
    aoa_sps = {a: arms['ema']['by_aoa'][a]['sps_score'] - arms['raw']['by_aoa'][a]['sps_score']
               for a in ('0', '5', '10', '15', '20')}
    aoa_final = {a: arms['ema']['by_aoa'][a]['fixed_time_fitted_final']
                - arms['v3']['by_aoa'][a]['fixed_time_fitted_final'] for a in aoa_sps}
    point_equal = arms['ema']['prediction_sha256'] == arms['raw']['prediction_sha256']
    direction = point_equal and sps >= .10 and sum(v > 0 for v in aoa_sps.values()) >= 3
    candidate = direction and final >= .10 and sum(v > 0 for v in aoa_final.values()) >= 3
    decision = {'decision': ('PASS_EXPLORATORY_NEEDS_INDEPENDENT_VALIDATION' if candidate
                             else 'STOP_UQ_EMA999_ENDPOINT_CONFIG'),
                'ema_minus_raw_sps': sps, 'ema_minus_v3_fixed_time_fitted_final': final,
                'by_aoa_sps': aoa_sps, 'by_aoa_final': aoa_final,
                'point_prediction_bitwise': point_equal, 'direction_pass': direction,
                'candidate_pass': candidate, 'release_verified': False,
                'official_improvement_claimed': False}
    save('decision.json', decision)
    save('queue.status.json', {'status': 'TERMINAL', **decision, 'completed_epoch': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'preflight-receipt', 'point-receipt', 'baseline-zip', 'clean-manifest', 'real-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
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
