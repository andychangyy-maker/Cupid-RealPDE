"""One fixed training run. All learned dependencies are produced inside output_dir."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def schedule(track):
    if track != 2:
        raise ValueError('Track1 calibration ancestry is still being bound; only Track2 is executable.')
    stages = [('teacher', name) for name in 'ABC']
    stages += [('cache', 'full'), ('closure', 'full')]
    stages += [('sim', stage) for stage in ('pretext', 'standard1', 'standard2')]
    for angle in (0, 5, 10, 15, 20):
        stages += [('point', str(angle)), ('shape', str(angle)), ('mean_cache', str(angle))]
    stages += [('point', 'full'), ('native_uq', 'full'), ('shape', 'full'), ('mean', 'full')]
    stages += [('tail', stage) for stage in ('uq8000', 'reflection4000', 'blend')]
    return stages + [('export', 'full')]


def inspect_data(data_dir):
    data = Path(data_dir).resolve()
    rows = json.loads((ROOT / 'assets/manifest.json').read_text())
    expected_real = {r['file'] for r in rows}
    real = {p.name for p in (data / 'train_real').glob('*.h5')}
    if real != expected_real:
        raise ValueError(f'Real file set differs: missing={expected_real-real}, extra={real-expected_real}')
    sim = sorted((data / 'train_sim').glob('*.h5'))
    if len(sim) != 100:
        raise ValueError('Expected the 100 released Sim trajectories')
    expected_stats = hashlib.sha256((ROOT / 'assets/mean_std_real.pt').read_bytes()).hexdigest()
    stats_sha = hashlib.sha256((data / 'mean_std_real.pt').read_bytes()).hexdigest()
    if stats_sha != expected_stats:
        raise ValueError('Official normalization file differs from the recovered training contract')
    import h5py
    for row in rows:
        with h5py.File(data / 'train_real' / row['file'], 'r') as stream:
            if len(stream['u']) != row['frames'] or int(stream['aoa'][()]) != row['aoa']:
                raise ValueError('Public Real metadata mismatch: ' + row['file'])
    return {'real_files': len(real), 'sim_files': len(sim), 'real_stats_sha256': stats_sha,
            'excluded': '7575_0.h5', 'prefix_windows': sum(len(r['train_starts']) for r in rows),
            'tail_windows': sum(len(range(r['val_boundary'], r['frames'] - 39, 20)) for r in rows)}


def train(data_dir, output_dir, device, track=2):
    stages = schedule(track)
    if str(device) not in ('cuda', 'cuda:0'):
        raise ValueError('Full training requires the assigned single CUDA GPU')
    data, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty output directory; historical artifacts are never resumed.')
    source_data = inspect_data(data)
    import torch
    if torch.__version__.split('+')[0] != '2.4.1' or torch.version.cuda != '12.4':
        raise RuntimeError('Run with the pinned PyTorch 2.4.1 / CUDA 12.4 environment')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU')
    output.mkdir(parents=True, exist_ok=True)
    work = output / 'training'
    (work / 'logs').mkdir(parents=True)
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       PYTHONUNBUFFERED='1')
    started = time.monotonic()
    report = {'track': track, 'data': source_data, 'stages': [], 'complete': False,
              'budget_seconds': 24 * 3600, 'network_isolation': 'must be enforced by docker --network none'}
    report_path = output / 'training_report.json'
    try:
        for index, (stage, scope) in enumerate(stages, 1):
            remaining = 24 * 3600 - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('Official 24-hour training limit reached')
            command = [sys.executable, '-u', str(ROOT / 'stages.py'), stage, '--scope', scope,
                       '--track', str(track), '--data', str(data), '--work', str(work)]
            record = {'index': index, 'stage': stage, 'scope': scope, 'complete': False}
            report['stages'].append(record)
            report_path.write_text(json.dumps(report, indent=2))
            print(json.dumps({'event': 'stage_started', **record}), flush=True)
            stage_started = time.monotonic()
            log = work / 'logs' / f'{index:02d}_{stage}_{scope}.log'
            with log.open('x') as stream:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                               stderr=subprocess.STDOUT, timeout=remaining, check=True)
            record.update(complete=True, seconds=time.monotonic() - stage_started, log=str(log))
            print(json.dumps({'event': 'stage_completed', **record}), flush=True)
        report['complete'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        reports = [json.loads(p.read_text()) for p in (work / 'resources').glob('*.json')]
        report['peak_allocated_bytes'] = max((r['peak_allocated_bytes'] for r in reports), default=None)
        report['peak_reserved_bytes'] = max((r['peak_reserved_bytes'] for r in reports), default=None)
        report_path.write_text(json.dumps(report, indent=2))
