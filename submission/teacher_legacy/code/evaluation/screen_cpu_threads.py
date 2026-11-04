"""Source-driven CPU-thread sensitivity test; this is not the official harness."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/time_audit_919767_20260909')
INPUT = '/root/autodl-tmp/spuq2_a800/artifacts/clean716_input_cache.npy'


def worker(arm, threads, n):
    import numpy as np
    import torch
    if threads == 'four':
        torch.set_num_threads(4)
    start = time.perf_counter()
    sys.path.insert(0, str(ROOT / arm))
    import submission
    x = np.load(INPUT, mmap_mode='r')[:n].copy()
    warm = submission.predict(x[:16].copy())
    torch.cuda.synchronize()
    del warm
    ready = time.perf_counter()
    output = submission.predict(x)
    torch.cuda.synchronize()
    done = time.perf_counter()
    checksum = sum(float(np.sum(v, dtype=np.float64)) for v in output.values())
    assert np.isfinite(checksum)
    print(json.dumps({'arm': arm, 'threads_mode': threads, 'threads': torch.get_num_threads(),
                      'interop_threads': torch.get_num_interop_threads(), 'n': n,
                      'warm_seconds': ready-start, 'predict_seconds': done-ready,
                      'consume_seconds': time.perf_counter()-done, 'checksum': checksum,
                      'affinity_cpus': len(os.sched_getaffinity(0))}), flush=True)


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env.pop(key, None)
    rows = []
    begin = time.monotonic()
    for n in (16, 716):
        for repeat in range(3):
            order = [('restored', 'default'), ('pinned', 'default'),
                     ('restored', 'four'), ('pinned', 'four')]
            if repeat % 2:
                order.reverse()
            for arm, threads in order:
                left = 1200 - (time.monotonic()-begin)
                assert left > 0
                cp = subprocess.run([sys.executable, __file__, '--arm', arm, '--threads', threads,
                                     '--n', str(n)], capture_output=True, text=True, env=env,
                                    timeout=min(120, left))
                (output/f'{n}_{repeat}_{arm}_{threads}.log').write_text(cp.stdout+'\n'+cp.stderr)
                assert cp.returncode == 0, cp.stderr
                row = json.loads(cp.stdout)
                row['repeat'] = repeat
                rows.append(row)
                (output/'rows.json').write_text(json.dumps(rows, indent=2))
                print(json.dumps(row), flush=True)
    summary = {}
    for n in (16, 716):
        summary[n] = {f'{arm}_{threads}': statistics.median(r['predict_seconds'] for r in rows
                       if r['n']==n and r['arm']==arm and r['threads_mode']==threads)
                      for arm in ('restored','pinned') for threads in ('default','four')}
    (output/'RESULT.json').write_text(json.dumps({'scope':'CPU sensitivity, not official attribution',
        'rows':rows,'summary':summary,'seconds':time.monotonic()-begin}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path)
    p.add_argument('--arm', choices=('restored','pinned'))
    p.add_argument('--threads', choices=('default','four'))
    p.add_argument('--n', type=int)
    a = p.parse_args()
    worker(a.arm,a.threads,a.n) if a.arm else main(a.output)
