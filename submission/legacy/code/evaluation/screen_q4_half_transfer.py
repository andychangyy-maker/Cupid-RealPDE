"""One paired test of FP16 D2H followed by CPU FP32 conversion."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path('/root/roy_route_20260910/q4_half_transfer')


def main():
    import numpy as np
    import torch
    import evaluate_b210_transfer as ev
    torch.set_num_threads(4)
    begun = time.monotonic()
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    finite = bits[torch.isfinite(bits)]
    assert torch.equal(finite.float(), finite.cuda().float().cpu())
    before = ev.import_file(ROOT / 'reference/submission.py', 'transfer_before')
    after = ev.import_file(ROOT / 'candidate/submission.py', 'transfer_after')
    after._state = before._state
    x = np.load('/root/autodl-tmp/spuq2_a800/artifacts/clean716_input_cache.npy', mmap_mode='r')[:16].copy()
    left, right = before.predict(x), after.predict(x)
    maximum = {k: float(np.max(np.abs(left[k] - right[k]))) for k in left}
    assert max(maximum.values()) <= 2e-6, maximum
    before._state.clear()
    after._state.clear()
    del before, after, left, right
    import gc
    gc.collect(); torch.cuda.empty_cache()
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
    timing = []
    for n in (16, 716):
        for repeat in range(3):
            for arm in (('reference', 'candidate') if repeat % 2 == 0 else ('candidate', 'reference')):
                p = subprocess.run([sys.executable, __file__, '--arm', arm, '--n', str(n)],
                    env=env, capture_output=True, text=True,
                    timeout=min(90, max(1, 360 - (time.monotonic() - begun))))
                (ROOT / f'time_{n}_{repeat}_{arm}.log').write_text(p.stdout + '\n' + p.stderr)
                assert p.returncode == 0, p.stderr
                row = json.loads(p.stdout)
                row['repeat'] = repeat
                timing.append(row)
    ratios = {}
    for n in (16, 716):
        med = {a: statistics.median(r['predict_seconds'] for r in timing if r['n'] == n and r['arm'] == a)
               for a in ('reference', 'candidate')}
        ratios[n] = {**med, 'ratio': med['candidate'] / med['reference']}
    passed = max(v['ratio'] for v in ratios.values()) <= 1 and min(v['ratio'] for v in ratios.values()) <= .97
    result = {'decision': 'PASS_FULL_ZIP_VERIFICATION' if passed else 'STOP_HALF_TRANSFER',
              'max_abs_output_difference': maximum, 'finite_half_values_checked': len(finite),
              'ratios': ratios, 'timing': timing, 'seconds': time.monotonic() - begun}
    (ROOT / 'SCREEN_RESULT.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('decision', 'max_abs_output_difference', 'ratios', 'seconds')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=('reference', 'candidate'))
    p.add_argument('--n', type=int)
    args = p.parse_args()
    if args.arm:
        import verify_q4_cap4_release as verify
        verify.ROOT = ROOT
        args.mode = 'time'
        verify.worker(args)
    else:
        main()
