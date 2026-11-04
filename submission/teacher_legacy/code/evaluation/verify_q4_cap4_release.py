"""Verify one fixed Q4-weight / original-runtime / conditional-thread-cap ZIP."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(os.environ.get('REALPDE_Q4_RELEASE_ROOT', '/root/roy_route_20260910/q4_cap4_release'))
INPUT = Path('/root/autodl-tmp/spuq2_a800/artifacts/clean716_input_cache.npy')
TRUTH = Path('/root/autodl-tmp/real_time_release_20260908/time_replay_truth_20260907.npz')


def worker(args):
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / args.arm))
    import submission
    x = np.load(INPUT, mmap_mode='r')
    if args.mode == 'quality':
        import evaluate_b210_transfer as ev
        ev.TIME_SCORE = 92.03266035232576
        scorer = ev.import_file(ev.SCORER, 'q4_cap4_scorer')
        with np.load(TRUTH) as truth:
            y, labels = truth['truth'], truth['labels'].astype(str)
        output = submission.predict(x.copy())
        states = {s: ev.metric_state(scorer, output['prediction'][labels == s], y[labels == s],
            output['lower'][labels == s], output['upper'][labels == s]) for s in dict.fromkeys(labels)}
        print(json.dumps({'scores': ev.finalize(scorer, ev.merge_states(list(states.values()))),
                          'states': states, 'threads': torch.get_num_threads()}), flush=True)
        return
    if args.mode == 'cpu':
        assert not torch.cuda.is_available()
    else:
        warm = submission.predict(x[:16].copy())
        torch.cuda.synchronize()
        del warm
    batch = x[:args.n].copy()
    start = time.perf_counter()
    output = submission.predict(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    assert set(output) == {'prediction', 'lower', 'upper'}
    assert all(v.shape == batch.shape and v.dtype == np.float32 and np.isfinite(v).all()
               and np.all(v[..., 2] == 0) for v in output.values())
    assert np.all(output['lower'] <= output['upper'])
    checks = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in output.items()}
    submission.predict(x[:1].copy())
    assert all(checks[k] == hashlib.sha256(v.tobytes()).hexdigest() for k, v in output.items())
    print(json.dumps({'arm': args.arm, 'n': args.n, 'threads': torch.get_num_threads(),
        'predict_seconds': seconds, 'contract': True, 'retained_arrays_unchanged': True,
        'cuda': torch.cuda.is_available()}), flush=True)


def main():
    out = ROOT / 'verification'
    out.mkdir(exist_ok=False)
    begun = time.monotonic()
    env = {k: v for k, v in os.environ.items() if k not in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}

    def call(arm, mode, n, name, extra=None):
        p = subprocess.run([sys.executable, __file__, '--arm', arm, '--mode', mode, '--n', str(n)],
            env={**env, **(extra or {})}, capture_output=True, text=True,
            timeout=min(120, max(1, 1200 - (time.monotonic() - begun))))
        (out / (name + '.log')).write_text(p.stdout + '\n' + p.stderr)
        assert p.returncode == 0, p.stderr
        value = json.loads(p.stdout)
        if arm in ('candidate', 'old_cap4'):
            assert value['threads'] == 4
        return value

    cpu = call('candidate', 'cpu', 1, 'cpu', {'CUDA_VISIBLE_DEVICES': ''})
    quality = {a: [call(a, 'quality', 716, f'quality_{a}_{r}') for r in range(2)]
               for a in ('q4_original', 'candidate')}
    baseline = call('baseline', 'quality', 716, 'quality_baseline')
    keys = ('rel_l2_score', 'tke_rel_l2_score', 'mvpe_rel_l2_score', 'sps_score')
    deltas = [{k: quality['candidate'][r]['scores'][k] - quality['q4_original'][r]['scores'][k] for k in keys} for r in range(2)]
    numerical = all(abs(v) <= 1e-4 for d in deltas for v in d.values())
    gain = quality['candidate'][0]['scores']['fixed_time_fitted_final'] - baseline['scores']['fixed_time_fitted_final']
    (out / 'quality.json').write_text(json.dumps({'quality': quality, 'baseline': baseline,
        'deltas_vs_original_q4': deltas, 'fitted_quality_gain_vs_918343': gain, 'numerical_pass': numerical}, indent=2))
    assert numerical and gain > 0, 'quality identity or positive quality gate failed'
    timing, ratios = [], {}
    settings = [(False, n, 'baseline') for n in (16, 716)] + [(True, n, 'baseline') for n in (1, 16, 64, 716)] + [(True, 716, 'old_cap4')]
    for forced, n, reference in settings:
        group = f'{forced}_{n}_{reference}'
        for repeat in range(3):
            for arm in ((reference, 'candidate') if repeat % 2 == 0 else ('candidate', reference)):
                row = call(arm, 'time', n, f'time_{group}_{repeat}_{arm}',
                    {'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'} if forced else None)
                row.update(group=group, repeat=repeat)
                timing.append(row)
        med = {a: statistics.median(r['predict_seconds'] for r in timing if r['group'] == group and r['arm'] == a)
               for a in (reference, 'candidate')}
        ratios[group] = {**med, 'candidate_over_reference': med['candidate'] / med[reference]}
        print(json.dumps({'group': group, **ratios[group]}), flush=True)
    passed = all(v['candidate_over_reference'] <= (.95 if k == 'False_716_baseline' else 1.05) for k, v in ratios.items())
    result = {'decision': 'PASS_Q4_CAP4_RELEASE' if passed else 'STOP_Q4_CAP4_RELEASE',
        'quality_deltas_vs_original_q4': deltas, 'fitted_quality_gain_vs_918343': gain,
        'timing': timing, 'ratios': ratios, 'cpu': cpu, 'seconds': time.monotonic() - begun,
        'scope': 'Exact 919767 weights, runtime bound by launch hashes; local timing, official Time not guaranteed',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / 'RESULT.json').write_text(json.dumps(result, indent=2))
    print(result['decision'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=('baseline', 'q4_original', 'candidate', 'old_cap4'))
    p.add_argument('--mode', choices=('cpu', 'quality', 'time'))
    p.add_argument('--n', type=int)
    args = p.parse_args()
    worker(args) if args.arm else main()
