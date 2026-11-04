"""Explicit numerical gate after the original bitwise gate failed on base repeats."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--arm', choices=['base', 'candidate'])
a = p.parse_args()
r = a.root
if a.arm:
    import numpy as np
    import torch
    import evaluate_b210_transfer as ev
    torch.set_num_threads(4)
    sys.path.insert(0, str(r / a.arm))
    import submission
    # Fixed tensor isolates the only changed transfer operation from model numerics.
    for n in (1, 12, 16):
        t = torch.randn(3, n, 20, 32, 64, 3, device='cuda', dtype=torch.float32)
        plain = t.cpu().numpy()
        pinned = torch.empty(t.shape, dtype=torch.float32, pin_memory=True)
        pinned.copy_(t, non_blocking=True)
        torch.cuda.synchronize()
        assert np.array_equal(plain, pinned.numpy())
    x = np.load('/root/autodl-tmp/spuq2_a800/artifacts/clean716_input_cache.npy')
    truth = np.load('/root/autodl-tmp/real_time_release_20260908/time_replay_truth_20260907.npz')['truth']
    official = ev.import_file(ev.SCORER, 'pinned_official_scorer')
    out = submission.predict(x)
    states = []
    for start in range(0, len(x), 16):
        s = slice(start, start+16)
        states.append(ev.metric_state(official, out['prediction'][s], truth[s], out['lower'][s], out['upper'][s]))
    metrics = ev.finalize(official, ev.merge_states(states))
    print(json.dumps({'arm': a.arm, 'metrics': metrics, 'fixed_tensor_copy_bitwise': True}))
    raise SystemExit

runs = []
for arm in ('base', 'candidate', 'candidate', 'base', 'base', 'candidate'):
    cp = subprocess.run([sys.executable, __file__, '--root', str(r), '--arm', arm],
                        capture_output=True, text=True, timeout=180)
    assert cp.returncode == 0, cp.stderr
    runs.append(json.loads(cp.stdout))
    (r / 'numerical_progress.json').write_text(json.dumps(runs, indent=2))
    print(cp.stdout, flush=True)
keys = ['rel_l2_score', 'tke_rel_l2_score', 'mvpe_rel_l2_score', 'sps_score']
differences = [{k: runs[i if runs[i]['arm']=='candidate' else i+1]['metrics'][k] -
                    runs[i if runs[i]['arm']=='base' else i+1]['metrics'][k] for k in keys}
               for i in (0, 2, 4)]
passed = all(abs(v) <= 1e-5 for d in differences for v in d.values())
result = {'numerical_gate_pass': passed, 'paired_deltas': differences,
          'original_bitwise_gate': 'FAILED_AND_PRESERVED', 'score_tolerance': 1e-5,
          'scope': 'same weights; engineering numerical replay, not quality improvement'}
(r / 'numerical_result.json').write_text(json.dumps(result, indent=2))
if not passed:
    print('STOP_NUMERICAL_GATE', flush=True)
    raise SystemExit(2)

# Existing diagnostic timing has N16/716. Complete the three missing call sizes.
timings = json.loads((r / 'diagnostic_timing.json').read_text())
for n in (1, 63, 64):
    for arm in ('base', 'candidate', 'candidate', 'base', 'base', 'candidate'):
        start = time.perf_counter()
        cp = subprocess.run([sys.executable, str(r / 'run_track1_pinned_output.py'),
                             '--root', str(r), '--child', arm, '--n', str(n)],
                            capture_output=True, text=True, timeout=120)
        assert cp.returncode == 0, cp.stderr
        row = json.loads(cp.stdout)
        row.update(arm=arm, process_to_ready_seconds=row.pop('ready_clock')-start)
        timings.append(row)
        (r / 'complete_timing.json').write_text(json.dumps(timings, indent=2))
summary = {}
for n in (1, 16, 63, 64, 716):
    summary[str(n)] = {}
    for metric in ('predict_seconds', 'process_to_ready_seconds'):
        values = {arm: [x[metric] for x in timings if x['samples']==n and x['arm']==arm]
                  for arm in ('base', 'candidate')}
        med = {arm: statistics.median(v) for arm, v in values.items()}
        summary[str(n)][metric] = {'median': med, 'gain': 1-med['candidate']/med['base'], 'values': values}
no_regression = all(v['gain'] >= -.05 for s in summary.values() for v in s.values())
large_stable = max(summary['716']['predict_seconds']['values']['candidate']) < min(summary['716']['predict_seconds']['values']['base'])
result.update(summary=summary, timing_gate_pass=no_regression and large_stable,
              decision='PASS_NUMERICAL_RUNTIME_CANDIDATE' if no_regression and large_stable else 'STOP_RUNTIME_GATE')
(r / 'numerical_result.json').write_text(json.dumps(result, indent=2))
print(result['decision'], flush=True)
