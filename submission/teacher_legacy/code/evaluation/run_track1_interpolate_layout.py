"""Preregistered local-layout timing screen, followed by numerical equivalence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import zipfile

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--arm', choices=['base', 'candidate'])
p.add_argument('--n', type=int, default=716)
p.add_argument('--score', action='store_true')
p.add_argument('--direct-pinned', action='store_true')
a = p.parse_args()
r = a.root
INPUT = '/root/autodl-tmp/spuq2_a800/artifacts/clean716_input_cache.npy'
BASE = '/root/autodl-tmp/track1_night_20260909/candidate.zip'
if a.arm:
    import numpy as np
    import torch
    torch.set_num_threads(4)
    assert torch.__version__ == '2.2.2+cu121'
    x = np.load(INPUT, mmap_mode='r')[:a.n].copy()
    sys.path.insert(0, str(r / a.arm))
    import submission
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = submission.predict(x)
    torch.cuda.synchronize()
    ready = time.perf_counter()
    result = {'arm': a.arm, 'n': a.n, 'predict_seconds': ready-start, 'ready_clock': ready}
    if a.score:
        import evaluate_b210_transfer as ev
        truth = np.load('/root/autodl-tmp/real_time_release_20260908/time_replay_truth_20260907.npz')['truth']
        assert set(out) == {'prediction', 'lower', 'upper'}
        assert all(v.shape == truth.shape and np.isfinite(v).all() for v in out.values())
        assert np.all(out['lower'] <= out['upper'])
        official = ev.import_file(ev.SCORER, 'layout_scorer')
        states = [ev.metric_state(official, out['prediction'][i:i+16], truth[i:i+16],
                                 out['lower'][i:i+16], out['upper'][i:i+16]) for i in range(0, len(x), 16)]
        result['metrics'] = ev.finalize(official, ev.merge_states(states))
        saved = {k:hashlib.sha256(v[:16].tobytes()).hexdigest() for k,v in out.items()}
        submission.predict(x[:1].copy())
        assert all(hashlib.sha256(v[:16].tobytes()).hexdigest()==saved[k] for k,v in out.items())
    print(json.dumps(result), flush=True)
    raise SystemExit

r.mkdir(exist_ok=True)
assert hashlib.sha256(Path(BASE).read_bytes()).hexdigest() == 'fe80d3d2c5b24bc9c362839be4150524e9df07e0214f5fec3a11bc56f18d4ffa'
for arm, zpath in [('base', BASE), ('candidate', r/'candidate.zip')]:
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        dest = r/arm
        dest.mkdir(exist_ok=False)
        z.extractall(dest)
with zipfile.ZipFile(BASE) as b, zipfile.ZipFile(r/'candidate.zip') as c:
    assert b.namelist() == c.namelist()
    expected = ['submission.py'] if a.direct_pinned else ['models/ensemble_unet2d.py', 'models/unet2d.py']
    assert sorted(n for n in b.namelist() if b.read(n) != c.read(n)) == expected
deadline = time.monotonic() + 1800
runs = []

def run(arm, n, score=False):
    assert time.monotonic() < deadline
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    start = time.perf_counter()
    command = [sys.executable, __file__, '--root', str(r), '--arm', arm, '--n', str(n)]
    if score:
        command.append('--score')
    cp = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert cp.returncode == 0, cp.stderr
    row = json.loads(cp.stdout)
    row['process_to_ready_seconds'] = row.pop('ready_clock') - start
    return row

def summarize(rows):
    summary = {}
    for n in sorted(set(x['n'] for x in rows)):
        summary[str(n)] = {}
        for metric in ('predict_seconds', 'process_to_ready_seconds'):
            vals = {arm: [x[metric] for x in rows if x['n']==n and x['arm']==arm] for arm in ('base','candidate')}
            meds = {arm: statistics.median(v) for arm,v in vals.items()}
            summary[str(n)][metric] = {'values':vals, 'median':meds, 'gain':1-meds['candidate']/meds['base']}
    return summary

for arm in ('base', 'candidate', 'candidate', 'base'):
    runs.append(run(arm, 716))
    (r/'timing.json').write_text(json.dumps(runs, indent=2))
screen = summarize(runs)
if screen['716']['predict_seconds']['gain'] < .03:
    (r/'decision.json').write_text(json.dumps({'decision':'STOP_DIRECT_PINNED_NO_GAIN' if a.direct_pinned else 'STOP_LAYOUT_NO_GAIN','summary':screen}, indent=2))
    raise SystemExit
for n in (716, 1, 16, 63, 64):
    for arm in (('base','candidate') if n==716 else ('base','candidate','candidate','base','base','candidate')):
        runs.append(run(arm, n))
        (r/'timing.json').write_text(json.dumps(runs, indent=2))
summary = summarize(runs)
timing_pass = (summary['716']['predict_seconds']['gain'] >= .05
               and summary['716']['process_to_ready_seconds']['gain'] > 0
               and all(v['gain'] >= -.05 for s in summary.values() for v in s.values()))
scores = []
if timing_pass:
    for arm in ('base','candidate','candidate','base','base','candidate'):
        scores.append(run(arm, 716, score=True))
        (r/'scores.json').write_text(json.dumps(scores, indent=2))
keys = ['rel_l2_score','tke_rel_l2_score','mvpe_rel_l2_score','sps_score']
deltas = [{k: next(x for x in scores[i:i+2] if x['arm']=='candidate')['metrics'][k]
              - next(x for x in scores[i:i+2] if x['arm']=='base')['metrics'][k] for k in keys}
          for i in range(0,len(scores),2)]
passed = timing_pass and len(deltas)==3 and all(abs(v)<=1e-5 for d in deltas for v in d.values())
(r/'decision.json').write_text(json.dumps({'decision':('PASS_DIRECT_PINNED_RUNTIME' if passed else 'STOP_DIRECT_PINNED_GATE') if a.direct_pinned else ('PASS_LAYOUT_RUNTIME' if passed else 'STOP_LAYOUT_GATE'),
    'summary':summary, 'timing_pass':timing_pass, 'score_deltas':deltas, 'score_tolerance':1e-5}, indent=2))
