"""Bounded same-weight graph test, including startup, tails and output ownership."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import zipfile

import evaluate_b210_transfer as replay

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--base', type=Path)
p.add_argument('--candidate', type=Path)
p.add_argument('--source-commit')
p.add_argument('--contract', choices=('cuda','cpu'))
a = p.parse_args()
source = Path(__file__).resolve().parents[2]
input_path = '/root/autodl-tmp/spuq2_a800/artifacts/clean716_input_cache.npy'

if a.contract:
    import numpy as np
    import torch
    assert torch.__version__ == '2.2.2+cu121'
    torch.set_num_threads(4)
    if a.contract == 'cpu': torch.cuda.is_available = lambda: False
    sys.path.insert(0, str(a.root/'candidate'))
    import submission
    x = np.load(input_path, mmap_mode='r')
    guard = None
    checks = []
    for n in ((1,) if a.contract=='cpu' else (16,1,63,64,716)):
        offset = 0 if n==716 else n*3
        y = submission.predict(x[offset:offset+n].copy())
        assert set(y)=={'prediction','lower','upper'}
        assert all(v.shape==(n,20,32,64,3) and v.dtype==np.float32 and np.isfinite(v).all()
                   and np.all(v[...,2]==0) for v in y.values())
        assert np.all(y['lower']<=y['upper'])
        if guard is not None:
            assert all(hashlib.sha256(v.tobytes()).hexdigest()==guard_hash[k] for k,v in guard.items())
        if n==16:
            guard=y; guard_hash={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in y.items()}
        checks.append(n)
    print(json.dumps({'device':a.contract,'sizes':checks,'contract':'PASS','different_inputs':True}))
    raise SystemExit

assert a.base and a.candidate and a.source_commit and len(a.source_commit)==40
assert replay.sha256(input_path)=='436dbc2db9bb63ef013db7a56774ae49d7acbebf3738ac1038923f3901fffbe6'
a.root.mkdir(exist_ok=False)
deadline = time.monotonic()+900
env = {**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
members = {}
for arm, path in (('base',a.base),('candidate',a.candidate)):
    runtime = a.root/arm; runtime.mkdir()
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None and set(z.namelist())==set((*replay.RUNTIME_FILES,'ensemble_unet2d.pth'))
        members[arm] = {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
        z.extractall(runtime)
assert {n for n,v in members['base'].items() if v!=members['candidate'][n]}=={'submission.py','models/ensemble_unet2d.py'}
assert members['base']['submission.py']=='86e8489308abd13ccde99289016a1185862b36fdc06d9971c5f5fb62ad9d4d78'
assert members['candidate']['submission.py']=='8e02bce2f7de87a6461157362b23744e00020cd533181a22792cee780254f6ec'
assert members['base']['models/ensemble_unet2d.py']=='5f18b5069fd2354da06166ae6aaa96a6faf35e10acc05dc1d6adbad720b0e9bd'
assert members['candidate']['models/ensemble_unet2d.py']=='db842254030af7265b143e32497ce6305994cdd5563f3b8291742c850522d96b'
inputs = {'base':str(a.base),'candidate':str(a.candidate),'base_sha256':replay.sha256(a.base),
          'candidate_sha256':replay.sha256(a.candidate),'source_commit':a.source_commit,'members':members}


def save(name, value):
    (a.root/name).write_text(json.dumps(value,indent=2)+'\n')


def run(command, name, cap=180):
    assert time.monotonic()<deadline
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    started = time.perf_counter()
    result = subprocess.run(command,capture_output=True,text=True,env=env,
                            timeout=min(cap,max(1,deadline-time.monotonic())))
    (a.root/name).write_text(result.stdout+'\n'+result.stderr)
    assert result.returncode==0, result.stderr
    return result.stdout, started


save('inputs.json',inputs)
runs,summary,quality,contracts = [],{},{},[]
try:
    for n in (716,1,16,63,64):
        for i,arm in enumerate(('base','candidate','candidate','base','base','candidate')):
            stdout,started = run([sys.executable,str(source/'code/evaluation/benchmark_real_time_release.py'),
                                  '--runtime',str(a.root/arm),'--input',input_path,'--n',str(n)],f'time_{n}_{i}.log',120)
            row = json.loads(stdout)
            row.update(arm=arm,process_to_ready_seconds=row.pop('ready_clock')-started)
            runs.append(row); save('timing.json',runs)
        summary[str(n)] = {}
        for metric in ('predict_seconds','process_to_ready_seconds'):
            med = {arm:statistics.median(r[metric] for r in runs if r['samples']==n and r['arm']==arm)
                   for arm in ('base','candidate')}
            summary[str(n)][metric] = {'median':med,'gain':1-med['candidate']/med['base']}
        save('timing_summary.json',summary)
        if (any(v['gain']<-.05 for v in summary[str(n)].values())
                or (n==716 and summary['716']['predict_seconds']['gain']<.10)):
            save('RESULT.json',{'decision':'STOP_GRAPH_RUNTIME','summary':summary,'inputs':inputs})
            raise SystemExit
    for device in ('cuda','cpu'):
        stdout,_ = run([sys.executable,str(Path(__file__).resolve()),'--root',str(a.root),'--contract',device],device+'_contract.log')
        contracts.append(json.loads(stdout))
    for i,arm in enumerate(('base','candidate','candidate','base','base','candidate')):
        output = a.root/f'quality_{i}.json'
        run([sys.executable,str(source/'code/evaluation/evaluate_b210_transfer.py'),'--execute',
             '--worker-runtime',str(a.root/arm),'--worker-label',arm,'--worker-output',str(output),
             '--study','real-reflection','--device','cuda:0',
             '--clean-manifest','/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json',
             '--real-dir','/root/autodl-tmp/realpde/data/real'],f'quality_{i}.log')
        quality.setdefault(arm,[]).append(json.loads(output.read_text()))
    all_results = quality['base']+quality['candidate']
    for key in ('input_sha256','target_sha256','scorer_sha256'):
        assert len({r[key] for r in all_results})==1
    deltas = [{k:c['metrics'][k]-b['metrics'][k] for k in replay.METRICS}
              for b,c in zip(quality['base'],quality['candidate'])]
    max_difference = max(abs(v) for d in deltas for v in d.values())
    save('RESULT.json',{'decision':'PASS_GRAPH_RUNTIME' if max_difference<=1e-5 else 'STOP_GRAPH_QUALITY_DRIFT',
                       'inputs':inputs,'summary':summary,'contracts':contracts,'quality_deltas':deltas,
                       'max_quality_score_difference':max_difference,'official_gain':None,
                       'scope':'A800 PCIe/Torch2.2.2 fresh processes; startup included separately; hidden evaluator IPC excluded'})
except Exception as error:
    save('failure.json',{'type':type(error).__name__,'message':str(error),'summary':summary,'inputs':inputs})
    raise
