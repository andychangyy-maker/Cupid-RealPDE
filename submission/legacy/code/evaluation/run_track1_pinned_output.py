"""Bounded paired runtime test; no scoring, extra warmup, or profiling in timed children."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import zipfile

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--child', choices=['base', 'candidate'])
p.add_argument('--n', type=int)
a = p.parse_args()
INPUT = '/root/autodl-tmp/spuq2_a800/artifacts/clean716_input_cache.npy'
BASE = '/root/autodl-tmp/real_time_release_20260908/combined.zip'

if a.child:
    entered = time.perf_counter()
    import numpy as np
    import torch
    assert torch.__version__ == '2.2.2+cu121'
    torch.set_num_threads(4)
    x = np.load(INPUT, mmap_mode='r')[:a.n].copy()
    sys.path.insert(0, str(a.root / a.child))
    import submission
    torch.cuda.synchronize()
    imported = time.perf_counter()
    values = submission.predict(x)
    torch.cuda.synchronize()
    ready = time.perf_counter()
    print(json.dumps({'ready_clock': ready, 'predict_seconds': ready-imported,
                      'process_body_seconds': ready-entered, 'samples': a.n}), flush=True)
    raise SystemExit

a.root.mkdir(exist_ok=True)
for name, archive in [('base', BASE), ('candidate', str(a.root / 'candidate.zip'))]:
    dest = a.root / name
    dest.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        z.extractall(dest)
with zipfile.ZipFile(BASE) as b, zipfile.ZipFile(a.root / 'candidate.zip') as c:
    assert b.namelist() == c.namelist()
    assert [n for n in b.namelist() if b.read(n) != c.read(n)] == ['submission.py']

# Compare the actual candidate predict function on the identical loaded state.
# Fresh processes below are reserved for timing, avoiding two resident models.
check_code = '''
import ast,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
r=Path(sys.argv[1]);torch.set_num_threads(4)
sys.path.insert(0,str(r/'base'));import submission as b
source=(r/'candidate/submission.py').read_text()
tree=ast.parse(source);fn=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='predict']
assert len(fn)==1
ns=dict(vars(b));exec(compile(ast.Module(body=fn,type_ignores=[]),'candidate/predict','exec'),ns)
x=np.load(sys.argv[2],mmap_mode='r');records=[]
for n in (1,16,63,64,716):
    v=x[:n].copy();old=b.predict(v);new=ns['predict'](v)
    same={k:np.array_equal(old[k],new[k]) for k in old}
    record={'n':n,'bitwise':same,'max_abs':{k:float(np.max(np.abs(old[k]-new[k]))) for k in old}}
    records.append(record);(r/'quality.json').write_text(json.dumps(records,indent=2))
    if not all(same.values()):
        repeat=b.predict(v)
        record['base_repeat_bitwise']={k:np.array_equal(old[k],repeat[k]) for k in old}
        (r/'quality.json').write_text(json.dumps(records,indent=2));raise SystemExit(2)
    assert all(np.isfinite(new[k]).all() for k in new)
    assert np.all(new['lower']<=new['upper'])
    if n==16:
        saved={k:hashlib.sha256(z.tobytes()).hexdigest() for k,z in new.items()}
        ns['predict'](x[100:101].copy())
        assert all(hashlib.sha256(new[k].tobytes()).hexdigest()==h for k,h in saved.items())
    del old,new
print('BITWISE_PASS')
'''
cp = subprocess.run([sys.executable, '-c', check_code, str(a.root), INPUT],
                    capture_output=True, text=True, timeout=600)
(a.root/'quality.log').write_text(cp.stdout+'\n'+cp.stderr)
if cp.returncode:
    (a.root/'decision.json').write_text(json.dumps({'decision':'STOP_OUTPUT_GATE','returncode':cp.returncode}))
    raise SystemExit(cp.returncode)
runs=[]
deadline=time.monotonic()+1800
for n in (16,716,1,63,64):
    for arm in ('base','candidate','candidate','base','base','candidate'):
        assert time.monotonic()<deadline
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        start=time.perf_counter()
        cp=subprocess.run([sys.executable,__file__,'--root',str(a.root),'--child',arm,'--n',str(n)],
                          capture_output=True,text=True,timeout=120,
                          env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
        assert cp.returncode==0,cp.stderr
        row=json.loads(cp.stdout);row.update(arm=arm,process_to_ready_seconds=row.pop('ready_clock')-start)
        runs.append(row);(a.root/'timing_progress.json').write_text(json.dumps(runs,indent=2))
        print(json.dumps(row),flush=True)
summary={}
for n in (1,16,63,64,716):
    group=[r for r in runs if r['samples']==n]
    summary[str(n)]={}
    for metric in ('predict_seconds','process_to_ready_seconds'):
        vals={arm:[r[metric] for r in group if r['arm']==arm] for arm in ('base','candidate')}
        meds={arm:statistics.median(v) for arm,v in vals.items()}
        summary[str(n)][metric]={'median':meds,'gain':1-meds['candidate']/meds['base'],'values':vals}
regressions=[(n,m) for n,s in summary.items() for m,v in s.items() if v['gain']<-.05]
gain=summary['716']['predict_seconds']['gain']
decision='PASS_RUNTIME_CANDIDATE' if gain>=.10 and not regressions else 'STOP_NO_DECISIVE_NET_GAIN'
(a.root/'decision.json').write_text(json.dumps({'decision':decision,'summary':summary,'regressions':regressions,
    'scope':'same-state bitwise check; fresh-process pre-extracted ZIP latency; no hidden evaluator or output IPC'},indent=2))
print(decision,flush=True)
