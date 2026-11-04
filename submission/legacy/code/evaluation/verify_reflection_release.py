"""Check the actual combined ZIP contract and retain the validated Time gain."""
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

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--contract',choices=('cuda','cpu'))
a=p.parse_args();root=a.root;out=root/'final/release_checks'
source=Path(__file__).resolve().parents[2]
input_path='/root/autodl-tmp/spuq2_a800/artifacts/clean716_input_cache.npy'
package=json.loads((root/'final/package/RESULT.json').read_text())
assert package['decision']=='PASS_QUALITY_ZIP_DEPLOYMENT_DIAGNOSTICS'
if a.contract:
 import numpy as np
 import torch
 torch.set_num_threads(4)
 if a.contract=='cpu':torch.cuda.is_available=lambda:False
 sys.path.insert(0,str(out/'combined'));import submission
 x=np.load(input_path,mmap_mode='r');guard=None;checks=[]
 for n in ((1,) if a.contract=='cpu' else (16,1,63,64,716)):
  y=submission.predict(x[:n].copy())
  assert set(y)=={'prediction','lower','upper'}
  assert all(v.shape==(n,20,32,64,3) and v.dtype==np.float32 and np.isfinite(v).all() and np.all(v[...,2]==0) for v in y.values())
  assert np.all(y['lower']<=y['upper'])
  if guard is not None:assert all(hashlib.sha256(v.tobytes()).hexdigest()==guard_hash[k] for k,v in guard.items())
  if n==16:guard=y;guard_hash={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in y.items()}
  checks.append(n)
 print(json.dumps({'device':a.contract,'checked_sizes':checks,'public_contract_pass':True,
                   'prior_output_unchanged':a.contract=='cuda'}));sys.exit(0)
out.mkdir(exist_ok=False);deadline=time.monotonic()+1000
assets={'official':('/root/autodl-tmp/real_time_release_20260908/combined.zip','b8938d06c100cc68bb4b7e310de112754a9053b4d2834134cb2b4e6f4b5c2a35'),
        'time_only':('/root/autodl-tmp/track1_direct_pinned_20260909/candidate.zip','83d60c9aff83b07373b31d9b036e0ca42001402d8c6d12c19a10b8166b62e621'),
        'combined':(package['zip_path'],package['zip_sha256'])}
members={}
for arm,(path,digest) in assets.items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
 runtime=out/arm;runtime.mkdir()
 with zipfile.ZipFile(path) as z:
  assert z.testzip() is None
  assert set(z.namelist())=={'submission.py','models/__init__.py','models/unet2d.py','models/ensemble_unet2d.py','ensemble_unet2d.pth'}
  members[arm]={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()};z.extractall(runtime)
assert [k for k,v in members['combined'].items() if v!=members['time_only'][k]]==['ensemble_unet2d.pth']
env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
contracts=[]
for device in ('cuda','cpu'):
 cp=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--contract',device],capture_output=True,text=True,env=env,timeout=180)
 (out/(device+'_contract.log')).write_text(cp.stdout+'\n'+cp.stderr);assert cp.returncode==0,cp.stderr
 contracts.append(json.loads(cp.stdout))
runs=[]
for n in (1,16,63,64,716):
 orders=[('time_only','combined'),('combined','time_only'),('time_only','combined')]
 if n==716:orders=[('official','time_only','combined'),('combined','time_only','official'),('official','time_only','combined')]
 for order in orders:
  for arm in order:
   assert time.monotonic()<deadline
   assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
   start=time.perf_counter()
   cp=subprocess.run([sys.executable,str(source/'code/evaluation/benchmark_real_time_release.py'),'--runtime',str(out/arm),'--input',input_path,'--n',str(n)],capture_output=True,text=True,env=env,timeout=120)
   assert cp.returncode==0,cp.stderr
   row=json.loads(cp.stdout);row.update(arm=arm,process_to_ready_seconds=row.pop('ready_clock')-start);runs.append(row)
   (out/'timing.json').write_text(json.dumps(runs,indent=2))
summary={};regressions=[]
for n in (1,16,63,64,716):
 rows=[r for r in runs if r['samples']==n];entry={}
 for metric in ('predict_seconds','process_to_ready_seconds'):
  med={arm:statistics.median(r[metric] for r in rows if r['arm']==arm) for arm in {r['arm'] for r in rows}}
  if med['combined']/med['time_only']>1.05:regressions.append([n,metric])
  entry[metric]={'median':med,'gain_vs_time_only':1-med['combined']/med['time_only']}
  if n==716:entry[metric]['gain_vs_official']=1-med['combined']/med['official']
 summary[str(n)]=entry
passed=not regressions and all(summary['716'][m]['gain_vs_official']>0 for m in ('predict_seconds','process_to_ready_seconds'))
result={'decision':'PASS_REFLECTION_COMBINED_RELEASE' if passed else 'STOP_REFLECTION_RELEASE_TIME',
        'zip_sha256':package['zip_sha256'],'zip_path':package['zip_path'],'members':members,'contracts':contracts,
        'timing':summary,'regressions':regressions,'official_gain':None,
        'scope':'A800 PCIe Torch2.2.2; pre-extracted fresh process, no hidden evaluator IPC; quality evidence in five-fold seal'}
(out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
