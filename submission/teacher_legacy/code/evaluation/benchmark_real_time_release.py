"""Fresh-process sensitivity benchmark; input/IPC boundaries are not claimed official."""
import argparse,json,os,statistics,subprocess,sys,time,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path);p.add_argument('--out',type=Path);p.add_argument('--runtime');p.add_argument('--input');p.add_argument('--n',type=int);a=p.parse_args()
if a.runtime:
 entry=time.perf_counter();import numpy as np;import torch
 assert torch.__version__=='2.2.2+cu121'
 torch.set_num_threads(4);torch.cuda.get_device_name();libs=time.perf_counter()
 x=np.load(a.input,mmap_mode='r')[:a.n].copy();loaded=time.perf_counter()
 sys.path.insert(0,a.runtime);import submission
 torch.cuda.synchronize();imported=time.perf_counter()
 values=submission.predict(x);torch.cuda.synchronize();ready=time.perf_counter()
 # No output hashing, scoring, or large-output serialization in this timing process.
 print(json.dumps({'ready_clock':ready,'predict_seconds':ready-imported,'module_import_seconds':imported-loaded,
  'libraries_seconds':libs-entry,'input_load_seconds':loaded-libs,'samples':a.n}),flush=True)
else:
 import hashlib
 cfg=json.loads(a.manifest.read_text());a.out.mkdir(exist_ok=False);deadline=time.monotonic()+1800
 for arm,z in cfg['zips'].items():
  assert hashlib.sha256(Path(z['path']).read_bytes()).hexdigest()==z['sha256']
  runtime=a.out/arm;runtime.mkdir()
  with zipfile.ZipFile(z['path']) as f:f.extractall(runtime)
 schedule=[(716,arm) for order in [('official','quality','combined','halfcopy'),('halfcopy','combined','quality','official'),('official','quality','combined','halfcopy')] for arm in order]
 schedule += [(n,arm) for n in (1,16,63,64) for arm in ('combined','halfcopy','halfcopy','combined','combined','halfcopy')]
 runs=[]
 for i,(n,arm) in enumerate(schedule):
  assert time.monotonic()<deadline
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  start=time.perf_counter()
  cp=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--runtime',str(a.out/arm),'--input',cfg['input'],'--n',str(n)],
    text=True,capture_output=True,timeout=120,env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
  end=time.perf_counter();assert cp.returncode==0,cp.stderr
  result=json.loads(cp.stdout);result.update(arm=arm,index=i,process_to_ready_seconds=result.pop('ready_clock')-start,process_exit_seconds=end-start)
  runs.append(result);(a.out/'progress.json').write_text(json.dumps(runs,indent=2));print(json.dumps(result),flush=True)
 summaries={}
 for n in (1,16,63,64,716):
  group=[r for r in runs if r['samples']==n];arms=sorted({r['arm'] for r in group});entry={}
  for metric in ('predict_seconds','process_to_ready_seconds','process_exit_seconds'):
   med={arm:statistics.median(r[metric] for r in group if r['arm']==arm) for arm in arms}
   span={arm:(max(r[metric] for r in group if r['arm']==arm)-min(r[metric] for r in group if r['arm']==arm))/med[arm] for arm in arms}
   entry[metric]={'median':med,'relative_span':span,'halfcopy_gain':1-med['halfcopy']/med['combined']}
  summaries[str(n)]=entry
 clear_regressions=[]
 for n,s in summaries.items():
  for metric in ('predict_seconds','process_to_ready_seconds','process_exit_seconds'):
   d=s[metric]
   if d['halfcopy_gain'] < -max(.05,d['relative_span']['combined']):clear_regressions.append([n,metric])
 passed=summaries['716']['predict_seconds']['halfcopy_gain']>=.05 and not clear_regressions
 result={'timing_gate_pass':passed,'clear_regressions':clear_regressions,'summaries':summaries,'runs':runs,
  'scope':'pre-extracted ZIP, fresh Python process, input materialization and module prewarm included; no output IPC or hidden evaluator',
  'official_score':None,'quality_gate_separate':True}
 (a.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
