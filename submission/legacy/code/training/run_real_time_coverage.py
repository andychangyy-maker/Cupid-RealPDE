"""Real time-coverage paired screen or one prespecified confirmation fold."""
import argparse,json,os,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/evaluation'))
import evaluate_b210_transfer as replay

def save(path,value):
 path.write_text(json.dumps(value,indent=2)+'\n')

def run(a):
 assert re.fullmatch('[0-9a-f]{40}',a.source_commit)
 a.output_dir.mkdir(parents=True,exist_ok=False)
 folds=(a.confirmation_fold,) if a.confirmation_fold is not None else (0,15)
 wall_cap=4200 if a.confirmation_fold is not None else 8400
 start=time.monotonic();deadline=start+wall_cap;results={}
 env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
 save(a.output_dir/'identity.json',{'source_commit':a.source_commit,'fold_order':list(folds),
  'same_gpu_prefix_control_retrained':True,'wall_cap_seconds':wall_cap,
  'scope':'Real/UQ-stage LOAO only; shared Sim has historical Real ancestry'})
 def command(script,options,log,cap):
  remaining=deadline-time.monotonic()
  if remaining<=0:raise TimeoutError('Real coverage budget expired')
  with log.open('w') as f:
   proc=subprocess.Popen([sys.executable,str(ROOT/script),*map(str,options)],stdout=f,stderr=subprocess.STDOUT,env=env)
   save(a.output_dir/'status.json',{'status':'RUNNING','pid':proc.pid,'log':str(log),'source_commit':a.source_commit})
   try:code=proc.wait(timeout=min(cap,remaining))
   except subprocess.TimeoutExpired:
    proc.terminate()
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
    raise
   if code:raise RuntimeError(f'{script}: exit {code}, see {log}')
 for aoa in folds:
  f=a.output_dir/f'held{aoa}';f.mkdir()
  common=['--execute','--held-aoa',aoa,'--clean-manifest',a.clean_manifest,'--real-dir',a.real_dir,'--device','cuda:0']
  for label in ('inherit','full'):
   point_dir=f/('point_'+label)
   command('code/training/train_b210_transfer_real4k.py',[*common,'--arm','C',
    *(['--full-real-trajectories'] if label=='full' else []),
    '--source-commit',a.source_commit,'--start-sim',a.start_sim,'--start-sim-sha256',replay.SIM_INHERITED_POINTS['inherit']['start_sim_sha256'],
    '--start-metadata',a.start_metadata,'--start-metadata-sha256','282307574088299de967a7d64597a417150dd81cb18ea7b8d2897ee29b00e96f',
    '--output-dir',point_dir,'--wall-budget-seconds',1500],f/(label+'_point.log'),1550)
   receipt=point_dir/'point_endpoint.json'
   command('code/training/train_b210_transfer_uq.py',[*common,'--label',label,'--point-label','C',
    '--initialization','point',*(['--full-real-point'] if label=='full' else []),
    '--point-receipt',receipt,'--point-receipt-sha256',replay.sha256(receipt),
    '--output-dir',f/label,'--wall-budget-seconds',600],f/(label+'_uq.log'),650)
  scratch=f/'scratch';scratch.mkdir()
  control=f/'inherit/uq_endpoint.json'
  command('code/evaluation/evaluate_b210_transfer.py',[*common,'--baseline-zip',a.baseline_zip,
   '--expected-labels','inherit','full','--study','real-time-coverage','--compare','full:inherit',
   '--uq-receipt',f'inherit={control}','--uq-receipt',f'full={f / "full/uq_endpoint.json"}',
   '--output',f/'evaluation.json','--scratch-dir',scratch,'--wall-budget-seconds',600],f/'evaluation.log',650)
  results[str(aoa)]=json.loads((f/'evaluation.json').read_text())
  delta=results[str(aoa)]['contrasts']['full-inherit']['fixed_time_fitted_final']
  print(json.dumps({'held_aoa':aoa,'fixed_time_fitted_final_delta':delta}),flush=True)
  if aoa==15:
   values=[r['contrasts']['full-inherit']['fixed_time_fitted_final'] for r in results.values()]
   gate=all(v>0 for v in values) and sum(values)/2>=.05
   save(a.output_dir/'screen.json',{'passed':gate,'deltas':values,'mean':sum(values)/2,'gate':'both >0 and mean >=0.05'})
   if not gate:break
 official=replay.import_file(replay.SCORER,'coverage_scoring')
 pooled={label:replay.finalize(official,replay.merge_states([r['arms'][label]['pooled_state'] for r in results.values()])) for label in ('inherit','full')}
 delta=pooled['full']['fixed_time_fitted_final']-pooled['inherit']['fixed_time_fitted_final']
 ds={aoa:r['contrasts']['full-inherit']['fixed_time_fitted_final'] for aoa,r in results.items()}
 passed=len(results)==2 and all(v>0 for v in ds.values()) and sum(ds.values())/2>=.05
 outcome=('PAIRED_FOLD_COMPLETE_PENDING_FIVE_FOLD_CONFIRMATION' if all(v>0 for v in ds.values())
          else 'STOP_REAL_TIME_COVERAGE') if a.confirmation_fold is not None else (
          'PASS_TWO_FOLD_SCREEN_NOT_CONFIRMATION' if passed else 'STOP_REAL_TIME_COVERAGE')
 decision={'decision':outcome,
  'folds':list(results),'pooled_metrics':pooled,'pooled_fitted_final_delta':delta,'fold_deltas':ds,
  'scope':'Real/UQ-stage LOAO only; shared Sim has historical Real ancestry',
  'official_gain':None,'release_verified':False,'elapsed_seconds':time.monotonic()-start}
 save(a.output_dir/'decision.json',decision);save(a.output_dir/'status.json',{'status':'TERMINAL',**decision})
 return decision

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('output-dir','start-sim','start-metadata','clean-manifest','real-dir','baseline-zip'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--source-commit',required=True)
 p.add_argument('--confirmation-fold',type=int,choices=(5,10,20))
 a=p.parse_args()
 try:print(json.dumps(run(a),indent=2))
 except Exception as e:
  if a.output_dir.is_dir():save(a.output_dir/'failure.json',{'error':repr(e),'decision':'EXECUTION_FAILURE_NOT_NEGATIVE_SCORE'})
  raise
