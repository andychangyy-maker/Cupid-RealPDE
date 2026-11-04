"""Fixed final Real full-coverage fit followed by prefix-only inherited UQ."""
import argparse,json,os,subprocess,sys,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/training'))
from train_b210_transfer_real4k import sha256
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=Path,required=True)
a=p.parse_args();cfg=json.loads(a.config.read_text());out=Path(cfg['output'])
if cfg.get('vertical_reflection'):
 assert sha256(Path(cfg['promotion_result']))==cfg['promotion_sha256']
 assert json.loads(Path(cfg['promotion_result']).read_text())['decision']=='PASS_REFLECTION_FIVE_FOLD_FINAL_TRAINING_PENDING'
out.mkdir(parents=True,exist_ok=False)
deadline=time.monotonic()+2400

def save(name,value):
 (out/name).write_text(json.dumps(value,indent=2)+'\n')

def command(script,options,label,cap):
 with (out/(label+'.log')).open('w') as log:
  proc=subprocess.Popen([sys.executable,str(ROOT/script),*map(str,options)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
    env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'})
  save('status.json',{'status':'RUNNING','stage':label,'pid':proc.pid,'source_commit':cfg['source_commit']})
  try:code=proc.wait(timeout=min(cap,deadline-time.monotonic()))
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise
  if code:raise RuntimeError(f'{label} exited {code}')

save('identity.json',cfg)
try:
 common=['--execute','--full-real-final','--clean-manifest',cfg['clean-manifest'],'--real-dir',cfg['real-dir'],'--device','cuda:0']
 command('code/training/train_b210_transfer_real4k.py',[*common,'--arm','C','--full-real-trajectories',
   *(['--vertical-reflection'] if cfg.get('vertical_reflection') else []),
   '--source-commit',cfg['source_commit'],'--start-sim',cfg['start-sim'],'--start-sim-sha256',cfg['start-sim-sha256'],
   '--start-metadata',cfg['start-metadata'],'--start-metadata-sha256',cfg['start-metadata-sha256'],
   '--output-dir',out/'point','--wall-budget-seconds',1500],'point',1550)
 receipt=out/'point/point_endpoint.json'
 if cfg.get('vertical_reflection'):
  assert json.loads(receipt.read_text())['vertical_reflection']=={'probability':.25,'isolated_cpu_seed':29009,'space':'physical_uvp_before_normalization'}
 command('code/training/train_b210_transfer_uq.py',[*common,'--label','full','--point-label','C',
   '--initialization','point','--full-real-point','--point-receipt',receipt,'--point-receipt-sha256',sha256(receipt),
   '--output-dir',out/'uq','--wall-budget-seconds',600],'uq',650)
 save('status.json',{'status':'TERMINAL','decision':'FIXED_FINAL_TRAINING_COMPLETE_ZIP_PENDING'})
except Exception as e:
 save('status.json',{'status':'TERMINAL','decision':'EXECUTION_FAILURE_NOT_NEGATIVE_SCORE','error':repr(e)})
 raise
