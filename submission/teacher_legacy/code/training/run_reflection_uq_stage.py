"""One paired inherited-UQ stage on already sealed reflected/control points."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--held-aoa', type=int, choices=(0,5,10,15,20), required=True)
p.add_argument('--source-commit', required=True)
a = p.parse_args()
source = Path(__file__).resolve().parents[2]
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
out = a.root/f'uq_stage{a.held_aoa}'
out.mkdir(exist_ok=False)
deadline = time.monotonic()+3600
env = {**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
def command(script, options, label, cap):
    with (out/(label+'.log')).open('w') as f:
        child = subprocess.Popen([sys.executable,str(source/script),*map(str,options)],stdout=f,stderr=subprocess.STDOUT,env=env)
        (out/'status.json').write_text(json.dumps({'status':'RUNNING','stage':label,'pid':child.pid,'held':a.held_aoa}))
        try:code = child.wait(timeout=min(cap,max(1,deadline-time.monotonic())))
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
        assert code==0, f'{label} failed, see {out/(label+".log")}'
manifest = '/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json'
real = '/root/autodl-tmp/realpde/data/real'
receipts = {}
try:
    for arm in ('control','reflection'):
        receipt = a.root/f'{arm}{a.held_aoa}'/'point_endpoint.json'
        point = json.loads(receipt.read_text())
        assert sha(Path(point['checkpoint_path'])) == point['checkpoint_sha256']
        command('code/training/train_b210_transfer_uq.py',[
            '--execute','--label',arm,'--point-label','C','--initialization','point','--held-aoa',a.held_aoa,
            '--full-real-point','--reflection-comparison','--source-commit',a.source_commit,
            '--point-receipt',receipt,'--point-receipt-sha256',sha(receipt),
            '--clean-manifest',manifest,'--real-dir',real,'--output-dir',out/arm,'--wall-budget-seconds',600],arm,650)
        receipts[arm] = out/arm/'uq_endpoint.json'
    scratch = out/'scratch';scratch.mkdir()
    command('code/evaluation/evaluate_b210_transfer.py',[
        '--execute','--study','real-reflection','--compare','reflection:control','--expected-labels','control','reflection',
        '--held-aoa',a.held_aoa,'--baseline-zip','/root/autodl-tmp/spuq2_a800/artifacts/track1_v3.zip',
        '--uq-receipt',f'control={receipts["control"]}','--uq-receipt',f'reflection={receipts["reflection"]}',
        '--clean-manifest',manifest,'--real-dir',real,'--scratch-dir',scratch,'--output',out/'evaluation.json',
        '--wall-budget-seconds',600], 'evaluation',650)
    evaluation = json.loads((out/'evaluation.json').read_text())
    delta = evaluation['contrasts']['reflection-control']
    if a.held_aoa in (20, 5):
        passed = delta['fixed_time_fitted_final'] > 0 and delta['sps_score'] >= -.10
        decision = 'PASS_REFLECTION_FOLD_NOT_CONFIRMATION' if passed else 'STOP_REFLECTION_UQ_FOLD'
    else:
        passed = delta['fixed_time_fitted_final'] >= -.03
        decision = 'CONTINUE_REFLECTION_CONFIRMATION' if passed else 'STOP_REFLECTION_WORST_FOLD'
    result = {'decision':decision,
              'held':a.held_aoa,'delta':delta,'fixed_time_score':92.032660,'core_source_commit':a.source_commit,
              'official_gain':None,'release_verified':False}
    (out/'decision.json').write_text(json.dumps(result,indent=2))
    (out/'status.json').write_text(json.dumps({'status':'TERMINAL',**result},indent=2))
    print(json.dumps(result),flush=True)
except Exception as error:
    (out/'failure.json').write_text(json.dumps({'error':repr(error),'decision':'EXECUTION_FAILURE_NOT_NEGATIVE_SCORE'}))
    raise
