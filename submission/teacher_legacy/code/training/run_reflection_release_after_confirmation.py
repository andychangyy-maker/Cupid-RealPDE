"""Consume a passing five-fold seal; fixed final fit and ZIP diagnostics only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/root/track1_reflection_confirm_20260909')
source = Path(__file__).resolve().parents[2]
promotion = root/'five_fold_decision.json'
deadline = time.monotonic()+4500
while not promotion.is_file():
    status = root/'confirmation_status.json'
    if status.is_file() and json.loads(status.read_text()).get('status') == 'TERMINAL':
        print('Confirmation stopped; final training not started', flush=True)
        sys.exit(0)
    assert time.monotonic()<deadline, 'confirmation wait expired'
    time.sleep(10)
result = json.loads(promotion.read_text())
if result['decision'] != 'PASS_REFLECTION_FIVE_FOLD_FINAL_TRAINING_PENDING':
    print('Five-fold gate failed; final training not started', flush=True)
    sys.exit(0)
out = root/'final';out.mkdir(exist_ok=False)
cfg = {'output':str(out/'train'), 'vertical_reflection':True,
       'promotion_result':str(promotion), 'promotion_sha256':hashlib.sha256(promotion.read_bytes()).hexdigest(),
       'source_commit':'923761c4ff0d41ce96386c49107b685e1bf63a01', 'release_tools_commit':'01b920e',
       'start-sim':str(root/'final_assets/sim.pth'),
       'start-sim-sha256':'628860c7d82239d942895f08a6fd72ede81feb6a7cad4de4e32e110ed2742b9b',
       'start-metadata':str(root/'final_assets/sim_binding.json'),
       'start-metadata-sha256':'282307574088299de967a7d64597a417150dd81cb18ea7b8d2897ee29b00e96f',
       'clean-manifest':'/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json',
       'real-dir':'/root/autodl-tmp/realpde/data/real',
       'baseline_zip':'/root/autodl-tmp/track1_direct_pinned_20260909/candidate.zip',
       'baseline_sha256':'83d60c9aff83b07373b31d9b036e0ca42001402d8c6d12c19a10b8166b62e621'}
config = out/'config.json';config.write_text(json.dumps(cfg,indent=2))
env = {**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
for script, cap in [('code/training/run_real_time_release.py',2450),('code/evaluation/package_real_time_release.py',1000)]:
    with (out/(Path(script).stem+'.log')).open('x') as log:
        subprocess.run(['timeout','--signal=TERM','--kill-after=15s',str(cap)+'s',sys.executable,
            str(source/script),'--config',str(config)],check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
print('Final package diagnostics complete; full release checks pending',flush=True)
