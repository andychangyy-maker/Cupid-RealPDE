"""Fixed 5334-update reflection fit: overlapping point precheck, never a release."""
import dataclasses
import functools
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE/'code/training'),str(SOURCE/'code/submissions'),str(SOURCE/'code')]
import train_b210_transfer_real4k as point
import evaluate_b210_transfer as replay

ROOT = Path('/root/track1_reflection_exposure_20260909')
ASSETS = Path('/root/track1_reflection_confirm_20260909/final_assets')
BASE = Path('/root/autodl-tmp/track1_direct_pinned_20260909/candidate.zip')
commit = sys.argv[1]
assert len(commit)==40 and not ROOT.exists()
ROOT.mkdir()
deadline = time.monotonic()+2100
point.UPDATES = 5334
dense = point.load_runtime()
dense.small4k.UPDATES = 5334
dense.small4k.PRIMARY_ENDPOINT = 'ema_u5334'
dense.small4k.train_candidate = functools.partial(dense.small4k.train_candidate,endpoint_filename='ema999_u5334_primary.pth')
original_fields = point.intervention_fields
point.intervention_fields = lambda args:{**original_fields(args),
    'duration_intervention':'ceil(4000/0.75)=5334; approximately match original-view exposures'}
args = point.parse_args([
    '--execute','--arm','C','--full-real-trajectories','--full-real-final','--vertical-reflection',
    '--source-commit',commit,'--start-sim',str(ASSETS/'sim.pth'),
    '--start-sim-sha256','628860c7d82239d942895f08a6fd72ede81feb6a7cad4de4e32e110ed2742b9b',
    '--start-metadata',str(ASSETS/'sim_binding.json'),
    '--start-metadata-sha256','282307574088299de967a7d64597a417150dd81cb18ea7b8d2897ee29b00e96f',
    '--clean-manifest','/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json',
    '--real-dir','/root/autodl-tmp/realpde/data/real','--output-dir',str(ROOT/'point'),
    '--device','cuda:0','--wall-budget-seconds','1500'])
point.run(args)
receipt=json.loads((ROOT/'point/point_endpoint.json').read_text())
assert receipt['updates']==5334 and receipt['vertical_reflection']['probability']==.25
assert replay.sha256(receipt['checkpoint_path'])==receipt['checkpoint_sha256']
assert replay.sha256(BASE)=='83d60c9aff83b07373b31d9b036e0ca42001402d8c6d12c19a10b8166b62e621'

import torch
from build_track1_blended_ensemble_unet2d_package import compact_state_dict
from models.ensemble_unet2d import EnsembleUNet2dConfig
torch.set_num_threads(4)
payload=torch.load(receipt['checkpoint_path'],map_location='cpu',weights_only=False)
assert payload['iteration']==5334 and payload['endpoint']=='ema_u5334' and payload['ema']['updates']==5334
with zipfile.ZipFile(BASE) as z:
    template=torch.load(io.BytesIO(z.read('ensemble_unet2d.pth')),map_location='cpu',weights_only=False)
    configs=[]
    for p in (payload,template):
        cfg=dict(p['model_config']);assert cfg.pop('architecture')=='ensemble_unet'
        configs.append(dataclasses.asdict(EnsembleUNet2dConfig(**cfg)))
    assert configs[0]==configs[1]
    assert replay.canonical_sha256(payload['normalizer'])==replay.canonical_sha256(template['normalizer'])
    for arm in ('baseline','candidate'):
        runtime=ROOT/arm;runtime.mkdir()
        for name in (*replay.RUNTIME_FILES,'ensemble_unet2d.pth'):
            dst=runtime/name;dst.parent.mkdir(exist_ok=True,parents=True);dst.write_bytes(z.read(name))
compact,_=compact_state_dict(payload['model_state_dict'],1024)
torch.save({'model_config':template['model_config'],'quantized_state_dict':compact,'normalizer':payload['normalizer'],
            'phase':'real','inference_dtype':'float16','storage_format':'symmetric_per_dim0_int8_v1'},ROOT/'candidate/ensemble_unet2d.pth')
del payload,template,compact
torch.cuda.empty_cache()
results={}
for arm in ('baseline','candidate'):
    output=ROOT/(arm+'.json')
    with (ROOT/(arm+'.log')).open('x') as log:
        subprocess.run([sys.executable,str(SOURCE/'code/evaluation/evaluate_b210_transfer.py'),'--execute',
            '--worker-runtime',str(ROOT/arm),'--worker-label',arm,'--worker-output',str(output),
            '--device','cuda:0','--study','real-reflection','--clean-manifest',str(args.clean_manifest),
            '--real-dir',str(args.real_dir)],check=True,stdout=log,stderr=subprocess.STDOUT,
            timeout=max(1,deadline-time.monotonic()),env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
    results[arm]=json.loads(output.read_text())
b,c=results['baseline'],results['candidate']
for key in ('input_sha256','target_sha256','scorer_sha256'):assert b[key]==c[key]
replay.TIME_SCORE=92.032660
metrics={**c['metrics'],'sps_score':b['metrics']['sps_score']}
contribution=replay.fixed_time_final(metrics)-replay.fixed_time_final(b['metrics'])
delta={k:c['metrics'][k]-b['metrics'][k] for k in replay.METRICS[:3]}
passed=contribution>=.02 and all(delta[k]>=limit for k,limit in zip(replay.METRICS[:3],(-.02,-.05,-.05)))
training=json.loads((ROOT/'point/result.json').read_text())
result={'decision':'PASS_EXPOSURE_POINT_NEEDS_UQ_AND_STAGE_VALIDATION' if passed else 'STOP_EXPOSURE_POINT',
        'point_contribution_fixed_sps_time':contribution,'point_delta':delta,
        'draws':training['reflection_draws'],'reflected':training['reflected_windows'],
        'unreflected':training['reflection_draws']-training['reflected_windows'],
        'control_unreflected':31988,'source_commit':commit,'point_receipt':receipt,
        'baseline_zip_sha256':replay.sha256(BASE),'results':results,'uq_complete':False,
        'independent_validation':False,'release_verified':False,'official_gain':None}
(ROOT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('results','point_receipt')},indent=2))
