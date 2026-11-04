"""One overlapping full-quality precheck; stage confirmation is still required."""
import json
from pathlib import Path
import subprocess
import sys

import torch
import evaluate_b210_transfer as replay

SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'code/submissions'))
from build_track1_blended_ensemble_unet2d_package import compact_state_dict

root=Path('/root/track1_reflection_exposure_20260909')
before=json.loads((root/'RESULT.json').read_text())
assert before['decision']=='PASS_EXPOSURE_POINT_NEEDS_UQ_AND_STAGE_VALIDATION'
receipt=json.loads((root/'uq/uq_endpoint.json').read_text())
assert receipt['point_receipt_sha256']==replay.sha256(root/'point/point_endpoint.json')
assert receipt['point_receipt']['updates']==5334 and receipt['point_state_bitwise']
assert receipt['updates']==3200 and receipt['seed']==18 and receipt['tail_arrays_read']==0
assert receipt['initialization']=='point_checkpoint' and receipt['training']['point_updates']==5334
assert replay.sha256(receipt['checkpoint_path'])==receipt['checkpoint_sha256']
torch.set_num_threads(4)
payload=torch.load(receipt['checkpoint_path'],map_location='cpu',weights_only=False)
template=torch.load(root/'baseline/ensemble_unet2d.pth',map_location='cpu',weights_only=False)
assert replay.canonical_sha256(payload['normalizer'])==replay.canonical_sha256(template['normalizer'])
runtime=root/'uq_runtime';runtime.mkdir(exist_ok=False)
for name in replay.RUNTIME_FILES:
    target=runtime/name;target.parent.mkdir(exist_ok=True,parents=True)
    target.write_bytes((root/'baseline'/name).read_bytes())
compact,_=compact_state_dict(payload['model_state_dict'],1024)
torch.save({'model_config':template['model_config'],'quantized_state_dict':compact,'normalizer':payload['normalizer'],
            'phase':'real','inference_dtype':'float16','storage_format':'symmetric_per_dim0_int8_v1'},runtime/'ensemble_unet2d.pth')
del payload,template,compact
output=root/'uq_evaluation.json'
manifest=Path('/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json')
assert replay.sha256(manifest)==receipt['manifest_sha256']
with (root/'uq_evaluation.log').open('x') as log:
    subprocess.run([sys.executable,str(SOURCE/'code/evaluation/evaluate_b210_transfer.py'),'--execute',
        '--worker-runtime',str(runtime),'--worker-label','exposure_uq','--worker-output',str(output),
        '--device','cuda:0','--study','real-reflection','--clean-manifest',str(manifest),
        '--real-dir','/root/autodl-tmp/realpde/data/real'],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=180)
b=before['results']['baseline'];c=json.loads(output.read_text())
for key in ('input_sha256','target_sha256','scorer_sha256'):assert b[key]==c[key]
drift={k:abs(c['metrics'][k]-before['results']['candidate']['metrics'][k]) for k in replay.METRICS[:3]}
assert max(drift.values())<=1e-4,drift
delta={k:c['metrics'][k]-b['metrics'][k] for k in (*replay.METRICS,'fixed_time_fitted_final')}
angles={k:c['by_aoa'][k]['fixed_time_fitted_final']-v['fixed_time_fitted_final'] for k,v in b['by_aoa'].items()}
passed=delta['fixed_time_fitted_final']>=0 and sum(v>=0 for v in angles.values())>=4
result={'decision':'PASS_EXPOSURE_UQ_PREFLIGHT_NEEDS_STAGE_CONFIRMATION' if passed else 'STOP_EXPOSURE_UQ_PREFLIGHT',
        'delta':delta,'by_aoa':angles,'point_score_drift_after_uq':drift,'uq_receipt_sha256':replay.sha256(root/'uq/uq_endpoint.json'),
        'runtime_checkpoint_sha256':replay.sha256(runtime/'ensemble_unet2d.pth'),
        'independent_validation':False,'release_verified':False,'official_gain':None,'evaluation':c}
(root/'uq_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='evaluation'},indent=2))
