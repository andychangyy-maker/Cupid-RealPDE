"""Package the sealed Q4 endpoint only after its own five-fold gate passes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import evaluate_b210_transfer as replay

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--five-fold', type=Path, required=True)
p.add_argument('--source-commit', required=True)
a = p.parse_args()
source = Path(__file__).resolve().parents[2]
sealed = Path('/root/track1_reflection_exposure_20260909')
base = Path('/root/autodl-tmp/track1_direct_pinned_20260909/candidate.zip')
replay.TIME_SCORE = 92.032660
promotion = json.loads(a.five_fold.read_text())
assert promotion['decision'] == 'PASS_EXPOSURE_FIVE_FOLD_NEEDS_RELEASE'
assert promotion['samples'] == 716 and promotion['trajectories'] == 81
assert promotion['fixed_time_score'] == replay.TIME_SCORE
assert promotion['delta']['fixed_time_fitted_final'] >= .06
finals = [row['fixed_time_fitted_final'] for row in promotion['by_fold'].values()]
assert len(finals) == 5 and sum(v > 0 for v in finals) >= 4 and min(finals) >= -.03
for path, digest in promotion['inputs'].items():
    assert replay.sha256(path) == digest
assert len(a.source_commit) == 40
assert replay.sha256(base) == '83d60c9aff83b07373b31d9b036e0ca42001402d8c6d12c19a10b8166b62e621'
precheck = json.loads((sealed/'uq_preflight.json').read_text())
assert precheck['decision'] == 'PASS_EXPOSURE_UQ_PREFLIGHT_NEEDS_STAGE_CONFIRMATION'
receipt_path = sealed/'uq/uq_endpoint.json'
assert replay.sha256(receipt_path) == precheck['uq_receipt_sha256'] == '5f10aa04ca8475ae1435f47bd28b71e2e48dcc3a8c715ce87b4b860a9115f001'
receipt = json.loads(receipt_path.read_text())
assert receipt['point_receipt']['updates'] == 5334 and receipt['updates'] == 3200
assert receipt['point_state_bitwise'] and receipt['initialization'] == 'point_checkpoint'
assert replay.sha256(sealed/'point/point_endpoint.json') == receipt['point_receipt_sha256']
assert replay.sha256(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
weights = sealed/'uq_runtime/ensemble_unet2d.pth'
assert replay.sha256(weights) == precheck['runtime_checkpoint_sha256'] == '7c9a0c77f76f39a53ff51a6bb23aa3733fc9c6569ca94095eb899cbc9aa97fdb'

package = a.root/'final/package'
package.mkdir(parents=True, exist_ok=False)
candidate = package/'quality.zip'
with zipfile.ZipFile(base) as src, zipfile.ZipFile(candidate, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as dst:
    assert src.testzip() is None
    assert set(src.namelist()) == set((*replay.RUNTIME_FILES, 'ensemble_unet2d.pth'))
    for name in src.namelist():
        if name == 'ensemble_unet2d.pth':
            dst.write(weights, name)
        else:
            assert src.read(name) == (sealed/'uq_runtime'/name).read_bytes()
            dst.writestr(src.getinfo(name), src.read(name))
for arm, path in (('baseline', base), ('candidate', candidate)):
    runtime = package/arm
    runtime.mkdir()
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set((*replay.RUNTIME_FILES, 'ensemble_unet2d.pth'))
        z.extractall(runtime)

deadline = time.monotonic()+800
results = {}
for label in ('baseline1', 'candidate1', 'baseline2', 'candidate2'):
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
    output = package/(label+'.json')
    with (package/(label+'.log')).open('x') as log:
        subprocess.run([sys.executable, str(source/'code/evaluation/evaluate_b210_transfer.py'), '--execute',
                        '--worker-runtime', str(package/label[:-1]), '--worker-label', label,
                        '--worker-output', str(output), '--device', 'cuda:0', '--study', 'real-reflection',
                        '--clean-manifest', '/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json',
                        '--real-dir', '/root/autodl-tmp/realpde/data/real'], check=True,
                       stdout=log, stderr=subprocess.STDOUT, timeout=min(180,max(1,deadline-time.monotonic())),
                       env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
    results[label] = json.loads(output.read_text())
b, c = results['baseline1'], results['candidate1']
for key in ('input_sha256', 'target_sha256', 'scorer_sha256'):
    assert len({row[key] for row in results.values()}) == 1
keys = (*replay.METRICS, 'fixed_time_fitted_final')
delta = {k:c['metrics'][k]-b['metrics'][k] for k in keys}
angles = {k:c['by_aoa'][k]['fixed_time_fitted_final']-v['fixed_time_fitted_final'] for k,v in b['by_aoa'].items()}
jitter = {k:abs(results['baseline2']['metrics'][k]-b['metrics'][k]) for k in keys}
repeat = {k:abs(results['candidate2']['metrics'][k]-c['metrics'][k]) for k in keys}
stable = all(repeat[k] <= max(2*jitter[k],1e-4) for k in keys)
passed = stable and delta['fixed_time_fitted_final'] >= 0 and sum(v >= 0 for v in angles.values()) >= 4
result = {'decision':'PASS_QUALITY_ZIP_DEPLOYMENT_DIAGNOSTICS' if passed else 'STOP_EXPOSURE_FINAL_ZIP',
          'zip_path':str(candidate),'zip_sha256':replay.sha256(candidate),'zip_bytes':candidate.stat().st_size,
          'promotion_basis':promotion,'promotion_sha256':replay.sha256(a.five_fold),'source_commit':a.source_commit,
          'sealed_uq_receipt_sha256':replay.sha256(receipt_path),'runtime_checkpoint_sha256':replay.sha256(weights),
          'quality_delta_overlapping_diagnostic':delta,'by_aoa_overlapping_diagnostic':angles,
          'stable':stable,'baseline_repeat_jitter':jitter,'candidate_repeat_jitter':repeat,
          'official_gain':None,'clean716_overlaps_point_training':True,'release_verified':False,'results':results}
(package/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('results','promotion_basis')},indent=2))
