"""One fixed deployed-weight midpoint, conditional on full-control reproduction."""
import dataclasses
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import torch
import evaluate_b210_transfer as replay

SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE/'code/submissions'), str(SOURCE/'code')]
from build_track1_blended_ensemble_unet2d_package import compact_state_dict
from models.ensemble_unet2d import EnsembleUNet2dConfig

ROOT = Path('/root/track1_reflection_confirm_20260909')
OUT = Path('/root/track1_reflection_weight_mean_20260909')
BASE = Path('/root/autodl-tmp/track1_direct_pinned_20260909/candidate.zip')
V3 = Path('/root/autodl-tmp/spuq2_a800/artifacts/track1_v3.zip')
KEYS = (*replay.METRICS, 'fixed_time_fitted_final')
replay.TIME_SCORE = 92.032660
torch.set_num_threads(4)
official = replay.import_file(replay.SCORER, 'mean_scoring')
deadline = time.monotonic()+1800


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def difference(candidate, control):
    return {k:candidate[k]-control[k] for k in KEYS}


def same_inputs(a, b):
    for key in ('input_sha256', 'target_sha256', 'scorer_sha256'):
        assert a[key] == b[key], key


def deployed_state(payload):
    compact = payload.get('quantized_state_dict')
    if compact is None:
        compact, _ = compact_state_dict(payload['model_state_dict'], 1024)
    return {k:((v['quantized'].float()*v['scale'].float()).half().float()
               if isinstance(v, dict) else v.half().float() if v.is_floating_point() else v)
            for k,v in compact.items()}


def config(payload):
    cfg = dict(payload['model_config'])
    assert cfg.pop('architecture') == 'ensemble_unet'
    return dataclasses.asdict(EnsembleUNet2dConfig(**cfg))


def runtime_mean(template_zip, a, b, destination):
    assert config(a) == config(b)
    assert replay.canonical_sha256(a['normalizer']) == replay.canonical_sha256(b['normalizer'])
    x, y = deployed_state(a), deployed_state(b)
    assert x.keys() == y.keys()
    for k in x:
        assert x[k].shape == y[k].shape and x[k].dtype == y[k].dtype
        if x[k].is_floating_point():
            x[k] = (x[k]+y[k])*.5
            assert torch.isfinite(x[k]).all(), k
        else:
            assert torch.equal(x[k], y[k]), k
    compact, _ = compact_state_dict(x, 1024)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(template_zip) as z:
        template = torch.load(io.BytesIO(z.read('ensemble_unet2d.pth')), map_location='cpu', weights_only=False)
        assert config(template) == config(a)
        for name in replay.RUNTIME_FILES:
            target = destination/name
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_bytes(z.read(name))
    torch.save({'model_config':template['model_config'], 'quantized_state_dict':compact,
                'normalizer':a['normalizer'], 'phase':'real', 'inference_dtype':'float16',
                'storage_format':'symmetric_per_dim0_int8_v1'}, destination/'ensemble_unet2d.pth')


def evaluate(runtime, label, angle=None):
    path = OUT/(label+'.json')
    args = [sys.executable, str(SOURCE/'code/evaluation/evaluate_b210_transfer.py'), '--execute',
            '--worker-runtime', str(runtime), '--worker-label', label, '--worker-output', str(path),
            '--device', 'cuda:0', '--study', 'real-reflection',
            '--clean-manifest', '/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json',
            '--real-dir', '/root/autodl-tmp/realpde/data/real']
    if angle is not None:
        args += ['--held-aoa', str(angle)]
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    with (OUT/(label+'.log')).open('x') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=max(1,deadline-time.monotonic()),
                       env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
    return json.loads(path.read_text())


def main():
    OUT.mkdir(exist_ok=False)
    assert replay.sha256(BASE) == '83d60c9aff83b07373b31d9b036e0ca42001402d8c6d12c19a10b8166b62e621'
    assert replay.sha256(V3) == replay.V3_SHA
    check = json.loads((ROOT/'final_control_check/package/RESULT.json').read_text())
    reproduced = {arm:replay.finalize(official,check['results'][arm]['pooled_state']) for arm in ('baseline1','candidate1')}
    parity = difference(reproduced['candidate1'], reproduced['baseline1'])
    save(OUT/'control_parity.json', parity)
    if any(abs(parity[k])>.02 for k in replay.METRICS) or abs(parity['fixed_time_fitted_final'])>.01:
        save(OUT/'RESULT.json', {'decision':'STOP_WEIGHT_MEAN_CONTROL_NOT_REPRODUCED','parity':parity})
        return
    controls, means, fold_deltas, bindings = [], [], {}, {}
    for angle in (20,5,0,10,15):
        parent = Path('/root/autodl-tmp/track1_reflection_20260909') if angle in (5,20) else ROOT
        stage = parent/f'uq_stage{angle}'
        original = json.loads((stage/'evaluation.json').read_text())
        payloads = []
        bindings[str(angle)] = {}
        for arm in ('control','reflection'):
            receipt_path = stage/arm/'uq_endpoint.json'
            assert replay.sha256(receipt_path) == original['receipts'][arm]['sha256']
            receipt = json.loads(receipt_path.read_text())
            assert receipt['held_aoa'] == angle and receipt['point_receipt']['held_aoa'] == angle
            assert replay.sha256(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
            bindings[str(angle)][arm] = {'receipt_sha256':replay.sha256(receipt_path),'checkpoint_sha256':receipt['checkpoint_sha256']}
            payloads.append(torch.load(receipt['checkpoint_path'],map_location='cpu',weights_only=False))
        runtime = OUT/f'fold{angle}'
        runtime_mean(V3, *payloads, runtime)
        del payloads
        mean = evaluate(runtime, f'mean{angle}', angle)
        control = original['arms']['control']
        same_inputs(mean, control)
        fold_deltas[str(angle)] = difference(mean['metrics'],control['metrics'])
        controls.append(control);means.append(mean)
        save(OUT/'progress.json', {'by_fold':fold_deltas,'bindings':bindings})
        if fold_deltas[str(angle)]['fixed_time_fitted_final'] < -.03:
            save(OUT/'RESULT.json', {'decision':'STOP_WEIGHT_MEAN_FOLD','by_fold':fold_deltas})
            return
    pooled = {k:replay.finalize(official,replay.merge_states([r['pooled_state'] for r in rows]))
              for k,rows in (('control',controls),('mean',means))}
    delta = difference(pooled['mean'],pooled['control'])
    passed = delta['fixed_time_fitted_final'] >= .06 and sum(d['fixed_time_fitted_final']>0 for d in fold_deltas.values())>=4
    evidence = {'decision':'PASS_WEIGHT_MEAN_FIVE_FOLD' if passed else 'STOP_WEIGHT_MEAN_FIVE_FOLD',
                'metrics':pooled,'delta':delta,'by_fold':fold_deltas,'bindings':bindings,'alpha':.5,
                'scope':'Real/UQ stage LOAO, shared Sim Real ancestry; development screens used', 'official_gain':None}
    save(OUT/'five_fold_result.json',evidence)
    if not passed:
        save(OUT/'RESULT.json',evidence)
        return
    rejected = ROOT/'final/package/quality.zip'
    assert replay.sha256(rejected) == '5f271341fb4a842b5054bb3f081bbc48c812d063e69d74e52755c8c20133619c'
    with zipfile.ZipFile(BASE) as a, zipfile.ZipFile(rejected) as b:
        runtime_mean(BASE,torch.load(io.BytesIO(a.read('ensemble_unet2d.pth')),map_location='cpu',weights_only=False),
                     torch.load(io.BytesIO(b.read('ensemble_unet2d.pth')),map_location='cpu',weights_only=False),OUT/'final/runtime')
    package = OUT/'final/package';package.mkdir()
    dst = package/'quality.zip'
    with zipfile.ZipFile(dst,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in (*replay.RUNTIME_FILES,'ensemble_unet2d.pth'):
            z.write(OUT/'final/runtime'/name,name)
    baseline = ROOT/'final/package/baseline1'
    results = {label:evaluate(runtime,label) for label,runtime in
               (('baseline1',baseline),('candidate1',OUT/'final/runtime'),('baseline2',baseline),('candidate2',OUT/'final/runtime'))}
    b,c = results['baseline1'],results['candidate1']
    for row in results.values():same_inputs(row,b)
    delta = difference(c['metrics'],b['metrics'])
    angles = {k:c['by_aoa'][k]['fixed_time_fitted_final']-v['fixed_time_fitted_final'] for k,v in b['by_aoa'].items()}
    jitter = {k:abs(results['baseline2']['metrics'][k]-b['metrics'][k]) for k in b['metrics']}
    repeat = {k:abs(results['candidate2']['metrics'][k]-c['metrics'][k]) for k in c['metrics']}
    stable = all(repeat[k]<=max(2*jitter[k],1e-4) for k in repeat)
    passed = stable and delta['fixed_time_fitted_final']>=0 and sum(v>=0 for v in angles.values())>=4
    result = {'decision':'PASS_QUALITY_ZIP_DEPLOYMENT_DIAGNOSTICS' if passed else 'STOP_WEIGHT_MEAN_FINAL',
              'zip_path':str(dst),'zip_sha256':replay.sha256(dst),'zip_bytes':dst.stat().st_size,
              'promotion_basis':evidence,'quality_delta_overlapping_diagnostic':delta,
              'by_aoa_overlapping_diagnostic':angles,'stable':stable,'baseline_repeat_jitter':jitter,
              'candidate_repeat_jitter':repeat,'official_gain':None,'clean716_overlaps_point_training':True,
              'inputs':{str(p):replay.sha256(p) for p in (BASE,rejected)},'results':results}
    save(package/'RESULT.json',result);save(OUT/'RESULT.json',result)


if __name__ == '__main__':
    main()
