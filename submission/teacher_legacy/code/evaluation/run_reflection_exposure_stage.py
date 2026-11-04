"""Matched Q4 stage folds, with fresh local controls and no checkpoint selection."""
import argparse
import dataclasses
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import evaluate_b210_transfer as replay

SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE/'code/training'), str(SOURCE/'code/submissions'), str(SOURCE/'code')]
import train_b210_transfer_uq as uq

p = argparse.ArgumentParser(description=__doc__)
for name in ('root', 'sim', 'metadata', 'manifest', 'real-dir', 'template'):
    p.add_argument('--'+name, type=Path, required=True)
p.add_argument('--source-commit', required=True)
p.add_argument('--angles', type=int, nargs='+', required=True)
p.add_argument('--wall-cap', type=int, required=True)
p.add_argument('--resume-control', action='store_true')
p.add_argument('--exposure-wall-cap', type=int, choices=(1500,2100), default=1500)
a = p.parse_args()
assert a.angles in ([20, 0, 15], [5, 10]) and len(a.source_commit) == 40
assert 0 < a.wall_cap <= 10800
assert a.root.exists() == a.resume_control
if a.resume_control:
    assert a.angles == [5,10] and a.exposure_wall_cap == 2100
    assert not (a.root/'held5/reflection').exists()
assert replay.sha256(a.template) == '83d60c9aff83b07373b31d9b036e0ca42001402d8c6d12c19a10b8166b62e621'
assert replay.sha256(a.manifest) == uq.MANIFEST_SHA
a.root.mkdir(parents=True, exist_ok=a.resume_control)
deadline = time.monotonic()+a.wall_cap
environment = {**os.environ, 'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4'}
replay.TIME_SCORE = 92.032660


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str)+'\n')


def run(command, log, cap):
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    save(a.root/'status.json', {'command':command, 'log':str(log), 'started_epoch':time.time()})
    with log.open('x') as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True,
                       timeout=min(cap, max(1, deadline-time.monotonic())), env=environment)


def evaluate_fold(angle, folder, receipts):
    import torch
    from build_track1_blended_ensemble_unet2d_package import compact_state_dict
    from models.ensemble_unet2d import EnsembleUNet2dConfig
    torch.set_num_threads(4)
    fit = [r for r in replay.clean_rows(a.manifest) if r['aoa'] != angle]
    for label, receipt in receipts.items():
        point = receipt['point_receipt']
        uq.validate_point_receipt(point, 'C', True, expected_updates=4000 if label=='control' else 5334)
        assert point['held_aoa'] == receipt['held_aoa'] == angle
        assert point['fit_files'] == [r['file'] for r in fit]
        allowed_sources = {a.source_commit}
        if a.resume_control and label=='control' and angle==5:
            allowed_sources.add('b9ecbefd36f00ea6cce437c16f16fbf25fb0d34c')
        assert point['source_commit'] in allowed_sources
        assert receipt['point_receipt_sha256'] == replay.sha256(folder/label/'point/point_endpoint.json')
        assert receipt['sealed'] and receipt['point_state_bitwise'] and receipt['updates']==3200 and receipt['seed']==18
        assert receipt['tail_arrays_read']==0 and not receipt['checkpoint_selection']
        assert receipt['initialization']=='point_checkpoint' and receipt['objective_kind']=='sharp2'
        assert receipt['manifest_sha256']==uq.MANIFEST_SHA and receipt['fit_windows']==sum(len(r['train_starts']) for r in fit)
        assert replay.sha256(receipt['checkpoint_path'])==receipt['checkpoint_sha256']
    for key in ('sample_stream_sha256', 'normalizer_sha256', 'model_config_sha256'):
        assert receipts['control'][key] == receipts['reflection'][key], key
    assert receipts['control']['point_receipt'].get('vertical_reflection') is None
    results, bindings = {}, {}
    with zipfile.ZipFile(a.template) as z:
        template = torch.load(io.BytesIO(z.read('ensemble_unet2d.pth')), map_location='cpu', weights_only=False)
        template_cfg = dict(template['model_config']); assert template_cfg.pop('architecture')=='ensemble_unet'
        for label, receipt in receipts.items():
            payload = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)
            assert payload['iteration']==3200 and payload['phase']=='real'
            cfg = dict(payload['model_config']); assert cfg.pop('architecture')=='ensemble_unet'
            assert dataclasses.asdict(EnsembleUNet2dConfig(**cfg))==dataclasses.asdict(EnsembleUNet2dConfig(**template_cfg))
            assert replay.canonical_sha256(payload['model_config'])==receipt['model_config_sha256']
            runtime = folder/label/'runtime'; runtime.mkdir()
            for name in replay.RUNTIME_FILES:
                target = runtime/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(z.read(name))
            compact, _ = compact_state_dict(payload['model_state_dict'], 1024)
            torch.save({'model_config':template['model_config'], 'quantized_state_dict':compact,
                        'normalizer':payload['normalizer'], 'phase':'real', 'inference_dtype':'float16',
                        'storage_format':'symmetric_per_dim0_int8_v1'}, runtime/'ensemble_unet2d.pth')
            del payload, compact
            bindings[label] = {'checkpoint_sha256':replay.sha256(runtime/'ensemble_unet2d.pth'),
                               'receipt_sha256':replay.sha256(folder/label/'uq/uq_endpoint.json')}
            out = folder/label/'evaluation.json'
            run([sys.executable, str(SOURCE/'code/evaluation/evaluate_b210_transfer.py'), '--execute',
                 '--worker-runtime',str(runtime),'--worker-label',label,'--worker-output',str(out),
                 '--study','real-reflection','--device','cuda:0','--held-aoa',str(angle),
                 '--clean-manifest',str(a.manifest),'--real-dir',str(a.real_dir)], folder/(label+'_evaluation.log'), 240)
            results[label] = json.loads(out.read_text())
    for key in ('input_sha256','target_sha256','scorer_sha256'):
        assert results['control'][key]==results['reflection'][key]
    delta = {k:results['reflection']['metrics'][k]-results['control']['metrics'][k]
             for k in (*replay.METRICS,'fixed_time_fitted_final')}
    result = {'study':'real-reflection-exposure', 'held_aoa':angle, 'source_commit':a.source_commit,
              'fixed_time_score':92.032660, 'arms':results, 'delta':delta, 'bindings':bindings,
              'device':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True).strip(),
              'torch_version':torch.__version__, 'full_fit_development_screen_used':True,
              'scope':'Real/UQ stage LOAO only; shared Sim has Real ancestry; fresh same-device controls',
              'official_gain':None, 'release_verified':False}
    save(folder/'evaluation.json', result)
    print(json.dumps({'angle':angle,'delta':delta}), flush=True)
    return delta


save(a.root/('resume_plan.json' if a.resume_control else 'plan.json'), {**vars(a), 'started_epoch':time.time(), 'hypothesis':'Q4 physical reflection with matched original-view exposures',
                         'gate':{'pooled_final':.06,'positive_angles':4,'worst_angle':-.03}},)
deltas = {}
try:
    for angle in a.angles:
        folder = a.root/f'held{angle}'; folder.mkdir(exist_ok=a.resume_control)
        receipts = {}
        for label in ('control','reflection'):
            arm = folder/label; arm.mkdir(exist_ok=a.resume_control)
            point_cap = a.exposure_wall_cap if label=='reflection' else 1500
            entry = 'train_b210_transfer_real4k.py' if label=='control' else 'train_b210_reflection_exposure.py'
            command = [sys.executable,str(SOURCE/'code/training'/entry),'--execute','--arm','C',
                       '--held-aoa',str(angle),'--full-real-trajectories','--source-commit',a.source_commit,
                       '--start-sim',str(a.sim),'--start-sim-sha256','628860c7d82239d942895f08a6fd72ede81feb6a7cad4de4e32e110ed2742b9b',
                       '--start-metadata',str(a.metadata),'--start-metadata-sha256','282307574088299de967a7d64597a417150dd81cb18ea7b8d2897ee29b00e96f',
                       '--clean-manifest',str(a.manifest),'--real-dir',str(a.real_dir),'--output-dir',str(arm/'point'),
                       '--device','cuda:0','--wall-budget-seconds',str(point_cap)]
            if label=='reflection': command += ['--vertical-reflection']
            point_receipt = arm/'point/point_endpoint.json'
            if a.resume_control and angle==5 and label=='control':
                receipt = json.loads(point_receipt.read_text())
                uq.validate_point_receipt(receipt, 'C', True)
                assert replay.sha256(receipt['checkpoint_path'])==receipt['checkpoint_sha256']
            else:
                run(command, arm/'point.log', point_cap+150)
            command = [sys.executable,str(SOURCE/'code/training/train_b210_transfer_uq.py'),'--execute',
                       '--label',label,'--point-label','C','--held-aoa',str(angle),'--full-real-point','--reflection-comparison',
                       '--initialization','point','--source-commit',a.source_commit,'--point-receipt',str(point_receipt),
                       '--point-receipt-sha256',replay.sha256(point_receipt),'--clean-manifest',str(a.manifest),
                       '--real-dir',str(a.real_dir),'--output-dir',str(arm/'uq'),'--device','cuda:0','--wall-budget-seconds','600']
            if label=='reflection': command += ['--reflection-exposure-point']
            run(command, arm/'uq.log', 700)
            receipts[label] = json.loads((arm/'uq/uq_endpoint.json').read_text())
        deltas[str(angle)] = evaluate_fold(angle, folder, receipts)
        save(a.root/'progress.json', deltas)
        if deltas[str(angle)]['fixed_time_fitted_final'] < -.03:
            save(a.root/'RESULT.json', {'decision':'STOP_EXPOSURE_STAGE_WORST_FOLD', 'by_fold':deltas})
            break
    else:
        save(a.root/'RESULT.json', {'decision':'SHARD_COMPLETE_NEEDS_FIVE_FOLD_POOL', 'by_fold':deltas})
except Exception as error:
    save(a.root/'failure.json', {'type':type(error).__name__,'message':str(error),'by_fold':deltas})
    raise
