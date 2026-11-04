"""Sealed Real-stage LOAO point screen; SPS and Time are fixed placeholders."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'training'), str(ROOT/'evaluation')]

p = argparse.ArgumentParser()
for name in ('control', 'candidate', 'manifest', 'real-dir', 'output'):
    p.add_argument('--'+name, type=Path, required=True)
a = p.parse_args()
import numpy as np
import torch
import evaluate_b210_transfer as ev
from train_b210_full_prefix_uq_refit import normalizer_from_point
from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
torch.set_num_threads(4)
torch.set_float32_matmul_precision('high')
receipts = {arm: json.loads(path.read_text()) for arm,path in [('control',a.control),('candidate',a.candidate)]}
for key in ('held_aoa','fit_files','optimizer_windows','sample_stream_sha256','start_sim_sha256','start_metadata_sha256','seed','updates'):
    assert receipts['control'][key] == receipts['candidate'][key], key
assert 'vertical_reflection' not in receipts['control']
assert receipts['candidate']['vertical_reflection']['probability'] == .25
held = receipts['control']['held_aoa']
assert held in (20,5)
rows = ev.clean_rows(a.manifest)
assert receipts['control']['fit_files'] == [x['file'] for x in rows if x['aoa'] != held]
models, normalizers, normalizer_reports = {}, {}, {}
for arm,receipt in receipts.items():
    assert ev.sha256(Path(receipt['checkpoint_path'])) == receipt['checkpoint_sha256']
    point = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)
    cfg = dict(point['model_config']); assert cfg.pop('architecture') == 'ensemble_unet'
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**cfg))
    model.load_state_dict(point['model_state_dict'], strict=True)
    models[arm] = model.cuda().eval()
    normalizers[arm], normalizer_reports[arm] = normalizer_from_point(point, torch.device('cuda'))
    del point
assert normalizer_reports['control'] == normalizer_reports['candidate']
official = ev.import_file(ev.SCORER, 'reflection_official')
sums = {arm: {} for arm in models}
deadline = time.monotonic()+600
with torch.inference_mode():
    for inputs,truth,labels in ev.tail_batches([x for x in rows if x['aoa']==held], a.real_dir):
        assert time.monotonic() < deadline
        for arm,model in models.items():
            normalizer = normalizers[arm]
            x,_ = normalizer.preprocess(torch.from_numpy(inputs).cuda(), torch.from_numpy(truth).cuda())
            pred = normalizer.postprocess_prediction(model(x)).float().cpu().numpy()
            pred[...,2] = 0
            assert np.isfinite(pred).all()
            errors = np.stack([official.rel_l2_per_sample(pred,truth,2), official.tke_rel_l2_per_sample(pred,truth,2),
                               official.mvpe_rel_l2_per_sample(pred,truth)], axis=1)
            for i,label in enumerate(labels):
                row = sums[arm].setdefault(label, {'n':0,'errors':np.zeros(3,dtype=np.float64)})
                row['n'] += 1; row['errors'] += errors[i]
def finish(group):
    n = sum(x['n'] for x in group.values())
    errors = sum(x['errors'] for x in group.values())/n
    scores = [official.score_error(float(v)) for v in errors]
    metrics = dict(zip(ev.METRICS[:3], scores))
    metrics['point_only_fitted_final'] = sum(w*s**.6854 for w,s in zip([.5,.1,.1,.1,.2], [*scores,92.032660,43.295329]))**(1/.6854)
    return metrics
metrics = {arm:finish(group) for arm,group in sums.items()}
delta = {k:metrics['candidate'][k]-metrics['control'][k] for k in metrics['control']}
passed = (delta['point_only_fitted_final'] >= .03 and delta['rel_l2_score'] >= -.03
          and delta['tke_rel_l2_score'] >= -.15 and delta['mvpe_rel_l2_score'] >= -.10)
result = {'decision':'PASS_REFLECTION_POINT_SCREEN' if passed else 'STOP_REFLECTION_POINT_CONFIG',
          'held_aoa':held, 'metrics':metrics, 'delta':delta, 'normalizer':normalizer_reports['control'],
          'receipts':receipts, 'scope':'FP32 point-only; Real stage LOAO, shared Sim ancestry; SPS/Time fixed, no official gain',
          'trajectory_deltas':{k:finish({k:sums['candidate'][k]})['point_only_fitted_final']-finish({k:sums['control'][k]})['point_only_fitted_final'] for k in sums['control']},
          'raw_states':{arm:{k:{'n':v['n'],'errors':v['errors'].tolist()} for k,v in group.items()} for arm,group in sums.items()}}
a.output.write_text(json.dumps(result,indent=2))
print(json.dumps({'decision':result['decision'],'delta':delta}),flush=True)
