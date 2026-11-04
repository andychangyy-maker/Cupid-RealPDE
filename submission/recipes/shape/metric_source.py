"""Fine-tune the complete P point network on its final physical prediction."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch


def physical(runtime, model, stats, raw, flip=None):
    view = raw
    if flip is not None:
        reflected = raw.flip(2)*raw.new_tensor([1., -1., 1.])
        view = torch.where(flip[:, None, None, None, None], reflected, raw)
    normalized = (view.half()-stats['mean_input'])/stats['std_input']
    pn, _ = model.forward_with_features(normalized)
    p = pn*stats['std_target']+stats['mean_target']
    p = torch.cat((p[..., :2], torch.zeros_like(p[..., 2:])), -1)
    if flip is not None:
        reflected = p.flip(2)*p.new_tensor([1., -1., 1.])
        p = torch.where(flip[:, None, None, None, None], reflected, p)
    return runtime.mean.correct(raw, p[None].float())[0]


def errors(p, truth):
    p, t = p[..., :2].float(), truth[..., :2].float()
    def relative(a, b):
        return torch.linalg.vector_norm((a-b).flatten(1), dim=1)/torch.linalg.vector_norm(b.flatten(1), dim=1).clamp_min(1e-8)
    rel = relative(p, t)
    pm, tm = p.mean(1, keepdim=True), t.mean(1, keepdim=True)
    pk = .5*(p-pm).square().sum(-1).mean(1)
    tk = .5*(t-tm).square().sum(-1).mean(1)
    tke = relative(pk, tk)
    ys = list(range(8, 25, 2))
    mvpe = torch.stack([relative(pm[:, 0, ys, x, :], tm[:, 0, ys, x, :]) for x in [13,21,29,37]]).mean(0)
    return rel, tke, mvpe


def evaluate(runtime, control, candidate, stats, rows, out):
    states = {a: {} for a in ['P', 'M']}
    with torch.inference_mode():
        for history, truth, labels in runtime.ev.tail_batches(rows, Path(runtime.plan['real_dir'])):
            raw = torch.from_numpy(history).cuda()
            for arm, model in [('P', control), ('M', candidate)]:
                prediction = physical(runtime, model, stats, raw).cpu().numpy()
                values = [runtime.scorer.rel_l2_per_sample(prediction, truth, 2), runtime.scorer.tke_rel_l2_per_sample(prediction, truth, 2), runtime.scorer.mvpe_rel_l2_per_sample(prediction, truth)]
                for name in dict.fromkeys(labels):
                    mask = np.asarray(labels) == name
                    state = states[arm].setdefault(name, {'samples':0, 'rel':0., 'tke':0., 'mvpe':0.})
                    state['samples'] += int(mask.sum())
                    for key, v in zip(['rel','tke','mvpe'], values):
                        state[key] += float(v[mask].sum(dtype=np.float64))
    metrics = {}
    for arm, per in states.items():
        n = sum(v['samples'] for v in per.values())
        metrics[arm] = {k: runtime.scorer.score_error(sum(v[k] for v in per.values())/n) for k in ['rel','tke','mvpe']}
    delta = {k: metrics['M'][k]-metrics['P'][k] for k in metrics['P']}
    per_file = {name: {k: runtime.scorer.score_error(states['M'][name][k]/p['samples'])-runtime.scorer.score_error(p[k]/p['samples']) for k in ['rel','tke','mvpe']} for name, p in states['P'].items()}
    result = {'angle':runtime.angle, 'metrics':metrics, 'delta':delta, 'raw_states':states, 'trajectory_delta':per_file,
              'evidence':'stage-held-angle; deployed INT8/FP16 point; matched UQ not yet trained; no official prediction'}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='raw_states'}),flush=True)


def main(root, old, angle):
    sys.path.insert(0, str(old))
    from point_runtime_k1 import Runtime
    from diagnose_k0 import compact
    torch.set_num_threads(4); torch.manual_seed(18018)
    assert torch.cuda.device_count() == 1
    runtime = Runtime(old, angle)
    model, stats = runtime.load('P')
    model.float().train().requires_grad_(True)
    for name, p in model.named_parameters():
        if name.startswith(('uq_', 'quantile_head', 'coverage_calibration_head', 'interval_refiner')):
            p.requires_grad_(False)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    out = root/f'M_held{angle}'; out.mkdir(exist_ok=False)
    rows = runtime.ev.clean_rows(Path(runtime.plan['manifest']))
    fit = [r for r in rows if r['aoa'] != angle]
    assert len({r['aoa'] for r in fit}) == 4
    trajectories, windows = [], []
    for row in fit:
        with h5py.File(Path(runtime.plan['real_dir'])/row['file'], 'r') as f:
            assert int(f['aoa'][()]) != angle and len(f['u']) == row['frames']
            u = np.asarray(f['u'][:, ::2, ::2], dtype=np.float32)
            v = np.asarray(f['v'][:, ::2, ::2], dtype=np.float32)
        trajectories.append(np.stack((u,v,np.zeros_like(u)), -1))
        windows.extend((len(trajectories)-1, t) for t in range(0,row['frames']-39,5))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=2e-6)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    rng = np.random.default_rng(18018)
    logs = []; start = time.monotonic()
    for step in range(1,3001):
        assert time.monotonic()-start < 2700, 'M training budget exceeded'
        chosen = rng.integers(len(windows), size=8)
        batch = np.stack([trajectories[windows[i][0]][windows[i][1]:windows[i][1]+40] for i in chosen])
        raw = torch.from_numpy(batch[:,:20]).cuda()
        truth = torch.from_numpy(batch[:,20:]).cuda()
        flip = torch.from_numpy(rng.random(8)<.25).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            prediction = physical(runtime, model, stats, raw, flip)
        e = errors(prediction, truth)
        means = [v.mean() for v in e]
        loss = sum(w*v/(2+v) for w,v in zip([5,2,1],means))/8
        assert torch.isfinite(loss)
        if step == 1:
            a, b = prediction.detach().cpu().numpy(), truth.cpu().numpy()
            reference = [runtime.scorer.rel_l2_per_sample(a,b,2), runtime.scorer.tke_rel_l2_per_sample(a,b,2), runtime.scorer.mvpe_rel_l2_per_sample(a,b)]
            for v, ref in zip(e, reference):
                assert np.allclose(v.detach().cpu().numpy(), ref, rtol=2e-5, atol=2e-6), 'metric formula mismatch'
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        assert float(norm) > 0
        scaler.step(optimizer); scaler.update(); scheduler.step()
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()):
                a.lerp_(b,.001)
        if step == 1 or step % 200 == 0:
            record = {'step':step, 'loss':float(loss.detach()), 'errors':[float(v.detach()) for v in means], 'grad_norm':float(norm), 'seconds':time.monotonic()-start}
            logs.append(record);(out/'progress.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
    for name, p in model.named_parameters():
        if name.startswith(('uq_', 'quantile_head', 'coverage_calibration_head', 'interval_refiner')):
            assert p.grad is None
    source = torch.load(runtime.plan['points']['P'+str(angle)]['path'], map_location='cpu', weights_only=False)
    endpoint = Path('/dev/shm/track1_breakthrough_20260917')/f'M{angle}_ema999_u3000.pth'
    assert not endpoint.exists()
    torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in ema.state_dict().items()},
                'model_config':source['model_config'], 'normalizer':source['normalizer'],
                'M_training':{'steps':3000,'seed':18018,'loss':'final physical score only 5:2:1','start_point_sha256':runtime.plan['points']['P'+str(angle)]['checkpoint_sha256']}},endpoint)
    with endpoint.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    receipt = {'checkpoint_path':str(endpoint),'checkpoint_sha256':digest,'fit_files':[r['file'] for r in fit],
               'fit_windows':len(windows),'held_angle':angle,'steps':3000,'training_seconds':time.monotonic()-start,'logs':logs,
               'storage':'RAM; must download and verify before shutdown'}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2))
    del source, model, optimizer, trajectories;torch.cuda.empty_cache()
    state={k:compact(v) for k,v in ema.state_dict().items()}
    ema.half().load_state_dict(state,strict=True);del state
    control,_ = runtime.load('P')
    evaluate(runtime,control,ema,stats,[r for r in rows if r['aoa']==angle],out)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--angle',type=int,choices=[0,20],required=True)
    a=p.parse_args();main(a.root,a.old,a.angle)
