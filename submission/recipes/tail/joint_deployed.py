"""Fine-tune the unchanged network through the complete deployed output function."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OLD = Path('/dev/shm/dual_gain_20260922')
BASE = Path('/root/autodl-tmp/track1_breakthrough_20260917')
sys.path.insert(0, str(OLD/'sps320'))
from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
import shape_model


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            d.update(b)
    return d.hexdigest()


def errors(point, truth):
    def rel(a, b):
        return (a-b).flatten(1).norm(dim=1)/b.flatten(1).norm(dim=1).clamp_min(1e-8)
    energy = lambda v: .5*(v-v.mean(1, keepdim=True)).square().mean(1).sum(-1)
    yy = list(range(8, 25, 2))
    return torch.stack((rel(point, truth), rel(energy(point), energy(truth)),
                        torch.stack([rel(point[:, :, yy, x].mean(1), truth[:, :, yy, x].mean(1))
                                     for x in (13, 21, 29, 37)]).mean(0)), -1)


def setup(angle, baseline='320'):
    assert baseline in ('320','080')
    if angle is None:
        assert baseline == '080'
        stage = Path('/root/track1_sps080_preserved_20260921/full')
        rec = json.loads((ROOT/'full080_point_endpoint.json').read_text())
        urec = json.loads((ROOT/'full080_matched_uq_done.json').read_text())
    else:
        stage = (OLD/'spatial_joint' if baseline=='320' else ROOT/'baseline080')/f'held{angle}'
        rec = json.loads((stage/'point_endpoint.json').read_text())
        urec = json.loads((stage/'matched_uq_done.json').read_text())
    pp, up = stage/'ema999_u5334_primary.pth', stage/'matched_uq_3200.pth'
    assert sha(pp) == rec['checkpoint_sha256'] and sha(up) == urec['checkpoint_sha256']
    assert rec['held_aoa'] == urec['held_aoa'] == angle
    assert all(int(Path(n).stem.split('_')[-1]) != angle for n in rec['fit_files'])
    p = torch.load(str(pp), map_location='cpu', mmap=True, weights_only=False)
    u = torch.load(str(up), map_location='cpu', mmap=True, weights_only=False)
    assert p['normalizer'] == u['normalizer']
    template = torch.load(str(OLD/f'sps{baseline}/ensemble_unet2d.pth'), map_location='cpu', mmap=True, weights_only=False)
    cfg = dict(template['model_config']); cfg.pop('architecture')
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**cfg)).cuda().float()
    state = dict(p['model_state_dict']); state.update(u['model_state_dict'])
    if angle is None:
        # The full-fit starting tensors must reproduce the shipped 0.80 member.
        names = [k for k,v in template['quantized_state_dict'].items() if isinstance(v,dict)]
        restored = quantize(state,names)
        for k,v in template['quantized_state_dict'].items():
            expected = (v['quantized'].float()*v['scale'].float()).half() if isinstance(v,dict) else (v.half() if v.is_floating_point() else v)
            assert torch.equal(restored[k],expected), k
        del restored
    model.load_state_dict(state, strict=True)
    stats = {k: torch.tensor(v, device='cuda', dtype=torch.float32) for k, v in p['normalizer'].items()}
    teacher_file = OLD/'sps320/shape_model.pth' if angle is None else BASE.parent/f'track1_normalized_point_20260918/T_held{angle}/ema999_u20000.pth'
    teacher_state = torch.load(teacher_file, map_location='cpu', weights_only=False)
    assert teacher_state.get('angle') == angle
    assert all(int(Path(n).stem.split('_')[-1]) != angle for n in teacher_state['fit_files'])
    if angle is None:
        assert sorted(teacher_state['fit_files']) == sorted(rec['fit_files'])
    teacher = shape_model.PointModel().cuda().half().eval().requires_grad_(False)
    teacher.load_state_dict(teacher_state['state_dict'], strict=True)
    table_path = OLD/'sps320/mean_calibration.npz' if angle is None else BASE/f'assets/mean_held{angle}.npz'
    table = np.load(table_path)
    region = (((np.arange(32)[:, None]//16)*4+np.arange(64)[None, :]//16)*2)[..., None]+np.arange(2)
    calibration = (torch.tensor(table['coefficients'][region][None, None], device='cuda', dtype=torch.float32),
                   float(table['center']), float(table['spread']), float(table['shrink']))
    ident = dict(held=angle, baseline=baseline, point_sha256=sha(pp), uq_sha256=sha(up), teacher_sha256=sha(teacher_file),
                 mean_sha256=sha(table_path), normalizer=p['normalizer'], fit_files=rec['fit_files'],
                 model_config=template['model_config'], seed=20260923,
                 evidence='Full-fit released data; training overlap, not generalization' if angle is None else 'Real/UQ/T held angle with shared Sim ancestry; not full pipeline holdout')
    quantized_names = [k for k, v in template['quantized_state_dict'].items() if isinstance(v, dict)]
    return model, stats, teacher, calibration, quantized_names, ident


def differentiable_mix(point, other):
    """Use the same energy-preserving mix, with a finite derivative at zero energy."""
    uv = point[..., :2]
    mean = uv.mean(1, keepdim=True)
    pd, td = uv-mean, other[..., :2]-other[..., :2].mean(1, keepdim=True)
    dynamic = .5*(pd+td)
    target = pd.square().sum(-1).mean(1)
    energy = dynamic.square().sum(-1).mean(1)
    gain = (target.clamp_min(1e-12)/energy.clamp_min(1e-12)).sqrt()
    valid = (energy > target*1e-6) & (target > 1e-12)
    dynamic = torch.where(valid[:,None,:,:,None], dynamic*gain[:,None,:,:,None], pd)
    return torch.cat((mean+dynamic, point[...,2:]), -1)


def bundle(raw, model, stats, teacher, calibration):
    dtype = next(model.parameters()).dtype
    st = {k: v.to(dtype) for k, v in stats.items()}
    xn = (raw.to(dtype)-st['mean_input'])/st['std_input']
    prediction, _, q, le, ue, _, _ = model.forward_uq_training(xn)
    native = (prediction*st['std_target']+st['mean_target'])[..., :2]
    lo = (native-le*st['std_target'][:2]).float()
    hi = (native+ue*st['std_target'][:2]).float()
    native = native.float()
    coefficients, center, spread, shrink = calibration
    uv = raw[..., :2]
    scale = .5*uv.square().mean((1, 2, 3, 4)).clamp_min(1e-12).log()
    f = ((scale-center)/spread)[:, None, None, None, None]
    alpha = (coefficients[..., 0]+f*coefficients[..., 1]).clamp(0, .5)*shrink
    active = (uv != 0).any(1, keepdim=True)
    shift = (uv.mean(1, keepdim=True)-native.mean(1, keepdim=True))*alpha*active
    native, lo, hi = native+shift, lo+shift, hi+shift
    zero = torch.zeros_like(native[..., :1])
    p = torch.cat((native, zero), -1)
    with torch.set_grad_enabled(model.training and teacher.training):
        t = teacher(raw)
    p = differentiable_mix(p, t) if model.training else shape_model.reconstruct(p, t)
    qphysical = q.float()*st['std_target'][:2].float()[..., None]
    center = native+(qphysical*q.new_tensor([.1, .125, .175, .2, .175, .125, .1])).sum(-1)
    center = torch.where(active, center, native)
    p = differentiable_mix(p, center) if model.training else shape_model.mix_distribution_shape(p, center)
    return p, torch.cat((lo, zero), -1), torch.cat((hi, zero), -1), qphysical, native


def objective(outputs, truth, tau=None):
    p, lo, hi, q, native = outputs
    y = truth[..., :2]; p, lo, hi = p[..., :2], lo[..., :2], hi[..., :2]
    er = errors(p, y)
    point = -((1/(1+.5*er.mean(0)))*er.new_tensor([.5, .1, .1])).sum()
    factor = (.5/(.5+er)*er.new_tensor([.5, .3, .2])).sum(-1)
    valid = y != 0
    ref = y.new_tensor([.01075, .00550])
    if tau is None:
        tau = .05*ref
    inside = torch.sigmoid((y-lo)/tau)*torch.sigmoid((hi-y)/tau)
    utility = (inside*torch.exp(-(hi-lo)/.0563870259)*valid*factor[:, None, None, None, None]).sum()/valid.sum()
    residual = y-native.detach()
    levels = q.new_tensor([.05, .15, .3, .5, .7, .85, .95])
    diff = residual[..., None]-q
    pin = torch.maximum(levels*diff, (levels-1)*diff)/ref[..., None]
    pin = (pin*valid[..., None]).sum()/(valid.sum()*7)
    loss = point-.25*utility+.01*pin
    return loss, dict(point=float(point.detach()), soft_sps=float(utility.detach()), pinball=float(pin.detach()))


def data(rows):
    trajectories, windows = [], []
    for row in rows:
        with h5py.File(Path('/root/autodl-tmp/realpde/data/real')/row['file']) as f:
            assert int(f['aoa'][()]) == row['aoa']
            uv = np.stack([np.asarray(f[k][:, ::2, ::2], dtype=np.float32) for k in ('u', 'v')], -1)
        value = np.concatenate((uv, np.zeros_like(uv[..., :1])), -1)
        index = len(trajectories)
        trajectories.append(torch.from_numpy(value).cuda())
        windows.extend((index, start) for start in range(0, len(value)-39, 5))
    return trajectories, windows


def quantize(state, names):
    result = {}
    for name, value in state.items():
        value = value.float() if value.is_floating_point() else value
        if name in names:
            scale = (value.abs().amax(tuple(range(1, value.ndim)), keepdim=True)/127).clamp_min(torch.finfo(torch.float32).tiny)
            value = (value/scale).round().clamp(-127, 127)*scale
        result[name] = value.half() if value.is_floating_point() else value
    return result


@torch.no_grad()
def evaluate(angle, model, stats, teacher, calibration, names, original, endpoint, out):
    ev = load(BASE/'src/code/evaluation/evaluate_b210_transfer.py', 'joint_eval')
    scorer = load(BASE/'src/kits/track1/realpde_t1_starting_kit_v9/scoring.py', 'joint_scorer')
    rows = [r for r in ev.clean_rows(BASE/'manifest.json') if angle is None or r['aoa'] == angle]
    all_states = {}
    model.half().eval()
    for arm, weights in [('baseline', original), ('candidate', endpoint)]:
        model.load_state_dict(quantize(weights, names), strict=True)
        states = {}
        for x, y, labels in ev.tail_batches(rows, Path('/root/autodl-tmp/realpde/data/real')):
            p, lo, hi, _, _ = bundle(torch.from_numpy(x).cuda(), model, stats, teacher, calibration)
            assert all(torch.isfinite(z).all() for z in (p, lo, hi)) and torch.all(lo <= hi)
            arrays = [v.cpu().numpy() for v in (p, lo, hi)]
            for n in dict.fromkeys(labels):
                ix = np.asarray(labels) == n
                states.setdefault(n, []).append(ev.metric_state(scorer, arrays[0][ix], y[ix], arrays[1][ix], arrays[2][ix]))
        all_states[arm] = {n: ev.merge_states(v) for n, v in states.items()}
    metrics = {a: {k: v for k, v in ev.finalize(scorer, ev.merge_states(list(s.values()))).items()
                   if k not in ('time_score', 'fixed_time_fitted_final')} for a, s in all_states.items()}
    result = dict(angle=angle, metrics=metrics, delta={k:metrics['candidate'][k]-v for k,v in metrics['baseline'].items()},
                  raw_states=all_states, evidence='Paired full-fit deployed-function; training overlap' if angle is None else 'Paired deployed-function stage-held-angle, quantized, fixed endpoint')
    (out/'evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(event='evaluation', angle=angle, delta=result['delta'])), flush=True)


def main(angle):
    started = time.monotonic(); torch.set_num_threads(4); torch.manual_seed(20260923)
    torch.backends.cudnn.allow_tf32 = True; torch.backends.cuda.matmul.allow_tf32 = True
    out = ROOT/f'joint_held{angle}'; out.mkdir(exist_ok=False)
    model, stats, teacher, calibration, names, identity = setup(angle)
    original = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    (out/'identity.json').write_text(json.dumps(identity, indent=2))
    ev = load(BASE/'src/code/evaluation/evaluate_b210_transfer.py', 'joint_data')
    rows = [r for r in ev.clean_rows(BASE/'manifest.json') if r['aoa'] != angle]
    assert sorted(r['file'] for r in rows) == sorted(identity['fit_files'])
    trajectories, windows = data(rows)
    model.train().requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    ema = {k:v.detach().clone() for k,v in model.state_dict().items()}
    rng = np.random.default_rng(20260923)
    train_start = time.monotonic()
    for step in range(1, 2401):
        assert time.monotonic()-train_start < 1200, 'training budget exceeded'
        chosen = [windows[i] for i in rng.integers(0, len(windows), 8)]
        raw = torch.stack([trajectories[i][s:s+20] for i,s in chosen])
        truth = torch.stack([trajectories[i][s+20:s+40] for i,s in chosen])
        flip = torch.tensor(rng.random(8) < .25, device='cuda')
        sign = raw.new_tensor([1., -1., 1.])
        raw = torch.where(flip[:,None,None,None,None], raw.flip(2)*sign, raw)
        truth = torch.where(flip[:,None,None,None,None], truth.flip(2)*sign, truth)
        outputs = bundle(raw, model, stats, teacher, calibration)
        loss, detail = objective(outputs, truth)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if step == 1:
            assert grad > 0 and all(p.grad is None for p in teacher.parameters())
            scorer = load(BASE/'src/kits/track1/realpde_t1_starting_kit_v9/scoring.py', 'joint_preflight')
            pn, yn = outputs[0].detach().cpu().numpy(), truth.cpu().numpy()
            er = errors(outputs[0][...,:2], truth[...,:2]).detach().cpu().numpy()
            reference = np.stack([scorer.rel_l2_per_sample(pn,yn,2), scorer.tke_rel_l2_per_sample(pn,yn,2), scorer.mvpe_rel_l2_per_sample(pn,yn)], -1)
            assert np.abs(er-reference).max() < 3e-6
            lo, hi = outputs[1].detach(), outputs[2].detach()
            y = truth[..., :2]; valid = y != 0
            factor = (.5/(.5+errors(outputs[0][...,:2].detach(), y))*y.new_tensor([.5,.3,.2])).sum(-1)
            reward = ((y >= lo[...,:2]) & (y <= hi[...,:2]))*torch.exp(-(hi[...,:2]-lo[...,:2])/.0563870259)
            hard = float((reward*valid*factor[:,None,None,None,None]).sum()/valid.sum())
            reference_sps, _ = scorer.aggregate_sps(pn, yn, 2, lo.cpu().numpy(), hi.cpu().numpy())
            assert abs(hard-reference_sps) < 3e-6
            assert len(model.config.quantile_levels) == 7 and np.allclose(model.config.quantile_levels,[.05,.15,.3,.5,.7,.85,.95])
            (out/'first_batch.json').write_text(json.dumps(dict(gradient_norm=float(grad), teacher_frozen=True,
                error_maxabs=float(np.abs(er-reference).max()), hard_sps_abs=abs(hard-reference_sps), training_windows=len(windows), steps=2400)))
        lr = 2e-6+.5*(1e-5-2e-6)*(1+math.cos(math.pi*step/2400))
        for group in opt.param_groups: group['lr'] = lr
        opt.step()
        with torch.no_grad():
            for k,v in model.state_dict().items():
                if v.is_floating_point(): ema[k].lerp_(v.detach(), .001)
                else: ema[k].copy_(v)
        if step == 1 or step % 200 == 0:
            print(json.dumps(dict(event='training',angle=angle,step=step,loss=float(loss.detach()),grad=float(grad),
                                  seconds=time.monotonic()-train_start,**detail)), flush=True)
    endpoint = {k:v.detach().cpu() for k,v in ema.items()}
    path = out/'joint_model.pth'
    torch.save(dict(model_state_dict=endpoint, normalizer=identity['normalizer'], model_config=identity['model_config'],
                    identity=identity, steps=2400, method='joint deployed-function fine tune'),path)
    (out/'endpoint.json').write_text(json.dumps(dict(sha256=sha(path),bytes=path.stat().st_size,seconds=time.monotonic()-started)))
    del ema, opt, trajectories, outputs, loss; torch.cuda.empty_cache()
    evaluate(angle, model, stats, teacher, calibration, names, original, endpoint, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--angle',type=int,required=True)
    args = parser.parse_args()
    try: main(args.angle)
    except Exception as error:
        (ROOT/f'failure{args.angle}.json').write_text(json.dumps(dict(error=repr(error))))
        raise
