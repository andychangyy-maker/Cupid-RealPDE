"""One fixed Real-only point experiment; no held-angle endpoint selection."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import shutil

import h5py
import numpy as np
import torch

from network import PointModel, combine

ROOT = Path('/root/autodl-tmp/track1_normalized_point_20260918')
OLD = Path('/root/autodl-tmp/track1_breakthrough_20260917')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def fit(runtime, rows, errors, out):
    trajectories, starts, offset = [], [], 0
    for row in rows:
        assert row['aoa'] != runtime.angle and row['file'] != '7575_0.h5'
        with h5py.File(Path(runtime.plan['real_dir'])/row['file'], 'r') as f:
            assert int(f['aoa'][()]) == row['aoa'] and len(f['u']) == row['frames']
            values = np.stack((np.asarray(f['u'][:, ::2, ::2], dtype=np.float32),
                               np.asarray(f['v'][:, ::2, ::2], dtype=np.float32)), -1)
        trajectories.append(torch.from_numpy(values).half())
        starts.extend(range(offset, offset+len(values)-39, 5)); offset += len(values)
    slab = torch.cat(trajectories).cuda(); del trajectories
    starts = torch.tensor(starts, device='cuda'); times = torch.arange(40, device='cuda')
    torch.manual_seed(18618)
    net = PointModel().cuda().train()
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20000, eta_min=3e-5)
    generator = torch.Generator(device='cuda').manual_seed(18619)
    started = time.monotonic(); reflected = 0
    print(json.dumps({'stage':'start', 'angle':runtime.angle, 'parameters':sum(p.numel() for p in net.parameters()),
                      'windows':len(starts), 'fit_files':len(rows), 'frames':len(slab)}), flush=True)
    for step in range(1, 20001):
        assert time.monotonic()-started < 7200 and shutil.disk_usage(out).free > 5*2**30, 'training wall/disk guard'
        ids = torch.randint(len(starts), (32,), generator=generator, device='cuda')
        batch = slab[starts[ids, None]+times[None]].float()
        flip = torch.rand(32, generator=generator, device='cuda') < .25
        reflected += int(flip.sum())
        batch = torch.where(flip[:, None, None, None, None],
                            batch.flip(2)*batch.new_tensor([1., -1.]), batch)
        history, truth = batch[:, :20], batch[:, 20:]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction, mean, rms = net(history, details=True)
        if step == 1:
            assert torch.allclose(prediction.mean(1), mean, rtol=1e-4, atol=1e-7)
            measured = (prediction-prediction.mean(1, keepdim=True)).square().mean(1).sqrt()
            assert torch.allclose(measured, rms, rtol=1e-4, atol=1e-7)
        scale = (history-history.mean(1, keepdim=True)).square().mean((1, 2, 3), keepdim=True).clamp_min(1e-8).sqrt()
        mse = ((prediction-truth)/scale).square().mean()
        rel, tke, mvpe = errors(prediction, truth)
        a, b, c = rel.mean(), tke.mean(), mvpe.mean()
        loss = mse if step <= 1000 else (5*a/(2+a)+2*b/(2+b)+c/(2+c))/8+.05*mse
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if step == 1:
            assert all(float(head.weight.grad.abs().sum()) > 0 for head in [net.mean_head, net.energy_head, net.shape_head])
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        with torch.no_grad():
            for dest, source in zip(ema.parameters(), net.parameters()): dest.lerp_(source, .001)
        if step == 1 or step % 1000 == 0:
            record = {'stage':'train', 'step':step, 'loss':float(loss.detach()), 'mse':float(mse.detach()),
                      'rel':float(a.detach()), 'tke':float(b.detach()), 'mvpe':float(c.detach()),
                      'gradient_norm':float(norm), 'seconds':time.monotonic()-started}
            (out/'progress.json').write_text(json.dumps(record, indent=2)); print(json.dumps(record), flush=True)
    endpoint = out/'ema999_u20000.pth'
    torch.save({'state_dict':{k:v.cpu() for k,v in ema.state_dict().items()}, 'angle':runtime.angle,
                'fit_files':[r['file'] for r in rows], 'seed':18618, 'batch_seed':18619,
                'steps':20000, 'ema':.999, 'initialization':'random; no pretrained checkpoint'}, endpoint)
    receipt = {'sha256':hashlib.sha256(endpoint.read_bytes()).hexdigest(), 'training_seconds':time.monotonic()-started,
               'parameters':sum(p.numel() for p in net.parameters()), 'training_windows':len(starts),
               'training_files':[r['file'] for r in rows], 'samples_drawn':20000*32, 'reflected':reflected,
               'manifest_sha256':hashlib.sha256(Path(runtime.plan['manifest']).read_bytes()).hexdigest()}
    (out/'point_receipt.json').write_text(json.dumps(receipt, indent=2))
    del slab, net, optimizer; torch.cuda.empty_cache()
    return ema.half().eval()


def evaluate(runtime, candidate, rows, out):
    base = module(ROOT.parent/'track1_dedicated_sps_20260918/train.py', 't_p_reference')
    _, control, stats = base.load_p(OLD, runtime.angle)
    states = {a:{} for a in ['P', 'T', 'H']}
    with torch.inference_mode():
        for history, truth, labels in runtime.ev.tail_batches(rows, Path(runtime.plan['real_dir'])):
            raw = torch.from_numpy(history).cuda()
            p, lo, hi = base.predict(runtime, control, stats, raw)
            t = candidate(raw)
            predictions = {'P':p, 'T':torch.cat((t, torch.zeros_like(t[..., :1])), -1),
                           'H':torch.cat((combine(p, t), torch.zeros_like(t[..., :1])), -1)}
            lower, upper = lo.cpu().numpy(), hi.cpu().numpy()
            for arm, prediction in predictions.items():
                assert torch.isfinite(prediction).all()
                values = prediction.cpu().numpy()
                for name in dict.fromkeys(labels):
                    mask = np.asarray(labels) == name
                    states[arm].setdefault(name, []).append(runtime.ev.metric_state(runtime.scorer, values[mask], truth[mask], lower[mask], upper[mask]))
    raw = {a:{n:runtime.ev.merge_states(v) for n,v in per.items()} for a,per in states.items()}
    metrics = {a:runtime.ev.finalize(runtime.scorer, runtime.ev.merge_states(list(per.values()))) for a,per in raw.items()}
    for v in metrics.values(): v.pop('time_score'); v.pop('fixed_time_fitted_final')
    delta = {a:{k:metrics[a][k]-metrics['P'][k] for k in metrics['P']} for a in ['T', 'H']}
    per = {a:{n:{k:runtime.ev.finalize(runtime.scorer,raw[a][n])[k]-runtime.ev.finalize(runtime.scorer,v)[k]
                 for k in ['rel_l2_score','tke_rel_l2_score','mvpe_rel_l2_score','sps_score']}
              for n,v in raw['P'].items()} for a in ['T', 'H']}
    result = {'angle':runtime.angle, 'metrics':metrics, 'delta':delta, 'trajectory_delta':per,
              'raw_states':raw, 'positive_rel_trajectories':{a:sum(v['rel_l2_score']>0 for v in per[a].values()) for a in per},
              'evidence':'Real-only held-angle T; P/H have shared historical ancestors; original P absolute intervals; no official score forecast'}
    (out/'evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['raw_states','trajectory_delta']}), flush=True)


def main(angle):
    started = time.monotonic(); torch.set_num_threads(4)
    assert torch.cuda.device_count() == 1
    sys.path[:0] = [str(OLD), str(ROOT.parent/'track1_dedicated_sps_20260918')]
    from point_runtime_k1 import Runtime
    runtime = Runtime(OLD, angle)
    rows = runtime.ev.clean_rows(Path(runtime.plan['manifest']))
    out = ROOT/f'T_held{angle}'; out.mkdir(exist_ok=False)
    metric = module(ROOT.parent/'track1_final_only_20260918/train.py', 't_point_errors')
    candidate = fit(runtime, [r for r in rows if r['aoa'] != angle], metric.errors, out)
    evaluate(runtime, candidate, [r for r in rows if r['aoa'] == angle], out)
    (out/'complete.json').write_text(json.dumps({'seconds':time.monotonic()-started, 'fixed_steps':20000}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--angle', type=int, choices=[0,20], default=0)
    main(parser.parse_args().angle)
