"""One fixed mature-UQ continuation contrast, with untouched point deployment state."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

SOURCE = Path('/root/autodl-tmp/q4_stage_source')
for folder in ('code', 'code/training', 'code/evaluation', 'code/submissions'):
    sys.path.insert(0, str(SOURCE / folder))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def direct_loss(prediction, extents, target, normalizer, common):
    import torch
    physical = common.to_physical_target(prediction, normalizer)[..., :2].detach()
    truth = common.to_physical_target(target, normalizer)[..., :2]
    residual = truth - physical
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    lower, upper = (v * scale for v in extents)
    tau = lower.new_tensor([.01075, .00550]).view(1, 1, 1, 1, 2) * .035
    inside = torch.sigmoid((residual+lower)/tau) * torch.sigmoid((upper-residual)/tau)
    factor = common.sample_sps_factor(physical, truth).detach().view(-1, 1, 1, 1, 1)
    scored = truth != 0
    return -2 * (factor * inside * torch.exp(-(lower+upper)/common.SPS_SIGMA))[scored].mean()


def train(args):
    import torch
    from torch.utils.data import DataLoader
    import train_b210_transfer_uq as transfer
    import train_b210_small4k_uq_release as native
    import train_b210_full_prefix_uq_refit as prefix
    from b210_stage_fold import select_rows
    from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
    from build_track1_blended_ensemble_unet2d_package import compact_state_dict
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    receipt = json.loads((args.base/'uq/uq_endpoint.json').read_text())
    checkpoint = Path(receipt['checkpoint_path'])
    assert sha(checkpoint) == receipt['checkpoint_sha256']
    assert receipt['held_aoa'] == args.held
    payload = torch.load(checkpoint, map_location='cpu')
    config = dict(payload['model_config']); assert config.pop('architecture') == 'ensemble_unet'
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**config)).float()
    model.load_state_dict(payload['model_state_dict'], strict=True)
    frozen = {k:v.clone() for k,v in model.state_dict().items() if not k.startswith(native.UQ_PREFIXES)}
    buffers = {k:v.clone() for k,v in model.named_buffers()}
    normalizer, _ = prefix.normalizer_from_point(payload, torch.device('cuda'))
    rows, _ = native.dr1.load_manifest(args.manifest, native.dr1.FORMAL_MANIFEST_SHA256)
    rows = select_rows(rows, args.held)
    assert [r['file'] for r in rows] == receipt['point_receipt']['fit_files']
    trajectories, _, windows, audit = transfer.load_uq_rows(args.real_dir, rows, False)
    tagged = native.ensemble_train.TrajectoryTaggedDataset(
        native.ensemble_train.common.repro.WindowDataset(trajectories, windows))
    native.uq_train.seed_everything(18)
    loader = DataLoader(tagged, batch_size=8, shuffle=True, num_workers=0,
                        pin_memory=True, generator=torch.Generator().manual_seed(18))
    batches = native.ensemble_train.common.repro.cycling(loader)
    model.cuda()
    parameters = native.configure_uq_only(model)
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800)
    objective = native.sharp2_args()
    stream = hashlib.sha256(); begin = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    for step in range(1, 801):
        assert time.monotonic()-begin < 900, 'arm budget expired'
        native.set_uq_training_mode(model)
        history, target, _, indices = next(batches)
        stream.update(indices.numpy().astype('<i8', copy=False).tobytes())
        x, y = normalizer.preprocess(history.cuda(), target.cuda())
        optimizer.zero_grad(set_to_none=True)
        prediction, quantiles, extents, logits = native.uq_train.quantile_forward(model, x)
        if args.arm == 'control':
            loss, _ = native.uq_train.uq_objective(objective, model, prediction, quantiles,
                y, normalizer, interval_extents=extents, coverage_logits=logits)
        else:
            loss = direct_loss(prediction, extents, y, normalizer, native.uq_train.common)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1)
        assert torch.isfinite(norm) and (step != 1 or norm > 0)
        optimizer.step(); scheduler.step()
        if step % 200 == 0:
            print(json.dumps({'arm':args.arm,'held':args.held,'step':step,
                'loss':float(loss),'seconds':time.monotonic()-begin}), flush=True)
    state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    assert all(torch.equal(v,state[k]) for k,v in frozen.items())
    assert all(torch.equal(v,state[k]) for k,v in buffers.items())
    uq = {k:v for k,v in state.items() if k.startswith(native.UQ_PREFIXES)}
    torch.save(uq, args.output/'uq_only.pth')
    runtime = args.output/'runtime'
    shutil.copytree(args.base/'runtime', runtime)
    deploy = torch.load(runtime/'ensemble_unet2d.pth', map_location='cpu')
    compact, _ = compact_state_dict(uq, 1024)
    assert set(compact) == {k for k in deploy['quantized_state_dict'] if k.startswith(native.UQ_PREFIXES)}
    deploy['quantized_state_dict'].update(compact)
    torch.save(deploy, runtime/'ensemble_unet2d.pth')
    result = {'arm':args.arm,'held':args.held,'updates':800,'source_checkpoint_sha256':sha(checkpoint),
        'source_receipt_sha256':sha(args.base/'uq/uq_endpoint.json'),'sample_stream_sha256':stream.hexdigest(),
        'frozen_point_and_all_buffers_bitwise':True,'audit':audit,'seconds':time.monotonic()-begin,
        'uq_sha256':sha(args.output/'uq_only.pth'),'runtime_sha256':sha(runtime/'ensemble_unet2d.pth')}
    (args.output/'receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result), flush=True)


def score(args):
    import numpy as np
    import torch
    import evaluate_b210_transfer as ev
    torch.set_num_threads(4)
    ev.TIME_SCORE = 92.03266035232576
    scorer = ev.import_file(Path('/root/autodl-tmp/time_audit_919767_20260909/source/scoring.py'),'direct_scorer')
    x = np.load('/root/autodl-tmp/spuq2_a800/artifacts/clean716_input_cache.npy',mmap_mode='r')
    with np.load('/root/autodl-tmp/real_time_release_20260908/time_replay_truth_20260907.npz') as data:
        y, labels = data['truth'], data['labels'].astype(str)
    selected = np.array([int(Path(s).stem.split('_')[-1]) == args.held for s in labels])
    x, y, labels = x[selected], y[selected], labels[selected]
    sys.path.insert(0,str(args.runtime))
    submission = ev.import_file(args.runtime/'submission.py','direct_submission')
    output = submission.predict(x.copy())
    assert set(output) == {'prediction','lower','upper'}
    assert all(v.shape == y.shape and v.dtype == np.float32 and np.isfinite(v).all() and
               np.all(v[...,2] == 0) for v in output.values())
    assert np.all(output['lower'] <= output['upper'])
    states = {s:ev.metric_state(scorer,output['prediction'][labels==s],y[labels==s],
        output['lower'][labels==s],output['upper'][labels==s]) for s in dict.fromkeys(labels)}
    merged = ev.merge_states(list(states.values()))
    args.output.write_text(json.dumps({'held':args.held,'scores':ev.finalize(scorer,merged),
        'states':states,'merged':merged,'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
        'truth_sha256':hashlib.sha256(y.tobytes()).hexdigest(),
        'prediction_sha256':hashlib.sha256(output['prediction'].tobytes()).hexdigest()},indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base',type=Path)
    p.add_argument('--held',type=int,required=True,choices=(0,5,10,15,20))
    p.add_argument('--arm',choices=('control','direct'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runtime',type=Path)
    p.add_argument('--manifest',type=Path,default='/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json')
    p.add_argument('--real-dir',type=Path,default='/root/autodl-tmp/realpde/data/real')
    a = p.parse_args()
    score(a) if a.runtime else train(a)
