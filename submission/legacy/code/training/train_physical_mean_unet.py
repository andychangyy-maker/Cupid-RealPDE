"""Fixed Sim10k -> Real4k mean-only network, leaving C fluctuations frozen."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def run(args):
    started = time.monotonic()
    plan = json.loads(args.plan.read_text())
    src = Path(plan['source_root'])
    sys.path[:0] = [str(src/'code'), str(src/'code/training'), str(src/'code/evaluation')]
    import torch
    import train_hybrid_fno2d as common
    import train_b210_transfer_real4k as base
    import train_b210_full_prefix_uq_refit as prefix
    import evaluate_b210_transfer as ev
    from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
    from models.physical_mean_unet import PhysicalMeanUNet, replace_mean
    from train_b210_full_released_real_support import load_full_rows
    from fp32_ema_shadow import FP32EMAShadow

    assert ev.sha256(__file__) == plan['source_files']['code/training/train_physical_mean_unet.py']
    assert ev.sha256(src/'code/models/physical_mean_unet.py') == plan['source_files']['code/models/physical_mean_unet.py']
    assert not args.output.exists()
    args.output.mkdir()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    dense = base.load_runtime()
    native = dense.ft12.npt1.ensemble_train
    seed, updates, lr, cosine = (210, 10000, 3e-4, 10000) if args.phase == 'sim' else (4, 4000, 2e-4, 12000)
    native.seed_everything(seed)
    model = PhysicalMeanUNet().float().cuda()
    control = None
    if args.phase == 'sim':
        trajectories, windows, manifest = common.scratch.load_simulations(Path(plan['sim_dir']), 0, 20, True)
        assert (len(trajectories), len(windows)) == (100, 4800)
        dataset = common.repro.WindowDataset([t[..., :2] for t in trajectories], windows)
        fit_files = [r['file'] for r in manifest]
        (args.output/'sim_manifest.json').write_text(json.dumps(manifest, indent=2))
        initialization = 'random'
        base_point = None
    else:
        sim_receipt = json.loads((args.plan.parent/'sim/endpoint.json').read_text())
        assert sim_receipt['updates'] == 10000 and sim_receipt['phase'] == 'sim'
        assert ev.sha256(sim_receipt['checkpoint_path']) == sim_receipt['checkpoint_sha256']
        initial = torch.load(sim_receipt['checkpoint_path'], map_location='cpu', weights_only=False)
        model.load_state_dict(initial['state_dict'], strict=True)
        initialization = sim_receipt['checkpoint_sha256']
        asset = plan['assets'][str(args.held_aoa)]
        assert ev.sha256(asset['receipt']) == asset['receipt_sha256']
        q = json.loads(Path(asset['receipt']).read_text())
        base_point = q['point_receipt']
        assert base_point['held_aoa'] == args.held_aoa and not base_point['held_data_read']
        assert ev.sha256(base_point['checkpoint_path']) == asset['point_sha256']
        payload = torch.load(base_point['checkpoint_path'], map_location='cpu', weights_only=False)
        cfg = dict(payload['model_config']); assert cfg.pop('architecture') == 'ensemble_unet'
        control = EnsembleUNet2d(EnsembleUNet2dConfig(**cfg)).float().cuda()
        control.load_state_dict(payload['model_state_dict'], strict=True)
        control.eval().requires_grad_(False)
        norm, norm_report = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
        assert norm_report['sha256'] == q['normalizer_sha256']
        assert ev.sha256(plan['manifest']) == plan['manifest_sha256']
        rows = [r for r in ev.clean_rows(Path(plan['manifest'])) if r['aoa'] != args.held_aoa]
        assert [r['file'] for r in rows] == base_point['fit_files']
        audit = dense.PrefixAudit()
        trajectories, _ = load_full_rows(Path(plan['real_dir']), rows, 'fit', audit)
        fit_files = base_point['fit_files']
        assert set(audit.opened) == set(fit_files)
        windows = dense.dense_windows(trajectories, 5)
        assert len(windows) == base_point['optimizer_windows']
        dataset = common.repro.WindowDataset(trajectories, windows)
        del payload, initial

    schedule = dense.smu1.paired_batch_schedule(range(len(windows)), windows, updates=updates, seed=seed)
    batches = [row['sample_indices'] for row in schedule['rows']]
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=batches, num_workers=0, pin_memory=True)
    physical = SimpleNamespace(mean_target=torch.zeros(2, device='cuda'), std_target=torch.ones(2, device='cuda'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine)
    ema = FP32EMAShadow(model, .999)
    native.seed_everything(seed)
    model.train()
    logs = []
    for step, (history, truth) in enumerate(loader, 1):
        assert time.monotonic()-started < plan['phase_seconds'][args.phase]
        history, truth = history.cuda(non_blocking=True), truth.cuda(non_blocking=True)
        mean = model(history)
        _, scale = model.scale(history)
        mse = ((mean-truth[..., :2].mean(1))/scale[:, 0]).square().mean()
        if step <= 1500:
            loss = mse
        else:
            if control is None:
                forecast, target = mean[:, None], truth[..., :2].mean(1, keepdim=True)
            else:
                with torch.no_grad():
                    x, _ = norm.preprocess(history, truth)
                    point = norm.postprocess_prediction(control(x).float())
                forecast = replace_mean(point, mean)[..., :2]
                target = truth[..., :2]
            accuracy, _ = common.accuracy_objective(forecast, target, physical, (.5, 0., .1), batch_mean_errors=True)
            loss = accuracy + .25*mse
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if step == 1:
            assert float(model.output.weight.grad.abs().sum()) > 0
        optimizer.step()
        ema.update_after_optimizer_step(model, optimizer_step=step)
        scheduler.step()
        if step in (1, 1501) or step % 1000 == 0:
            row = dict(step=step, loss=float(loss), mean_mse=float(mse), gradient=float(gradient), seconds=time.monotonic()-started)
            logs.append(row); print(json.dumps(row), flush=True)
    assert step == ema.updates == updates
    assert control is None or all(p.grad is None for p in control.parameters())
    endpoint = args.output/f'mean_ema999_u{updates}.pth'
    torch.save(dict(schema='physical_mean_unet_v1', state_dict={k:v.cpu() for k,v in ema.materialize_model_state(model).items()},
                    phase=args.phase, updates=updates, initialization=initialization, plan_sha256=ev.sha256(args.plan)), endpoint)
    receipt = dict(phase=args.phase, updates=updates, seed=seed, initialization=initialization,
                   checkpoint_path=str(endpoint), checkpoint_sha256=ev.sha256(endpoint),
                   plan_sha256=ev.sha256(args.plan), parameters=sum(p.numel() for p in model.parameters()),
                   held_aoa=args.held_aoa, fit_files=fit_files, optimizer_windows=len(windows),
                   sample_stream_sha256=schedule['sha256'], base_point_receipt=base_point,
                   held_data_read=False, uq_trained=False, teacher_used=False, existing_point_updated=False,
                   normalizer='per-history physical UV mean/common RMS only; C retains its original statistics',
                   logs=logs, seconds=time.monotonic()-started)
    (args.output/'endpoint.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k:receipt[k] for k in ['phase','updates','checkpoint_sha256','parameters','seconds']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['sim','real'], required=True)
    p.add_argument('--held-aoa', type=int, choices=[5,10])
    run(p.parse_args())
