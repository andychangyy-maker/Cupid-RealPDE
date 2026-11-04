"""Fixed Real4k phase training against current held-angle C endpoints."""
import argparse
import json
from pathlib import Path
import sys
import time


def run(args):
    started = time.monotonic()
    plan = json.loads(args.plan.read_text())
    src = Path(plan['source_root'])
    sys.path[:0] = [str(src/'code'), str(src/'code/training'), str(src/'code/evaluation')]
    import torch
    import train_b210_transfer_real4k as base
    import train_b210_full_prefix_uq_refit as prefix
    import train_b210_causal_phase_direction_real4k as phase
    import evaluate_b210_transfer as ev
    from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
    from models.causal_phase_direction import CausalPhaseDirectionAdapter, CausalPhaseDirectionConfig
    from train_b210_full_released_real_support import load_full_rows
    from fp32_ema_shadow import FP32EMAShadow

    for name, digest in plan['source_files'].items(): assert ev.sha256(src/name) == digest
    assert not args.output.exists()
    args.output.mkdir()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    dense = base.load_runtime()
    asset = plan['assets'][str(args.held_aoa)]
    assert ev.sha256(asset['receipt']) == asset['receipt_sha256']
    q = json.loads(Path(asset['receipt']).read_text())
    point = q['point_receipt']
    assert point['held_aoa'] == args.held_aoa and not point['held_data_read']
    assert ev.sha256(point['checkpoint_path']) == asset['point_sha256']
    payload = torch.load(point['checkpoint_path'], map_location='cpu', weights_only=False)
    cfg = dict(payload['model_config']); assert cfg.pop('architecture') == 'ensemble_unet'
    control = EnsembleUNet2d(EnsembleUNet2dConfig(**cfg)).float().cuda()
    control.load_state_dict(payload['model_state_dict'], strict=True)
    control.eval().requires_grad_(False)
    norm, norm_report = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
    assert norm_report['sha256'] == q['normalizer_sha256']
    del payload
    assert ev.sha256(plan['manifest']) == plan['manifest_sha256']
    rows = [r for r in ev.clean_rows(Path(plan['manifest'])) if r['aoa'] != args.held_aoa]
    assert [r['file'] for r in rows] == point['fit_files']
    audit = dense.PrefixAudit()
    trajectories, _ = load_full_rows(Path(plan['real_dir']), rows, 'fit', audit)
    assert set(audit.opened) == set(point['fit_files'])
    windows = dense.dense_windows(trajectories, 5)
    assert len(windows) == point['optimizer_windows']
    dataset = phase.mature_real.common.repro.WindowDataset(trajectories, windows)
    schedule = dense.smu1.paired_batch_schedule(range(len(windows)), windows, updates=4000, seed=4)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=[row['sample_indices'] for row in schedule['rows']],
                                       num_workers=0, pin_memory=True)
    phase.real4k.seed_real_recipe(torch.device('cuda:0'))
    model = CausalPhaseDirectionAdapter(CausalPhaseDirectionConfig()).float().cuda()
    assert sum(p.numel() for p in model.parameters()) == 41667
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12000)
    ema = FP32EMAShadow(model, .999)
    phase.real4k.seed_real_recipe(torch.device('cuda:0'))
    trace = []
    for step, (history, truth) in enumerate(loader, 1):
        assert time.monotonic()-started < plan['fold_seconds']
        history, truth = history.cuda(non_blocking=True), truth.cuda(non_blocking=True)
        adapter_history = phase.real4k.augment_real_history(history, norm)
        x, target = norm.preprocess(history, truth)
        with torch.no_grad():
            original = control(x).float()
        forecast = model(adapter_history, original)
        if step == 1: assert torch.equal(forecast, original)
        loss, details = phase.phase_objective(forecast, target, norm)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if step == 1: assert float(model.direction_head.weight.grad.abs().sum()) > 0
        if step == 2:
            assert float(model.frame_encoder.stem.weight.grad.abs().sum()) > 0
            assert float(model.gate_head.weight.grad.abs().sum()) > 0
        optimizer.step()
        ema.update_after_optimizer_step(model, optimizer_step=step)
        scheduler.step()
        if step in (1,2) or step % 1000 == 0:
            row = dict(step=step, loss=float(loss), gradient=float(gradient), seconds=time.monotonic()-started,
                       cosine=float(details['radius_product_weighted_direction_cosine']))
            trace.append(row); print(json.dumps(row), flush=True)
    assert step == ema.updates == 4000 and all(p.grad is None for p in control.parameters())
    endpoint = args.output/'phase_ema999_u4000.pth'
    torch.save(dict(schema='current_c_phase_v1', state_dict={k:v.cpu() for k,v in ema.materialize_model_state(model).items()},
                    updates=4000, model_config=model.architecture_config(), plan_sha256=ev.sha256(args.plan)), endpoint)
    receipt = dict(updates=4000, seed=4, initialization='random; no legacy adapter weights used',
                   checkpoint_path=str(endpoint), checkpoint_sha256=ev.sha256(endpoint),
                   plan_sha256=ev.sha256(args.plan), parameters=41667, held_aoa=args.held_aoa,
                   fit_files=point['fit_files'], optimizer_windows=len(windows), sample_stream_sha256=schedule['sha256'],
                   base_point_receipt=point, normalizer_sha256=norm_report['sha256'], held_data_read=False,
                   uq_trained=False, existing_point_updated=False, original_output_identity_at_step_zero=True,
                   trace=trace, seconds=time.monotonic()-started)
    (args.output/'endpoint.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k:receipt[k] for k in ['updates','checkpoint_sha256','parameters','seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--held-aoa', type=int, choices=[5,10], required=True)
    run(parser.parse_args())
