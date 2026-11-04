"""Fixed direct-GT mean refinement; frozen C fluctuations, no teacher targets."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main(args):
    started = time.monotonic()
    plan = json.loads(args.plan.read_text())
    sys.path[:0] = [str(args.source/'code'), str(args.source/'code/training'), str(args.source/'code/evaluation')]
    import torch
    import train_b210_transfer_real4k as base
    import evaluate_b210_transfer as ev
    import train_b210_full_prefix_uq_refit as prefix
    from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
    from models.history_mean_refiner import install
    from train_b210_full_released_real_support import load_full_rows
    from fp32_ema_shadow import FP32EMAShadow

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    assert not args.output_dir.exists()
    assert ev.sha256(args.source/'code/models/history_mean_refiner.py') == plan['source_files']['code/models/history_mean_refiner.py']
    assert ev.sha256(__file__) == plan['source_files']['code/training/train_history_mean_refiner.py']
    asset = plan['assets'][str(args.held_aoa)]
    assert ev.sha256(asset['receipt']) == asset['receipt_sha256']
    q = json.loads(Path(asset['receipt']).read_text())
    point = q['point_receipt']
    assert point['held_aoa'] == args.held_aoa and not point['held_data_read']
    assert point['checkpoint_sha256'] == asset['point_sha256']
    assert ev.sha256(point['checkpoint_path']) == asset['point_sha256']
    payload = torch.load(point['checkpoint_path'], map_location='cpu', weights_only=False)
    config = dict(payload['model_config'])
    assert config.pop('architecture') == 'ensemble_unet'
    model = EnsembleUNet2d(EnsembleUNet2dConfig(**config)).float().cuda()
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    norm, norm_report = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
    assert norm_report['sha256'] == q['normalizer_sha256']
    dense = base.load_runtime()
    assert ev.sha256(plan['manifest']) == plan['manifest_sha256']
    rows = [r for r in ev.clean_rows(Path(plan['manifest'])) if r['aoa'] != args.held_aoa]
    assert [r['file'] for r in rows] == point['fit_files']
    audit = dense.PrefixAudit()
    trajectories, _ = load_full_rows(Path(plan['real_dir']), rows, 'fit', audit)
    assert set(audit.opened) == set(point['fit_files'])
    windows = dense.dense_windows(trajectories, 5)
    assert len(windows) == point['optimizer_windows']
    dataset = dense.ft12.npt1.ensemble_train.common.repro.WindowDataset(trajectories, windows)
    schedule = dense.smu1.paired_batch_schedule(range(len(windows)), windows, updates=800, seed=4)
    batches = [r['sample_indices'] for r in schedule['rows']]
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=batches, num_workers=0, pin_memory=True)
    history, truth = next(iter(loader))
    x, target = norm.preprocess(history.cuda(), truth.cuda())
    with torch.no_grad():
        before = model(x)
    install(model, SimpleNamespace(**payload['normalizer']))
    with torch.no_grad():
        initial_drift = float((model(x)-before).abs().max())
    assert initial_drift <= 2e-6
    model.mean_profile_refiner.requires_grad_(True)
    parameters = list(model.mean_profile_refiner.parameters())
    trainable = [k for k,v in model.named_parameters() if v.requires_grad]
    assert all(k.startswith('mean_profile_refiner.') for k in trainable)
    dense.ft12.npt1.ensemble_train.seed_everything(4)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800)
    ema = FP32EMAShadow(model.mean_profile_refiner, .99)
    args.output_dir.mkdir(parents=True)
    logs = []
    objective = dense.ft12.npt1.ensemble_train.common.accuracy_objective
    for step, (history, truth) in enumerate(loader, 1):
        assert time.monotonic()-started < plan['fold_seconds']
        x, target = norm.preprocess(history.cuda(non_blocking=True), truth.cuda(non_blocking=True))
        prediction = model(x)
        loss, _ = objective(prediction, target, norm, (.5, 0., .1), batch_mean_errors=True)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        if step == 1:
            assert float(model.mean_profile_refiner.history_projection.weight.grad.abs().sum()) > 0
        optimizer.step()
        ema.update_after_optimizer_step(model.mean_profile_refiner, optimizer_step=step)
        scheduler.step()
        if step == 1 or step % 100 == 0:
            row = dict(step=step, loss=float(loss), gradient=float(gradient), seconds=time.monotonic()-started)
            logs.append(row)
            print(json.dumps(row), flush=True)
    assert step == ema.updates == 800
    state = model.state_dict()
    assert all(torch.equal(state[k].cpu(), v) for k,v in payload['model_state_dict'].items()
               if not k.startswith('mean_profile_refiner.'))
    head = {k:v.cpu() for k,v in ema.materialize_model_state(model.mean_profile_refiner).items()}
    assert any(torch.count_nonzero(v) for k,v in head.items() if k.startswith('history_projection.'))
    endpoint = args.output_dir/'mean_ema99_u800.pth'
    torch.save(dict(schema='direct_history_mean_v1', state_dict=head, base_point_sha256=asset['point_sha256'],
                    plan_sha256=ev.sha256(args.plan), updates=800), endpoint)
    receipt = dict(schema='direct_history_mean_v1', checkpoint_path=str(endpoint), checkpoint_sha256=ev.sha256(endpoint),
                   base_point_receipt=point, base_uq_receipt=asset, held_aoa=args.held_aoa,
                   fit_files=point['fit_files'], optimizer_windows=len(windows), held_data_read=False,
                   normalizer_sha256=q['normalizer_sha256'], sample_stream_sha256=schedule['sha256'],
                   updates=800, seed=4, ema_decay=.99, lr=1e-4, weight_decay=0., batch_size=8,
                   trainable_parameters=sum(p.numel() for p in parameters), additional_parameters=43*104,
                   frozen_other_parameters_bitwise=True, initial_max_abs=initial_drift,
                   loss='final fused point relative-L2/MVPE scores, weights .5/0/.1, batch-mean errors',
                   teacher_used=False, source_sha256=ev.sha256(__file__), plan_sha256=ev.sha256(args.plan),
                   evidence=plan['evidence'], logs=logs, seconds=time.monotonic()-started)
    (args.output_dir/'point_endpoint.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k:receipt[k] for k in ['checkpoint_sha256','seconds','trainable_parameters','initial_max_abs']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['source','plan','output-dir']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--held-aoa', type=int, choices=[5,10], required=True)
    main(p.parse_args())
