"""Fixed 800-update distillation of the confirmed PC3D teacher into native B210."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import train_b210_held_residual_uq as held

torch, dense, _, prefix = held.runtime()
import numpy as np
import evaluate_b210_transfer as ev
import train_heterogeneous_pc3d_real as pc
from fp32_ema_shadow import FP32EMAShadow, tensor_state_sha256
from models.ensemble_unet2d import EnsembleUNet2d
from screen_cross_source_fluct_teacher import combine
from train_b210_full_released_real_support import load_full_rows

C_SHA = {20: '612989b56b00c1d7f36b660bca88b99c70efbdf0f65c51381a7ce94683159522',
         0: '6e14d57277ce76a12208114c74a1216f9772d8f1be01dcb1cf8448512703db61'}
PC_SHA = {20: '38086b3d40ee91919aa9af8ae4ad9d961612cae30cfb97c3747132612c3de472',
          0: '1e89112c8a4a46c127a7bca3a4c6ab8220633d89d9c612e17ac9eacd22182695'}
UPDATES = 800


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_control(args):
    assert held.sha256(args.manifest) == pc.MANIFEST_SHA
    fold = args.confirm_root / f'held{args.held_aoa}'
    receipt = json.loads((fold / 'control/point_endpoint.json').read_text())
    assert receipt['checkpoint_sha256'] == C_SHA[args.held_aoa]
    assert held.sha256(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
    assert receipt['updates'] == 4000 and receipt['held_aoa'] == args.held_aoa
    payload = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)
    model = EnsembleUNet2d(dense.ft12.validate_b210_model_config(payload['model_config'])).float()
    model.load_state_dict(payload['model_state_dict'], strict=True)
    norm = prefix.normalizer_from_point(payload, torch.device('cuda:0'))[0]
    return fold, receipt, payload, model.cuda().eval(), norm


def physical(model, x, norm):
    return norm.postprocess_prediction(model(x).float())


def train(args):
    started = time.perf_counter()
    assert not args.output_dir.exists() and len(args.source_commit) == 40
    fold, control_receipt, payload, control, norm = load_control(args)
    control.requires_grad_(False)
    pc_receipt = json.loads((fold / 'candidate/point_endpoint.json').read_text())
    assert pc_receipt['endpoint']['file_sha256'] == PC_SHA[args.held_aoa]
    assert pc_receipt['source']['baseline_point_sha256'] == C_SHA[args.held_aoa]
    assert pc_receipt['source']['fit_files'] == control_receipt['fit_files']
    pc_payload = torch.load(pc_receipt['endpoint']['path'], map_location='cpu', weights_only=False)
    assert all(torch.equal(value.cpu(), getattr(norm, name).cpu())
               for name, value in pc_payload['normalizer'].items())
    del pc_payload
    pc_model = pc.stage_r.load_ema_endpoint(pc_receipt['endpoint'], torch.device('cuda:0')).eval()
    pc_model.requires_grad_(False)
    student = copy.deepcopy(control).requires_grad_(True).eval()
    rows = [r for r in ev.clean_rows(args.manifest) if r['aoa'] != args.held_aoa]
    assert [r['file'] for r in rows] == control_receipt['fit_files']
    audit = dense.PrefixAudit()
    trajectories, _ = load_full_rows(args.real_dir, rows, 'fit', audit)
    windows = dense.dense_windows(trajectories, 5)
    assert len(windows) == control_receipt['optimizer_windows']
    assert set(audit.opened) == set(control_receipt['fit_files'])
    dataset = dense.ft12.npt1.ensemble_train.common.repro.WindowDataset(trajectories, windows)
    schedule = dense.smu1.paired_batch_schedule(range(len(windows)), windows, updates=UPDATES, seed=4)
    batches = [row['sample_indices'] for row in schedule['rows']]
    assert len(batches) == UPDATES and all(len(b) == 8 for b in batches)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=batches, num_workers=0, pin_memory=True)
    dense.ft12.npt1.ensemble_train.seed_everything(4)
    args.output_dir.mkdir(parents=True)
    contract = dict(schema='pc3d_student_800_v1', source_commit=args.source_commit,
                    held_aoa=args.held_aoa, control_sha256=C_SHA[args.held_aoa],
                    pc3d_sha256=PC_SHA[args.held_aoa], fit_files=control_receipt['fit_files'],
                    optimizer_windows=len(windows), manifest_sha256=pc.MANIFEST_SHA,
                    normalizer_sha256=payload['normalizer_sha256'],
                    sample_indices_sha256=hashlib.sha256(json.dumps(batches).encode()).hexdigest(),
                    updates=UPDATES, batch_size=8, lr=1e-5, weight_decay=0., ema_decay=.99,
                    dropout=False, loss='per_window_physical_uv_mse_over_teacher_fluctuation_mse_floor1e-6',
                    teacher='unchanged screen_cross_source_fluct_teacher.combine', held_data_read=False)
    save_json(args.output_dir / 'execution_contract.json', contract)
    parameters = dict(student.named_parameters())
    ema = FP32EMAShadow(student, .99)
    optimizer = scheduler = None
    logs = []
    for update, (history, _truth) in enumerate(loader, 1):
        if time.perf_counter() - started >= 1200:
            raise TimeoutError('STOP_PC3D_STUDENT_TRAIN_BUDGET')
        history = history.cuda(non_blocking=True)
        x, _ = norm.preprocess(history, history)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            first = physical(control, x, norm)
            second = physical(pc_model, x, norm)
            first[..., 2] = 0
            second[..., 2] = 0
            target_np, _ = combine(first.cpu().numpy(), second.cpu().numpy())
            target = torch.from_numpy(target_np[..., :2].copy()).cuda()
            denominator = (target - target.mean(1, keepdim=True)).square().mean((1, 2, 3, 4)).clamp_min(1e-6)
        prediction = physical(student, x, norm)[..., :2]
        if update == 1:
            assert torch.equal(prediction.detach(), first[..., :2]), 'student start differs from frozen C'
        loss = ((prediction - target).square().mean((1, 2, 3, 4)) / denominator).mean()
        assert torch.isfinite(loss), 'nonfinite distillation loss'
        loss.backward()
        if update == 1:
            active = [name for name, value in parameters.items() if value.grad is not None]
            assert any(name.startswith('encoder.') for name in active)
            assert any(name.startswith('member_point_heads.') for name in active)
            inactive = [name for name in parameters if name not in active]
            for name in inactive:
                parameters[name].requires_grad_(False)
            optimizer = torch.optim.AdamW([parameters[name] for name in active], lr=1e-5, weight_decay=0.)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)
            save_json(args.output_dir / 'trainable_parameters.json', dict(active=active, inactive=inactive))
        gradient = torch.nn.utils.clip_grad_norm_([parameters[name] for name in active], 1., error_if_nonfinite=True)
        optimizer.step()
        ema.update_after_optimizer_step(student, optimizer_step=update)
        scheduler.step()
        if update == 1 or update % 100 == 0:
            row = dict(update=update, loss=float(loss), gradient_norm=float(gradient),
                       elapsed_seconds=time.perf_counter()-started)
            logs.append(row)
            print(json.dumps(row), flush=True)
    assert update == UPDATES and ema.updates == UPDATES
    state = ema.materialize_model_state(student)
    for name in inactive:
        state[name] = payload['model_state_dict'][name].clone()
    assert all(torch.equal(state[name], payload['model_state_dict'][name]) for name in inactive)
    endpoint = args.output_dir / 'ema99_u800.pth'
    checkpoint = dict(schema='pc3d_student_ema800_v1', model_state_dict=state,
                      model_config=payload['model_config'], normalizer=payload['normalizer'],
                      normalizer_sha256=payload['normalizer_sha256'], source=contract,
                      iteration=UPDATES, ema_decay=.99, parent_checkpoint_sha256=C_SHA[args.held_aoa])
    torch.save(checkpoint, endpoint)
    receipt = dict(contract, checkpoint_path=str(endpoint.resolve()), checkpoint_sha256=held.sha256(endpoint),
                   model_state_sha256=tensor_state_sha256(state), logs=logs,
                   elapsed_seconds=time.perf_counter()-started, cuda_peak_bytes=torch.cuda.max_memory_allocated(),
                   decision='PC3D_STUDENT_EMA800_SEALED_NOT_SCORED')
    assert receipt['elapsed_seconds'] < 1200
    save_json(args.output_dir / 'point_endpoint.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('decision', 'checkpoint_sha256', 'elapsed_seconds')}), flush=True)


def score(args):
    fold, _, _, control, norm = load_control(args)
    receipt = json.loads((args.output_dir / 'point_endpoint.json').read_text())
    assert receipt['decision'] == 'PC3D_STUDENT_EMA800_SEALED_NOT_SCORED'
    assert held.sha256(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
    payload = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)
    assert payload['source']['held_aoa'] == args.held_aoa and not payload['source']['held_data_read']
    assert payload['normalizer_sha256'] == receipt['normalizer_sha256']
    student = copy.deepcopy(control)
    student.load_state_dict(payload['model_state_dict'], strict=True)
    reference = json.loads((fold / 'RESULT.json').read_text())
    rows = [r for r in ev.clean_rows(args.manifest) if r['aoa'] == args.held_aoa]
    scorer = ev.import_file(ev.SCORER, 'pc3d_student_scorer')
    functions = (lambda p, y: scorer.rel_l2_per_sample(p, y, 2),
                 lambda p, y: scorer.tke_rel_l2_per_sample(p, y, 2), scorer.mvpe_rel_l2_per_sample)
    errors = {label: {r['file']: [] for r in rows} for label in ('control', 'student')}
    hashes = {k: hashlib.sha256() for k in ('input', 'truth')}
    with torch.inference_mode():
        for history, truth, files in ev.tail_batches(rows, args.real_dir):
            hashes['input'].update(history.tobytes())
            hashes['truth'].update(truth.tobytes())
            history = torch.from_numpy(history).cuda()
            x, _ = norm.preprocess(history, history)
            for label, model in (('control', control), ('student', student)):
                prediction = physical(model, x, norm).cpu().numpy()
                prediction[..., 2] = 0
                assert np.isfinite(prediction).all()
                values = np.stack([fn(prediction, truth) for fn in functions], axis=-1)
                for file, value in zip(files, values):
                    errors[label][file].append(value.tolist())
    assert {k: v.hexdigest() for k, v in hashes.items()} == reference['hashes']
    def metrics(values):
        return dict(zip(ev.METRICS[:3], (100/(1+.5*np.asarray(values, dtype=np.float64).mean(0))).tolist()))
    def contribution(point):
        return ev.fixed_time_final({**point, 'sps_score': 43.2953289552})
    results = {label: dict(point_metrics=metrics([v for values in files.values() for v in values]),
                           trajectory_metrics={f: metrics(values) for f, values in files.items()}, errors=files)
               for label, files in errors.items()}
    assert max(abs(results['control']['point_metrics'][k]-reference['results']['control']['point_metrics'][k])
               for k in ev.METRICS[:3]) <= 1e-5, 'frozen C replay mismatch'
    changes = {k: results['student']['point_metrics'][k]-results['control']['point_metrics'][k] for k in ev.METRICS[:3]}
    delta = contribution(results['student']['point_metrics'])-contribution(results['control']['point_metrics'])
    trajectory = {f: contribution(v)-contribution(results['control']['trajectory_metrics'][f])
                  for f, v in results['student']['trajectory_metrics'].items()}
    positive = sum(v >= 0 for v in trajectory.values())
    target = .5 * reference['point_fitted_delta']
    passed = delta >= target and min(changes.values()) >= -.03 and positive/len(rows) >= .60
    result = dict(decision=('PASS_' if passed else 'STOP_')+f'PC3D_STUDENT_HELD{args.held_aoa}',
                  held_aoa=args.held_aoa, point_fitted_delta=delta, minimum_point_delta=target,
                  teacher_point_delta=reference['point_fitted_delta'], changes=changes,
                  nonnegative_trajectories=positive, trajectory_count=len(rows), trajectory_deltas=trajectory,
                  results=results, hashes={k: v.hexdigest() for k, v in hashes.items()},
                  endpoint_sha256=receipt['checkpoint_sha256'], source_commit=args.source_commit,
                  scope='Real-stage reused angle holdout; native point architecture; no matched UQ or official gain.')
    save_json(args.output_dir / 'RESULT.json', result)
    print(json.dumps({k: result[k] for k in ('decision', 'point_fitted_delta', 'changes', 'nonnegative_trajectories')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('train', 'score'), required=True)
    parser.add_argument('--held-aoa', type=int, choices=(20, 0), required=True)
    parser.add_argument('--source-commit', required=True)
    for name in ('confirm-root', 'output-dir', 'manifest', 'real-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    ev.TIME_SCORE = 92.0326603523
    (train if args.phase == 'train' else score)(args)
