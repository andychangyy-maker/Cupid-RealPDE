"""One fixed endpoint on an existing stage-held point; no held-driven selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'training'), str(ROOT / 'evaluation')]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--real-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    started = time.monotonic()
    deadline = started + 1800
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import train_b210_small4k_uq_release as native
    from train_b210_full_prefix_uq_refit import normalizer_from_point
    from train_b210_transfer_uq import load_uq_rows
    import evaluate_b210_transfer as ev
    from metric_aligned_point_objective import metric_aligned_point_errors
    from models.ensemble_unet2d import EnsembleUNet2dConfig
    from models.anchored_factorized_transport_complement import AnchoredFactorizedTransportComplementUNet2d

    torch.set_num_threads(4)
    native.uq_train.seed_everything(18)
    torch.set_float32_matmul_precision('high')
    receipt = json.loads(a.receipt.read_text())
    held = receipt['held_aoa']
    expected = {10: '888739844bc401bc0ffbdb3edc7216512e8c5a2abc9d0b898a369c08be304b67',
                0: 'e58652449eb7a223c8b1ca9ae7dd8affe0cc3d7fddcd7b2dadd86bd80df8b38d'}
    assert sha(receipt['checkpoint_path']) == receipt['checkpoint_sha256'] == expected[held]
    point = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)
    cfg = dict(point['model_config']); assert cfg.pop('architecture') == 'ensemble_unet'
    model = AnchoredFactorizedTransportComplementUNet2d(EnsembleUNet2dConfig(**cfg))
    model.load_legacy_state_dict(point['model_state_dict'])
    frozen = native.dr1.model_state_sha256(model.backbone_state_dict())
    assert frozen == native.dr1.model_state_sha256(point['model_state_dict'])
    del point['model_state_dict']
    for param in model.base_parameters():
        param.requires_grad_(False)
    roles = list(model.role_parameters())
    model.to('cuda').eval()
    normalizer, normalizer_report = normalizer_from_point(point, torch.device('cuda'))
    rows = ev.clean_rows(a.manifest)
    fit = [r for r in rows if r['aoa'] != held]
    assert [r['file'] for r in fit] == receipt['fit_files']
    trajectories, metadata, windows, access = load_uq_rows(a.real_dir, fit, True)
    dataset = native.ensemble_train.common.repro.WindowDataset(trajectories, windows)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True,
                        generator=torch.Generator().manual_seed(18))
    batches = native.ensemble_train.common.repro.cycling(loader)
    optimizer = torch.optim.AdamW(roles, lr=1e-4, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)
    train_start = time.monotonic()
    for step in range(1, 2001):
        assert time.monotonic() < deadline, 'STOP_BUDGET'
        batch = next(batches)
        history, target = batch[:2]
        x, _ = normalizer.preprocess(history.cuda(), target.cuda())
        fields = model.forward_factorized_fields(x)
        if step == 1:
            assert torch.equal(fields.base_prediction, fields.candidate_prediction), 'STOP_INITIAL_IDENTITY'
            (a.output / 'initial.json').write_text(json.dumps({
                'held': held, 'point_sha256': expected[held], 'base_state_sha256': frozen,
                'role_parameters': sum(v.numel() for v in roles), 'fit_windows': len(windows),
                'access': access, 'normalizer': normalizer_report, 'initial_bitwise': True}, indent=2))
        physical = normalizer.postprocess_prediction(fields.candidate_prediction.float())
        errors = metric_aligned_point_errors(physical, target.cuda())
        means = torch.stack([errors.rel_l2.mean(), errors.tke.mean(), errors.mvpe.mean()])
        loss = 1 - ((1 / (1 + .5 * means)) * means.new_tensor([50, 10, 10])).sum() / 70
        assert bool(torch.isfinite(loss)), 'STOP_NONFINITE'
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert all(v.grad is None for v in model.base_parameters()), 'STOP_BASE_GRADIENT'
        norm = torch.nn.utils.clip_grad_norm_(roles, 1.)
        assert bool(torch.isfinite(norm)), 'STOP_NONFINITE_GRADIENT'
        if step == 1:
            assert float(norm) > 0
        optimizer.step(); schedule.step()
        if step == 1 or step % 200 == 0:
            print(json.dumps({'step': step, 'loss': float(loss), 'gradient_norm': float(norm),
                              'seconds': time.monotonic()-train_start}), flush=True)
    assert native.dr1.model_state_sha256(model.backbone_state_dict()) == frozen, 'STOP_BASE_STATE'
    endpoint = a.output / 'role_raw2000.pth'
    torch.save({k: v.cpu() for k, v in model.state_dict().items()
                if k.startswith(model.ROLE_PREFIX)}, endpoint)
    seal = {'role_sha256': sha(endpoint), 'point_sha256': expected[held], 'base_frozen': True,
            'held_read_before_seal': False, 'train_seconds': time.monotonic()-train_start}
    (a.output / 'seal.json').write_text(json.dumps(seal, indent=2))
    del trajectories, dataset, loader, batches, optimizer, fields, physical, loss

    # Open held HDF5 only after sealing. Official scorer handles per-sample errors.
    official = ev.import_file(ev.SCORER, 'frozen_probe_official')
    sums = {'base': {}, 'candidate': {}}
    with torch.inference_mode():
        for inputs, truth, labels in ev.tail_batches([r for r in rows if r['aoa'] == held], a.real_dir):
            assert time.monotonic() < deadline, 'STOP_EVALUATION_BUDGET'
            x, _ = normalizer.preprocess(torch.from_numpy(inputs).cuda(), torch.from_numpy(truth).cuda())
            f = model.forward_factorized_fields(x)
            for arm, value in [('base', f.base_prediction), ('candidate', f.candidate_prediction)]:
                pred = normalizer.postprocess_prediction(value).float().cpu().numpy()
                pred[..., 2] = 0
                err = np.stack([official.rel_l2_per_sample(pred, truth, 2),
                                official.tke_rel_l2_per_sample(pred, truth, 2),
                                official.mvpe_rel_l2_per_sample(pred, truth)], axis=1)
                for i, label in enumerate(labels):
                    row = sums[arm].setdefault(label, {'n': 0, 'errors': np.zeros(3, dtype=np.float64)})
                    row['n'] += 1; row['errors'] += err[i]
    def finish(group):
        n = sum(r['n'] for r in group.values())
        errors = sum(r['errors'] for r in group.values()) / n
        scores = [official.score_error(float(v)) for v in errors]
        metric = dict(zip(ev.METRICS[:3], scores))
        metric.update(sps_score=43.295329, time_score=92.032660)
        values = [*scores, 92.032660, 43.295329]
        metric['point_only_fitted_final'] = sum(w*s**.6854 for w, s in zip([.5,.1,.1,.1,.2], values))**(1/.6854)
        return metric
    metrics = {arm: finish(group) for arm, group in sums.items()}
    delta = {k: metrics['candidate'][k]-metrics['base'][k] for k in metrics['base']}
    by_traj = {k: finish({k:sums['candidate'][k]})['point_only_fitted_final']-
                  finish({k:sums['base'][k]})['point_only_fitted_final'] for k in sums['base']}
    fraction = sum(v > 0 for v in by_traj.values()) / len(by_traj)
    passed = (delta['point_only_fitted_final'] >= .10 and fraction >= .7
              and delta['tke_rel_l2_score'] >= -.05 and delta['mvpe_rel_l2_score'] >= -.02)
    result = {'decision': 'PASS_NEXT_STAGE_ONLY' if passed else 'STOP_NO_LARGE_POINT_GAIN',
              'held': held, 'metrics': metrics, 'delta': delta, 'trajectory_delta': by_traj,
              'positive_trajectory_fraction': fraction, 'seal': seal,
              'scope': 'Real/correction stage held AoA; shared Sim ancestry; SPS held constant, not measured',
              'seconds': time.monotonic()-started}
    (a.output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
