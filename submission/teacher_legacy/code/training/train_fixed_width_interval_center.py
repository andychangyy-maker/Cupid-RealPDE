"""One fixed held15 test: frozen B210 with a zero-initialized interval-center readout."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import train_b210_held_residual_uq as held


def main(root, source_commit):
    begun = time.monotonic()
    torch, dense, native, prefix = held.runtime()
    import h5py
    import numpy as np
    import evaluate_b210_transfer as ev
    import train_ensemble_uq_oof as uq
    from models.ensemble_unet2d import EnsembleUNet2d
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    ev.TIME_SCORE = 92.03266035232576
    manifest = Path('/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json')
    rows = ev.clean_rows(manifest)
    fit, test = [r for r in rows if r['aoa'] != 15], [r for r in rows if r['aoa'] == 15]
    control_path = Path('/root/autodl-tmp/track1_reflection_exposure_stage_20260909/held15/control/uq/uq_endpoint.json')
    control_sha = 'e965bca6eb35468be17c1cf5d0cf6cb7d6c61cd81ca1ca161819d97da01fad08'
    control = held.read_bound(control_path, control_sha)
    assert held.sha256(control['checkpoint_path']) == control['checkpoint_sha256']
    payload = torch.load(control['checkpoint_path'], map_location='cpu', weights_only=False)
    config = dense.ft12.validate_b210_model_config(payload['model_config'])
    model = EnsembleUNet2d(config)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    normalizer, report = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
    assert report['sha256'] == control['normalizer_sha256']
    model.cuda().eval().requires_grad_(False)
    torch.manual_seed(34011)
    head = torch.nn.Conv2d(config.uq_hidden_channels, 2 * config.out_steps, 1).cuda()
    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    assert sum(p.numel() for p in head.parameters()) == 6440
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3200)
    captured = {}
    hook = model.uq_fusion.register_forward_hook(lambda _module, _args, value: captured.update(fused=value.detach()))
    scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
    reference = scale.new_tensor([.01075, .00550]).view(1, 1, 1, 1, 2)

    def centered(lo, hi):
        raw = head(captured['fused']).reshape(lo.shape[0], 20, 2, *lo.shape[2:4]).permute(0, 1, 3, 4, 2)
        shift = raw * reference / scale
        return lo - shift, hi + shift

    real_dir = Path('/root/autodl-tmp/realpde/data/real')
    arrays, indices = {}, []
    for row in fit:
        assert row['aoa'] != 15
        boundary = row['val_boundary']
        with h5py.File(real_dir / row['file'], 'r') as data:
            assert int(data['aoa'][()]) == row['aoa'] and int(data['re'][()]) == row['re']
            u, v = [np.asarray(data[k][:boundary, ::2, ::2], dtype=np.float32) for k in ('u', 'v')]
        arrays[row['file']] = np.stack((u, v, np.zeros_like(u)), axis=-1)
        for start in row['train_starts']:
            assert start + 40 <= boundary
            indices.append((row['file'], start))
    assert len(indices) == 2000
    rng = np.random.default_rng(34011)
    stream = hashlib.sha256()
    objective = native.sharp2_args()
    started = time.monotonic()
    for step in range(1, 3201):
        selection = rng.integers(len(indices), size=8, dtype=np.int64)
        stream.update(selection.astype('<i8').tobytes())
        batch = [arrays[indices[i][0]][indices[i][1]:indices[i][1] + 40] for i in selection]
        values = torch.from_numpy(np.stack(batch)).cuda()
        x, target = normalizer.preprocess(values[:, :20], values[:, 20:])
        with torch.no_grad():
            prediction, quantiles, (lo, hi), logits = uq.quantile_forward(model, x)
        lower, upper = centered(lo, hi)
        if step == 1:
            assert torch.equal(lower, lo) and torch.equal(upper, hi)
        loss, _ = uq.uq_objective(objective, model, prediction, quantiles, target, normalizer,
                                  interval_extents=(lower, upper), coverage_logits=logits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        assert torch.isfinite(loss) and torch.isfinite(norm) and (step != 1 or norm > 0)
        optimizer.step(); scheduler.step()
        if step == 1 or step % 400 == 0:
            print(json.dumps({'step': step, 'loss': float(loss), 'seconds': time.monotonic() - started}), flush=True)
        if time.monotonic() - begun > 1000:
            raise TimeoutError('fixed center training budget')
    assert all(torch.equal(value.cpu(), payload['model_state_dict'][name]) for name, value in model.state_dict().items())
    checkpoint = root / 'center_u3200.pth'
    torch.save({'schema': 'fixed_width_center_readout_v1', 'head_state': {k: v.cpu() for k, v in head.state_dict().items()},
                'base_checkpoint_sha256': control['checkpoint_sha256'], 'normalizer_sha256': report['sha256'],
                'updates': 3200, 'sample_stream_sha256': stream.hexdigest(), 'fit_files': [r['file'] for r in fit],
                'fit_windows': len(indices), 'held_aoa': 15, 'held_arrays_read_before_seal': 0,
                'source_commit': source_commit}, checkpoint)
    scorer = ev.import_file(ev.SCORER, 'center_readout_scorer')
    states = {arm: {r['file']: [] for r in test} for arm in ('baseline', 'candidate')}
    hashes = {key: hashlib.sha256() for key in ('input', 'truth', 'prediction')}
    max_width_error = 0.
    with torch.inference_mode():
        for history, truth, labels in ev.tail_batches(test, real_dir):
            x, _ = normalizer.preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
            prediction, _quantiles, (lo, hi), _logits = uq.quantile_forward(model, x)
            lower, upper = centered(lo, hi)
            max_width_error = max(max_width_error, float((((lower + upper) - (lo + hi)) * scale).abs().max()))
            assert max_width_error <= 2e-7
            for arm, extents in [('baseline', (lo, hi)), ('candidate', (lower, upper))]:
                p, low, high = [v.cpu().numpy() for v in held.absolute_distribution(prediction, *extents, normalizer)]
                if arm == 'baseline':
                    for key, value in [('input', history), ('truth', truth), ('prediction', p)]: hashes[key].update(value.tobytes())
                    baseline_point = p.copy()
                else:
                    assert np.array_equal(p, baseline_point)
                for name in dict.fromkeys(labels):
                    take = np.asarray([name == label for label in labels])
                    states[arm][name].append(ev.metric_state(scorer, p[take], truth[take], low[take], high[take]))
            if time.monotonic() - begun > 1150: raise TimeoutError('fixed center scoring budget')
    states = {arm: {name: ev.merge_states(items) for name, items in groups.items()} for arm, groups in states.items()}
    metrics = {arm: ev.finalize(scorer, ev.merge_states(list(groups.values()))) for arm, groups in states.items()}
    delta = {k: metrics['candidate'][k] - value for k, value in metrics['baseline'].items()}
    assert all(delta[k] == 0 for k in ev.METRICS[:3])
    positive = sum(ev.finalize(scorer, states['candidate'][r['file']])['fixed_time_fitted_final'] >=
                   ev.finalize(scorer, states['baseline'][r['file']])['fixed_time_fitted_final'] for r in test)
    passed = delta['sps_score'] >= .4 and delta['fixed_time_fitted_final'] >= .09 and positive >= .6 * len(test)
    result = {'decision': 'PASS_NEXT_STAGE_FOLD' if passed else 'STOP_FIXED_WIDTH_CENTER_CONFIG',
              'metrics': metrics, 'delta': delta, 'states': states, 'positive_trajectories': positive,
              'trajectory_count': len(test), 'frozen_base_state_bitwise': True, 'initial_identity': True,
              'head_parameters': 6440, 'max_physical_width_error': max_width_error,
              'hashes': {k: v.hexdigest() for k, v in hashes.items()}, 'checkpoint_sha256': held.sha256(checkpoint),
              'control_receipt_sha256': control_sha, 'seconds': time.monotonic() - begun,
              'scope': 'Real/UQ stage held15; shared Sim has teacher ancestry; not official gain'}
    held.write_json(root / 'RESULT.json', result)
    hook.remove()
    print(json.dumps({k: result[k] for k in ('decision', 'delta', 'positive_trajectories', 'seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    try:
        main(args.root, args.source_commit)
    except Exception:
        held.write_json(args.root / 'FAILED.json', {'traceback': traceback.format_exc()})
        raise
