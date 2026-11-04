"""Two sealed stage folds: Q4 absolute-endpoint distillation into frozen-point B210."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import train_b210_held_residual_uq as held


def relocated_extents(point, teacher_point, lower, upper):
    difference = point[..., :2].detach() - teacher_point[..., :2]
    return lower + difference, upper - difference


def main(root, source_commit):
    torch, dense, native, prefix = held.runtime()
    import h5py
    import numpy as np
    import evaluate_b210_transfer as ev
    import train_ensemble_uq_oof as uq
    from models.ensemble_unet2d import EnsembleUNet2d
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    ev.TIME_SCORE = 92.03266035232576
    begun = time.monotonic()
    original = Path('/root/autodl-tmp/track1_reflection_exposure_stage_20260909')
    manifest = Path('/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json')
    real_dir = Path('/root/autodl-tmp/realpde/data/real')
    rows = ev.clean_rows(manifest)
    receipt_hashes = {
        15: ('e965bca6eb35468be17c1cf5d0cf6cb7d6c61cd81ca1ca161819d97da01fad08',
             '7fe6dcc90eb8f7b73445a2db2aa8f750f0e0128f5375d604fafdb46c906219da'),
        20: ('96ea76a6504b2b36b31515ccf36642d9450824b2600486f14f9664d38a7044d1',
             '095dd2762ec02ed4744f69d1127969f9f144aa84ccddd4e63fa5973b3cfc4c13'),
    }
    objective = native.sharp2_args()
    objective.frozen_point_eval_mode = True
    sealed, controls = {}, {}

    def load(receipt):
        assert held.sha256(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
        payload = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)
        model = EnsembleUNet2d(dense.ft12.validate_b210_model_config(payload['model_config']))
        model.load_state_dict(payload['model_state_dict'], strict=True)
        normalizer, report = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
        assert report['sha256'] == receipt['normalizer_sha256']
        return payload, model.cuda().eval(), normalizer

    for angle in (15, 20):
        folder = root / f'held{angle}'
        folder.mkdir()
        control, teacher_receipt = [held.read_bound(original / f'held{angle}' / label / 'uq/uq_endpoint.json', sha)
                                   for label, sha in zip(('control', 'reflection'), receipt_hashes[angle])]
        assert control['normalizer_sha256'] == teacher_receipt['normalizer_sha256']
        fit = [r for r in rows if r['aoa'] != angle]
        for receipt in (control, teacher_receipt):
            assert receipt['held_aoa'] == receipt['point_receipt']['held_aoa'] == angle
            assert receipt['point_receipt']['fit_files'] == [r['file'] for r in fit]
        controls[str(angle)] = control
        arrays, indices = {}, []
        for row in fit:
            boundary = row['val_boundary']
            with h5py.File(real_dir / row['file'], 'r') as data:
                assert int(data['aoa'][()]) == row['aoa'] and int(data['re'][()]) == row['re']
                u, v = [np.asarray(data[key][:boundary, ::2, ::2], dtype=np.float32) for key in ('u', 'v')]
            arrays[row['file']] = np.stack((u, v, np.zeros_like(u)), axis=-1)
            for start in row['train_starts']:
                assert start + 40 <= boundary
                indices.append((row['file'], start))
        assert len(indices) == control['fit_windows']
        teacher_payload, teacher, _ = load(teacher_receipt)
        teacher.requires_grad_(False)
        del teacher_payload
        arm_records = {}
        for arm in ('native_continue', 'distilled'):
            payload, model, normalizer = load(control)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith(uq.UQ_PREFIXES))
            uq.set_training_mode(objective, model)
            parameters = uq.trainable_parameters(model)
            optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3200)
            scale = normalizer.std_target[:2].view(1, 1, 1, 1, 2)
            reference = scale.new_tensor([.01075, .00550]).view(1, 1, 1, 1, 2)
            torch.manual_seed(18)
            rng, stream = np.random.default_rng(34012), hashlib.sha256()
            started = time.monotonic()
            for step in range(1, 3201):
                selected = rng.integers(len(indices), size=8, dtype=np.int64)
                stream.update(selected.astype('<i8').tobytes())
                batch = [arrays[indices[i][0]][indices[i][1]:indices[i][1] + 40] for i in selected]
                values = torch.from_numpy(np.stack(batch)).cuda()
                x, target = normalizer.preprocess(values[:, :20], values[:, 20:])
                point, quantiles, (lo, hi), logits = uq.quantile_forward(model, x)
                if arm == 'native_continue':
                    loss, _ = uq.uq_objective(objective, model, point, quantiles, target, normalizer,
                                              interval_extents=(lo, hi), coverage_logits=logits)
                else:
                    with torch.no_grad():
                        teacher_point, teacher_lo, teacher_hi = teacher.forward_distribution(x)
                        teacher_lo, teacher_hi = relocated_extents(point, teacher_point, teacher_lo, teacher_hi)
                    loss = .5 * (torch.nn.functional.smooth_l1_loss(lo * scale / reference, teacher_lo * scale / reference)
                                 + torch.nn.functional.smooth_l1_loss(hi * scale / reference, teacher_hi * scale / reference))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                assert torch.isfinite(loss) and torch.isfinite(norm) and (step != 1 or norm > 0)
                optimizer.step(); scheduler.step()
                if step == 1 or step % 400 == 0:
                    print(json.dumps({'angle': angle, 'arm': arm, 'step': step, 'loss': float(loss),
                                      'seconds': time.monotonic() - started}), flush=True)
                if time.monotonic() - started > 650 or time.monotonic() - begun > 2800:
                    raise TimeoutError('Q4 function distillation training budget')
            assert all(torch.equal(value.cpu(), payload['model_state_dict'][name])
                       for name, value in model.state_dict().items() if not name.startswith(uq.UQ_PREFIXES))
            checkpoint = folder / f'{arm}_uq3200.pth'
            record = {'schema': 'q4_absolute_interval_function_transfer_v1', 'arm': arm, 'held_aoa': angle,
                      'updates': 3200, 'fit_files': [r['file'] for r in fit], 'fit_windows': len(indices),
                      'sample_stream_sha256': stream.hexdigest(), 'base_checkpoint_sha256': control['checkpoint_sha256'],
                      'teacher_checkpoint_sha256': teacher_receipt['checkpoint_sha256'],
                      'normalizer_sha256': control['normalizer_sha256'], 'source_commit': source_commit,
                      'point_state_bitwise': True, 'held_arrays_read_before_seal': 0, 'seconds': time.monotonic() - started}
            torch.save({**record, 'uq_state_dict': uq.uq_state_dict(model)}, checkpoint)
            arm_records[arm] = {**record, 'checkpoint_path': str(checkpoint), 'checkpoint_sha256': held.sha256(checkpoint)}
            held.write_json(folder / f'{arm}_endpoint.json', arm_records[arm])
            del model, payload, optimizer, scheduler, parameters
        assert arm_records['native_continue']['sample_stream_sha256'] == arm_records['distilled']['sample_stream_sha256']
        sealed[str(angle)] = arm_records
        del teacher, arrays
    held.write_json(root / 'ALL_ENDPOINTS_SEALED.json', sealed)
    scorer = ev.import_file(ev.SCORER, 'q4_transfer_scorer')
    folds, pooled = {}, {key: [] for key in ('base', 'native_continue', 'distilled')}
    for angle in (15, 20):
        test = [r for r in rows if r['aoa'] == angle]
        fold = {}
        for arm in ('base', 'native_continue', 'distilled'):
            payload, model, normalizer = load(controls[str(angle)])
            if arm != 'base':
                record = sealed[str(angle)][arm]
                assert held.sha256(record['checkpoint_path']) == record['checkpoint_sha256']
                patch = torch.load(record['checkpoint_path'], map_location='cpu', weights_only=False)['uq_state_dict']
                assert set(patch) == {key for key in payload['model_state_dict'] if key.startswith(uq.UQ_PREFIXES)}
                uq.load_uq_state(model, patch)
            model.eval().requires_grad_(False)
            raw, point_hash = {r['file']: [] for r in test}, hashlib.sha256()
            with torch.inference_mode():
                for history, truth, labels in ev.tail_batches(test, real_dir):
                    x, _ = normalizer.preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
                    point, lo, hi = [value.cpu().numpy() for value in held.absolute_distribution(*model.forward_distribution(x), normalizer)]
                    point_hash.update(point.tobytes())
                    for name in dict.fromkeys(labels):
                        take = np.asarray([name == label for label in labels])
                        raw[name].append(ev.metric_state(scorer, point[take], truth[take], lo[take], hi[take]))
                    if time.monotonic() - begun > 2970:
                        raise TimeoutError('Q4 function distillation evaluation budget')
            states = {name: ev.merge_states(values) for name, values in raw.items()}
            fold[arm] = {'metrics': ev.finalize(scorer, ev.merge_states(list(states.values()))),
                         'states': states, 'point_sha256': point_hash.hexdigest()}
            pooled[arm].extend(states.values())
            del model, payload
        assert len({item['point_sha256'] for item in fold.values()}) == 1
        folds[str(angle)] = fold
    metrics = {arm: ev.finalize(scorer, ev.merge_states(states)) for arm, states in pooled.items()}
    delta = {reference: {key: metrics['distilled'][key] - metrics[reference][key]
                        for key in (*ev.METRICS, 'fixed_time_fitted_final')} for reference in ('base', 'native_continue')}
    fold_delta = {angle: item['distilled']['metrics']['sps_score'] - item['base']['metrics']['sps_score'] for angle, item in folds.items()}
    positive = sum(ev.finalize(scorer, item['distilled']['states'][name])['sps_score'] >= ev.finalize(scorer, state)['sps_score']
                   for item in folds.values() for name, state in item['base']['states'].items())
    total = sum(len(item['base']['states']) for item in folds.values())
    passed = (all(value >= 0 for value in fold_delta.values()) and delta['base']['sps_score'] >= .20
              and delta['base']['fixed_time_fitted_final'] >= .045 and delta['native_continue']['sps_score'] >= .10
              and positive >= .6 * total)
    result = {'decision': 'PASS_Q4_FUNCTION_TRANSFER_TWO_FOLD' if passed else 'STOP_Q4_FUNCTION_DISTILLATION',
              'metrics': metrics, 'delta': delta, 'fold_delta_sps': fold_delta, 'folds': folds,
              'positive_trajectories': positive, 'trajectories': total, 'sealed_endpoints': sealed,
              'source_commit': source_commit, 'seconds': time.monotonic() - begun,
              'scope': 'Selection-reused stage-LOAO FP32; no official gain or release claim'}
    held.write_json(root / 'RESULT.json', result)
    print(json.dumps({key: result[key] for key in ('decision', 'delta', 'fold_delta_sps', 'positive_trajectories', 'seconds')}), flush=True)


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
