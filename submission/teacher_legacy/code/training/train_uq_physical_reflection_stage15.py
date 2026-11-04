"""Frozen-point UQ reflection screen, with fixed original and same-H800 controls."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import train_b210_held_residual_uq as held


def main(root, source_commit):
    torch, dense, native, prefix = held.runtime()
    import h5py
    import numpy as np
    import evaluate_b210_transfer as ev
    import train_ensemble_uq_oof as uq
    from b210_stage_fold import select_rows
    from vertical_reflection_dataset import reflect_uvp
    from models.ensemble_unet2d import EnsembleUNet2d
    begun = time.monotonic()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    ev.TIME_SCORE = 92.03266035232576
    assets = root.parent / 'assets'
    manifest = Path('/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json')
    real_dir = Path('/root/autodl-tmp/realpde/data/real')
    point_sha = '8711cbe9e279a806dc29be9ab423672f77e498035929855d14a5fac877eb7a77'
    base_sha = '36ec9b40c7e26665b90245456fcf400cb7fff75aa75f9b7d0a01808c7c564a94'
    assert held.sha256(assets / 'outer15_point.pth') == point_sha
    assert held.sha256(assets / 'outer15_base_uq.pth') == base_sha
    rows = ev.clean_rows(manifest)
    fit = select_rows(rows, 15)
    point_receipt = held.read_bound(assets / 'outer15_point_endpoint.json',
                                   '9c3804cfd62d3c1161c71a712c38782f60a199ca0988796dd3dd765024753eb4')
    original = held.read_bound(assets / 'outer15_uq_endpoint.json',
                              'e965bca6eb35468be17c1cf5d0cf6cb7d6c61cd81ca1ca161819d97da01fad08')
    assert point_receipt['fit_files'] == [r['file'] for r in fit]
    def load_model(path):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        model = EnsembleUNet2d(dense.ft12.validate_b210_model_config(payload['model_config']))
        model.load_state_dict(payload['model_state_dict'], strict=True)
        normalizer, report = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
        return payload, model.cuda().eval(), normalizer, report

    arrays, identities, dataset = {}, [], []
    for row in fit:
        with h5py.File(real_dir / row['file'], 'r') as data:
            assert int(data['aoa'][()]) == row['aoa'] and int(data['re'][()]) == row['re']
            u, v = [np.asarray(data[k][:row['val_boundary'], ::2, ::2], dtype=np.float32) for k in ('u', 'v')]
        values = arrays[row['file']] = np.stack((u, v, np.zeros_like(u)), axis=-1)
        for start in row['train_starts']:
            assert start + 40 <= row['val_boundary']
            index = len(identities)
            identities.append({'file': row['file'], 'start': start, 'aoa': row['aoa']})
            dataset.append((torch.from_numpy(values[start:start + 20]),
                            torch.from_numpy(values[start + 20:start + 40]), index))
    assert len(dataset) == original['fit_windows'] == 2000
    native_path = root.parent / 'run15/native_inpoint_uq3200.pth'
    native_sha = '3e6204f4daa157ed0f1b789ffa655b3a0b23a06b3954c8f61a2ac42780c21f2c'
    assert held.sha256(native_path) == native_sha
    reflection_rng = torch.Generator().manual_seed(29009)
    reflection_stream = hashlib.sha256()
    reflected = 0
    endpoints = {}
    for arm in ('uq_reflection',):
        payload, model, normalizer, report = load_model(assets / 'outer15_point.pth')
        assert report['sha256'] == original['normalizer_sha256']
        native.uq_train.seed_everything(18)
        parameters = native.configure_uq_only(model)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0,
                    pin_memory=True, generator=torch.Generator().manual_seed(18))
        batches = native.ensemble_train.common.repro.cycling(loader)
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3200)
        objective = native.sharp2_args()
        stream = hashlib.sha256()
        started = time.monotonic()
        held.write_json(root / 'status.json', {'phase': 'uq', 'arm': arm})
        for step in range(1, 3201):
            native.set_uq_training_mode(model)
            history, truth, selected = next(batches)
            stream.update(selected.numpy().astype('<i8', copy=False).tobytes())
            mask = torch.rand(len(history), generator=reflection_rng) < .25
            reflected += int(mask.sum())
            reflection_stream.update(mask.numpy().astype('uint8').tobytes())
            history[mask] = reflect_uvp(history[mask])
            truth[mask] = reflect_uvp(truth[mask])
            x, target = normalizer.preprocess(history.cuda(), truth.cuda())
            p, q, extents, logits = uq.quantile_forward(model, x)
            loss, _ = uq.uq_objective(objective, model, p, q, target, normalizer,
                        interval_extents=extents, coverage_logits=logits)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            assert torch.isfinite(loss) and torch.isfinite(norm) and (step != 1 or norm > 0)
            optimizer.step(); scheduler.step()
            if step % 800 == 0:
                print(json.dumps({'arm': arm, 'step': step, 'loss': float(loss), 'seconds': time.monotonic() - started}), flush=True)
            if time.monotonic() - started > 650:
                raise TimeoutError('reflection UQ arm budget')
        assert stream.hexdigest() == original['sample_stream_sha256']
        assert all(torch.equal(value.cpu(), payload['model_state_dict'][name])
                   for name, value in model.state_dict().items() if not name.startswith(uq.UQ_PREFIXES))
        path = root / f'{arm}_uq3200.pth'
        torch.save({'uq_state_dict': uq.uq_state_dict(model), 'point_checkpoint_sha256': point_sha,
                    'normalizer_sha256': report['sha256'], 'sample_stream_sha256': stream.hexdigest(),
                    'source_commit': source_commit, 'physical_reflection': {
                        'probability': .25, 'isolated_cpu_seed': 29009, 'draws': 25600,
                        'reflected': reflected, 'mask_sha256': reflection_stream.hexdigest()}}, path)
        endpoints[arm] = {'path': str(path), 'sha256': held.sha256(path), 'point_frozen_bitwise': True}
        held.write_json(root / f'{arm}_sealed.json', endpoints[arm])
        del model, payload, optimizer, scheduler, parameters
    held.write_json(root / 'ALL_ENDPOINTS_SEALED.json', endpoints)
    evaluation_patches = {**endpoints, 'native_inpoint': {'path': str(native_path), 'sha256': native_sha}}
    scorer = ev.import_file(ev.SCORER, 'uq_reflection_scorer')
    test = select_rows(rows, 15, evaluation=True)
    results = {}
    eval_started = time.monotonic()
    for arm in ('original', 'native_inpoint', 'uq_reflection'):
        path = assets / ('outer15_base_uq.pth' if arm == 'original' else 'outer15_point.pth')
        payload, model, normalizer, _ = load_model(path)
        if arm != 'original':
            patch = torch.load(evaluation_patches[arm]['path'], map_location='cpu', weights_only=False)
            uq.load_uq_state(model, patch['uq_state_dict'])
        model.eval().requires_grad_(False)
        states = {r['file']: [] for r in test}
        hashes = {key: hashlib.sha256() for key in ('input', 'truth', 'point')}
        with torch.inference_mode():
            for history, truth, labels in ev.tail_batches(test, real_dir):
                x, _ = normalizer.preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
                p, lo, hi = [v.cpu().numpy() for v in held.absolute_distribution(*model.forward_distribution(x), normalizer)]
                for key, value in [('input', history), ('truth', truth), ('point', p)]: hashes[key].update(value.tobytes())
                for name in dict.fromkeys(labels):
                    take = np.asarray([name == label for label in labels])
                    states[name].append(ev.metric_state(scorer, p[take], truth[take], lo[take], hi[take]))
                if time.monotonic() - eval_started > 200:
                    raise TimeoutError('reflection held evaluation budget')
        states = {name: ev.merge_states(value) for name, value in states.items()}
        results[arm] = {'states': states, 'metrics': ev.finalize(scorer, ev.merge_states(list(states.values()))),
                        'hashes': {key: value.hexdigest() for key, value in hashes.items()}}
        del model, payload
    assert len({json.dumps(value['hashes'], sort_keys=True) for value in results.values()}) == 1
    delta = {reference: {key: results['uq_reflection']['metrics'][key] - results[reference]['metrics'][key]
                        for key in (*ev.METRICS, 'fixed_time_fitted_final')} for reference in ('original', 'native_inpoint')}
    positive = sum(ev.finalize(scorer, results['uq_reflection']['states'][name])['sps_score'] >=
                   ev.finalize(scorer, state)['sps_score'] for name, state in results['original']['states'].items())
    passed = all(value['sps_score'] >= .4 and value['fixed_time_fitted_final'] >= .09 for value in delta.values()) and positive >= .6 * len(test)
    result = {'decision': 'PASS_UQ_PHYSICAL_REFLECTION_STAGE15' if passed else 'STOP_UQ_PHYSICAL_REFLECTION',
              'delta': delta, 'results': results, 'positive_trajectories': positive,
              'trajectories': len(test), 'endpoints': endpoints,
              'physical_reflection': {'probability': .25, 'seed': 29009, 'reflected': reflected,
                                      'draws': 25600, 'mask_sha256': reflection_stream.hexdigest()},
              'source_commit': source_commit, 'seconds': time.monotonic() - begun,
              'scope': 'UQ physical augmentation; Real/UQ stage held15; shared Sim has Real ancestry; not official gain'}
    held.write_json(root / 'RESULT.json', result)
    print(json.dumps({key: result[key] for key in ('decision', 'delta', 'positive_trajectories', 'seconds')}), flush=True)


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
