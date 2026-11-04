"""One fixed nested-AoA residual experiment, retaining mature Real UQ initialization."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
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
    from models.ensemble_unet2d import EnsembleUNet2d
    begun = time.monotonic()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    ev.TIME_SCORE = 92.03266035232576
    source = Path(__file__).resolve().parents[2]
    assets = root.parent / 'assets'
    manifest = Path('/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json')
    real_dir = Path('/root/autodl-tmp/realpde/data/real')
    sim = Path('/root/autodl-tmp/b210_transfer_20260905/runs/causal_rebuild/causal_standard20k/model_10000.pth')
    sim_sha = '628860c7d82239d942895f08a6fd72ede81feb6a7cad4de4e32e110ed2742b9b'
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
    inner_receipts = {}
    for angle in (0, 5, 10, 20):
        folder = root / f'inner{angle}'
        command = [sys.executable, str(source / 'code/training/train_b210_transfer_real4k.py'),
                   '--execute', '--arm', 'C', '--held-aoa', '15', '--inner-held-aoa', str(angle),
                   '--full-real-trajectories', '--source-commit', source_commit,
                   '--start-sim', str(sim), '--start-sim-sha256', sim_sha,
                   '--start-metadata', str(assets / 'sim_binding.json'),
                   '--start-metadata-sha256', '282307574088299de967a7d64597a417150dd81cb18ea7b8d2897ee29b00e96f',
                   '--clean-manifest', str(manifest), '--real-dir', str(real_dir),
                   '--output-dir', str(folder), '--device', 'cuda:0', '--wall-budget-seconds', '1500']
        held.write_json(root / 'status.json', {'phase': 'inner_point', 'inner': angle, 'command': command})
        with (root / f'inner{angle}.log').open('wb') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1580)
        receipt = json.loads((folder / 'point_endpoint.json').read_text())
        assert receipt['held_aoa'] == 15 and receipt['inner_held_aoa'] == angle
        assert receipt['fit_files'] == [r['file'] for r in select_rows(rows, 15, inner_held_aoa=angle)]
        assert held.sha256(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
        inner_receipts[str(angle)] = receipt
        print(json.dumps({'inner_point_sealed': angle, 'sha256': receipt['checkpoint_sha256']}), flush=True)
    held.write_json(root / 'INNER_POINTS_SEALED.json', inner_receipts)

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
    bank_started = time.monotonic()
    height, width = dataset[0][0].shape[1:3]
    residuals = torch.empty((len(dataset), 20, height, width, 2), dtype=torch.float32)
    factors = torch.empty(len(dataset), dtype=torch.float32)
    filled = torch.zeros(len(dataset), dtype=torch.bool)
    with torch.inference_mode():
        for angle in (0, 5, 10, 20):
            payload, model, normalizer, norm_report = load_model(inner_receipts[str(angle)]['checkpoint_path'])
            assert norm_report['sha256'] == payload['normalizer_sha256']
            chosen = [i for i, row in enumerate(identities) if row['aoa'] == angle]
            for offset in range(0, len(chosen), 8):
                selected = chosen[offset:offset + 8]
                history = torch.stack([dataset[i][0] for i in selected]).cuda()
                truth = torch.stack([dataset[i][1] for i in selected]).cuda()
                x, _ = normalizer.preprocess(history, truth)
                point = held.absolute_distribution(*model.forward_distribution(x), normalizer)[0]
                residuals[selected] = (truth[..., :2] - point[..., :2]).cpu()
                factors[selected] = native.ensemble_train.common.sample_sps_factor(point[..., :2], truth[..., :2]).cpu()
                assert not filled[selected].any()
                filled[selected] = True
                if time.monotonic() - bank_started > 600:
                    raise TimeoutError('nested residual cache budget')
            del model, payload
    assert filled.all() and torch.isfinite(residuals).all() and torch.isfinite(factors).all()
    bank = root / 'nested_prefix_residuals.pth'
    torch.save({'residual_physical': residuals, 'point_factor': factors, 'identities': identities,
                'inner_receipts': inner_receipts, 'outer_aoa': 15, 'source_commit': source_commit}, bank)
    held.write_json(root / 'BANK_SEALED.json', {'sha256': held.sha256(bank), 'windows': 2000,
                    'identities_sha256': ev.canonical_sha256(identities), 'outer_arrays_read': 0})
    print(json.dumps({'residual_bank_sealed': True, 'windows': len(dataset)}), flush=True)
    endpoints = {}
    for arm in ('native_inpoint', 'nested_oof'):
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
            x, target = normalizer.preprocess(history.cuda(), truth.cuda())
            p, q, extents, logits = uq.quantile_forward(model, x)
            overrides = {} if arm == 'native_inpoint' else {
                'residual_physical_override': residuals[selected].cuda(),
                'point_factor_override': factors[selected].cuda()}
            loss, _ = uq.uq_objective(objective, model, p, q, target, normalizer,
                        interval_extents=extents, coverage_logits=logits, **overrides)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            assert torch.isfinite(loss) and torch.isfinite(norm) and (step != 1 or norm > 0)
            optimizer.step(); scheduler.step()
            if step % 800 == 0:
                print(json.dumps({'arm': arm, 'step': step, 'loss': float(loss), 'seconds': time.monotonic() - started}), flush=True)
            if time.monotonic() - started > 650:
                raise TimeoutError('nested UQ arm budget')
        assert stream.hexdigest() == original['sample_stream_sha256']
        assert all(torch.equal(value.cpu(), payload['model_state_dict'][name])
                   for name, value in model.state_dict().items() if not name.startswith(uq.UQ_PREFIXES))
        path = root / f'{arm}_uq3200.pth'
        torch.save({'uq_state_dict': uq.uq_state_dict(model), 'point_checkpoint_sha256': point_sha,
                    'normalizer_sha256': report['sha256'], 'sample_stream_sha256': stream.hexdigest(),
                    'source_commit': source_commit}, path)
        endpoints[arm] = {'path': str(path), 'sha256': held.sha256(path), 'point_frozen_bitwise': True}
        held.write_json(root / f'{arm}_sealed.json', endpoints[arm])
        del model, payload, optimizer, scheduler, parameters
    held.write_json(root / 'ALL_ENDPOINTS_SEALED.json', endpoints)
    scorer = ev.import_file(ev.SCORER, 'nested_oof_scorer')
    test = select_rows(rows, 15, evaluation=True)
    results = {}
    eval_started = time.monotonic()
    for arm in ('original', 'native_inpoint', 'nested_oof'):
        path = assets / ('outer15_base_uq.pth' if arm == 'original' else 'outer15_point.pth')
        payload, model, normalizer, _ = load_model(path)
        if arm != 'original':
            patch = torch.load(endpoints[arm]['path'], map_location='cpu', weights_only=False)
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
                    raise TimeoutError('nested held evaluation budget')
        states = {name: ev.merge_states(value) for name, value in states.items()}
        results[arm] = {'states': states, 'metrics': ev.finalize(scorer, ev.merge_states(list(states.values()))),
                        'hashes': {key: value.hexdigest() for key, value in hashes.items()}}
        del model, payload
    assert len({json.dumps(value['hashes'], sort_keys=True) for value in results.values()}) == 1
    delta = {reference: {key: results['nested_oof']['metrics'][key] - results[reference]['metrics'][key]
                        for key in (*ev.METRICS, 'fixed_time_fitted_final')} for reference in ('original', 'native_inpoint')}
    positive = sum(ev.finalize(scorer, results['nested_oof']['states'][name])['sps_score'] >=
                   ev.finalize(scorer, state)['sps_score'] for name, state in results['original']['states'].items())
    passed = all(value['sps_score'] >= .4 and value['fixed_time_fitted_final'] >= .09 for value in delta.values()) and positive >= .6 * len(test)
    result = {'decision': 'PASS_NESTED_RESIDUAL_STAGE15' if passed else 'STOP_NESTED_RESIDUAL_STAGE15',
              'delta': delta, 'results': results, 'positive_trajectories': positive,
              'trajectories': len(test), 'inner_receipts': inner_receipts, 'endpoints': endpoints,
              'source_commit': source_commit, 'seconds': time.monotonic() - begun,
              'scope': 'Nested Real-stage exclusion; shared Sim has Real ancestry; no official gain'}
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
