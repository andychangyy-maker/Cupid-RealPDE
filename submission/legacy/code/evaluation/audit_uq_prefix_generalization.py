"""Score sealed UQ variants on their original fit prefixes without retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import train_b210_held_residual_uq as held


def main(root, source_commit):
    torch, dense, native, prefix = held.runtime()
    import h5py
    import numpy as np
    import evaluate_b210_transfer as ev
    import train_ensemble_uq_oof as uq
    from models.ensemble_unet2d import EnsembleUNet2d
    from models.independent_interval_refiner import IndependentIntervalRefiner
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    ev.TIME_SCORE = 92.03266035232576
    control = held.read_bound(Path('/root/autodl-tmp/track1_reflection_exposure_stage_20260909/held15/control/uq/uq_endpoint.json'),
                              'e965bca6eb35468be17c1cf5d0cf6cb7d6c61cd81ca1ca161819d97da01fad08')
    assert held.sha256(control['checkpoint_path']) == control['checkpoint_sha256']
    payload = torch.load(control['checkpoint_path'], map_location='cpu', weights_only=False)
    normalizer, report = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
    assert report['sha256'] == control['normalizer_sha256']
    rows = [row for row in ev.clean_rows(Path('/root/autodl-tmp/c208_dr1_clean2544_loao_attested_v1/runtime_train_manifest.json')) if row['aoa'] != 15]
    assert [r['file'] for r in rows] == control['point_receipt']['fit_files']
    experiment = Path('/root/roy_route_20260910')
    bindings = {'base_checkpoint_sha256': control['checkpoint_sha256']}
    models = {}
    for arm, expected in [('base', None),
                          ('native_continue', 'ff9559ac414b9b59ffc006a26f36f7e33efab98adbb658387af67c2ff73fb2c4'),
                          ('distilled', 'f811dc12ba423c2411d02309b595bd8a87e4b63d3059c96066870f6ab979f492')]:
        model = EnsembleUNet2d(dense.ft12.validate_b210_model_config(payload['model_config']))
        model.load_state_dict(payload['model_state_dict'], strict=True)
        if expected:
            receipt = json.loads((experiment / f'q4_function_distillation/held15/{arm}_endpoint.json').read_text())
            assert receipt['checkpoint_sha256'] == expected == held.sha256(receipt['checkpoint_path'])
            assert receipt['base_checkpoint_sha256'] == control['checkpoint_sha256']
            patch = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)['uq_state_dict']
            uq.load_uq_state(model, patch)
            bindings[arm] = expected
        models[arm] = model.cuda().eval().requires_grad_(False)
    head_path = experiment / 'independent_interval_stage15/independent_interval_u8000.pth'
    assert held.sha256(head_path) == '3ac09ea6cbfd6ab00fd11fd5940e3781d8ce0c6bc8c852b93bfbf6e267603d03'
    head_payload = torch.load(head_path, map_location='cpu', weights_only=False)
    assert head_payload['base_checkpoint_sha256'] == control['checkpoint_sha256']
    assert head_payload['fit_files'] == [r['file'] for r in rows]
    head = IndependentIntervalRefiner(normalizer.std_target).cuda().eval().requires_grad_(False)
    head.load_state_dict(head_payload['head_state'], strict=True)
    bindings['independent'] = held.sha256(head_path)
    del payload, head_payload
    scorer = ev.import_file(ev.SCORER, 'uq_prefix_scorer')
    arms = ('base', 'native_continue', 'distilled', 'independent')
    states = {arm: {} for arm in arms}
    point_hashes = {arm: hashlib.sha256() for arm in arms}
    input_hash, truth_hash = hashlib.sha256(), hashlib.sha256()
    real_dir = Path('/root/autodl-tmp/realpde/data/real')
    windows = 0
    with torch.inference_mode():
        for row in rows:
            with h5py.File(real_dir / row['file'], 'r') as data:
                u, v = [np.asarray(data[key][:row['val_boundary'], ::2, ::2], dtype=np.float32) for key in ('u', 'v')]
            values = np.stack((u, v, np.zeros_like(u)), axis=-1)
            trajectory = {arm: [] for arm in arms}
            for offset in range(0, len(row['train_starts']), 8):
                starts = row['train_starts'][offset:offset + 8]
                assert all(t + 40 <= row['val_boundary'] for t in starts)
                history = np.stack([values[t:t + 20] for t in starts])
                truth = np.stack([values[t + 20:t + 40] for t in starts])
                input_hash.update(history.tobytes()); truth_hash.update(truth.tobytes())
                x, _ = normalizer.preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
                outputs = {arm: model.forward_distribution(x) for arm, model in models.items()}
                p, lo, hi = outputs['base']
                outputs['independent'] = (p, *head(x, p, lo, hi))
                for arm, output in outputs.items():
                    p, low, high = [value.cpu().numpy() for value in held.absolute_distribution(*output, normalizer)]
                    point_hashes[arm].update(p.tobytes())
                    trajectory[arm].append(ev.metric_state(scorer, p, truth, low, high))
                windows += len(starts)
                if time.monotonic() - started > 540:
                    raise TimeoutError('sealed prefix diagnostic budget')
            for arm in arms:
                states[arm][row['file']] = ev.merge_states(trajectory[arm])
    assert windows == control['fit_windows'] == 2000
    assert len({h.hexdigest() for h in point_hashes.values()}) == 1
    metrics = {arm: ev.finalize(scorer, ev.merge_states(list(value.values()))) for arm, value in states.items()}
    delta = {arm: {key: value[key] - metrics['base'][key] for key in (*ev.METRICS, 'coverage', 'mean_width', 'fixed_time_fitted_final')}
             for arm, value in metrics.items()}
    assert all(delta[arm][key] == 0 for arm in arms for key in ev.METRICS[:3])
    supported = all(delta[arm]['sps_score'] >= .10 for arm in ('native_continue', 'independent'))
    result = {'decision': 'PREFIX_GAIN_HELD_LOSS_SUPPORTED' if supported else 'PREFIX_GAIN_HELD_LOSS_NOT_ESTABLISHED',
              'metrics': metrics, 'delta': delta, 'states': states, 'bindings': bindings,
              'windows': windows, 'trajectories': len(rows), 'held_or_tail_arrays_read': 0,
              'input_sha256': input_hash.hexdigest(), 'truth_sha256': truth_hash.hexdigest(),
              'point_sha256': point_hashes['base'].hexdigest(), 'seconds': time.monotonic() - started,
              'source_commit': source_commit, 'training_performed': False,
              'scope': 'Training-prefix diagnostic of stopped configurations; not a release gate'}
    held.write_json(root / 'RESULT.json', result)
    print(json.dumps({key: result[key] for key in ('decision', 'delta', 'seconds')}), flush=True)


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
