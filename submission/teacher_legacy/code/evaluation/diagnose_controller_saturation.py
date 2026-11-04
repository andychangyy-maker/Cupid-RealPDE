"""Read-only energy-controller bound diagnostic on the frozen C held15 point."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/training'))
import train_b210_held_residual_uq as held


def main(args):
    started = time.time()
    torch, _, _, prefix = held.runtime()
    import numpy as np
    import evaluate_b210_transfer as ev
    from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    point_sha = held.sha256(args.point)
    assert point_sha == 'ce84999755b685bfc84061cec3367ee324f64d71661ee5fcfd33320182c5b5fa'
    assert held.sha256(args.manifest) == held.MANIFEST_SHA
    payload = torch.load(args.point, map_location='cpu', weights_only=False)
    assert payload['normalizer_sha256'] == '641e7fdfa05af5fefc195a8d7906e0cd44796e66b7d3cc4fac42ca99eabf7b1d'
    config = dict(payload['model_config'])
    assert config.pop('architecture') == 'ensemble_unet'
    config = EnsembleUNet2dConfig(**config)
    assert config.physical_energy_log_limit == 1.0
    assert config.physical_energy_member_sync and not config.physical_energy_horizon_local
    model = EnsembleUNet2d(config).float()
    model.load_state_dict(payload['model_state_dict'], strict=True)
    norm, _ = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
    model.cuda().eval()
    raw = []
    hook = model.energy_controller.register_forward_hook(lambda _m, _i, output: raw.append(output))
    rows = [r for r in ev.clean_rows(args.manifest) if r['aoa'] == 15]
    scorer = ev.import_file(ev.SCORER, 'controller_saturation_scorer')
    functions = (lambda p, y: scorer.rel_l2_per_sample(p, y, 2),
                 lambda p, y: scorer.tke_rel_l2_per_sample(p, y, 2), scorer.mvpe_rel_l2_per_sample)
    hashes = {key: hashlib.sha256() for key in ('input', 'truth')}
    records, metric_errors = [], []
    def energy(uv):
        return .5 * np.mean((uv - uv.mean(axis=1, keepdims=True)) ** 2, axis=1).sum(axis=-1)
    with torch.inference_mode():
        for history, truth, labels in ev.tail_batches(rows, args.real_dir):
            hashes['input'].update(history.tobytes()); hashes['truth'].update(truth.tobytes())
            x, _ = norm.preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
            raw.clear()
            prediction = model(x)
            assert len(raw) == 1
            delta = raw[0]
            if delta.shape[-2:] != prediction.shape[2:4]:
                delta = torch.nn.functional.interpolate(delta, size=prediction.shape[2:4], mode='bilinear', align_corners=False)
            assert delta.shape[1] == 1
            delta = torch.tanh(delta.float())[:, 0].cpu().numpy()
            zeros = torch.zeros_like(prediction[..., :2])
            prediction = held.absolute_distribution(prediction, zeros, zeros, norm)[0].cpu().numpy()
            assert np.isfinite(prediction).all() and np.isfinite(delta).all()
            metric_errors.extend(np.stack([fn(prediction, truth) for fn in functions], axis=-1).tolist())
            kp, kt = energy(prediction[..., :2].astype(np.float64)), energy(truth[..., :2].astype(np.float64))
            for i, file in enumerate(labels):
                error = (kp[i] - kt[i]) ** 2
                upper = (delta[i] >= .95) & (kp[i] < kt[i])
                lower = (delta[i] <= -.95) & (kp[i] > kt[i])
                denominator = max(float(error.sum()), 1e-30)
                records.append(dict(file=file, blocked_fraction=float(error[upper | lower].sum()) / denominator,
                                    upper_fraction=float(error[upper].sum()) / denominator,
                                    lower_fraction=float(error[lower].sum()) / denominator,
                                    positive_cap_pixels=float((delta[i] >= .95).mean()),
                                    negative_cap_pixels=float((delta[i] <= -.95).mean()),
                                    delta_min=float(delta[i].min()), delta_max=float(delta[i].max())))
    hook.remove()
    assert len(records) == 153 and len(rows) == 17
    scores = {key: scorer.score_error(value) for key, value in zip(ev.METRICS[:3], np.mean(metric_errors, axis=0))}
    reference = json.loads(args.control_result.read_text())['results']['control']
    hashes = {key: value.hexdigest() for key, value in hashes.items()}
    assert hashes == reference['hashes']
    assert max(abs(scores[key] - reference['point_metrics'][key]) for key in scores) <= 1e-5
    per_file = {row['file']: float(np.mean([r['blocked_fraction'] for r in records if r['file'] == row['file']])) for row in rows}
    mean_blocked = float(np.mean([r['blocked_fraction'] for r in records]))
    count = sum(value >= .10 for value in per_file.values())
    result = dict(decision=('PASS_' if mean_blocked >= .25 and count / 17 >= .70 else 'STOP_') + 'CONTROLLER_BOUND_MECHANISM',
                  mean_blocked_squared_energy_error_fraction=mean_blocked, trajectories_at_least_point10=count,
                  trajectory_count=17, per_trajectory=per_file, samples=records, point_metrics=scores, hashes=hashes,
                  point_sha256=point_sha, source_sha256=held.sha256(__file__), elapsed_seconds=time.time() - started,
                  scope='Frozen C Real-stage held15 mechanism diagnostic; no counterfactual, training, or official gain.')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('decision', 'mean_blocked_squared_energy_error_fraction', 'trajectories_at_least_point10', 'elapsed_seconds')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('point', 'manifest', 'real-dir', 'control-result', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
