"""Compare one sealed B256 teacher with fixed Real A/C, all native FP16."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import evaluate_b210_transfer as replay
from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig

REFERENCES = {
    'A': (Path('/dev/shm/b210_teacher_from_start_20260906/point_A/ema999_u4000_primary.pth'),
          '259999ea57ade0a87a368d0f13bc4aa51f0fdfd70e509c9925a235ccd46b55d1'),
    'old_C': (Path('/root/autodl-tmp/b210_transfer_20260905/runs/main_v1/point_C/ema999_u4000_primary.pth'),
              '96c0082c921c2aaeb449ef14ba5f94a7361395164b2064a191c407c337509903'),
}
POINT_KEYS = replay.METRICS[:3]
FIXED_SPS = 54.27448783998261


def summarize(official, errors):
    metrics = dict(zip(POINT_KEYS, (official.score_error(x) for x in errors.mean(axis=0, dtype=np.float64))))
    metrics['point_only_fitted_final'] = replay.fixed_time_final({**metrics, 'sps_score': FIXED_SPS})
    return metrics


@torch.no_grad()
def predict(model, stats, inputs):
    values = torch.as_tensor(inputs, dtype=torch.float16, device='cuda:0')
    normalized = (values - stats['mean_input']) / stats['std_input']
    physical = model(normalized) * stats['std_target'] + stats['mean_target']
    prediction = physical.float().cpu().numpy()
    prediction[..., 2] = 0
    return prediction


def execute(args):
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    receipt = json.loads(args.candidate_receipt.read_text())
    if receipt.get('updates') != 4000 or receipt.get('sealed') is not True:
        raise ValueError('Candidate must be a sealed Real4000 endpoint')
    assets = {name: (getattr(args, 'reference_' + name.lower()) or path, digest)
              for name, (path, digest) in REFERENCES.items()}
    assets['B256'] = (Path(receipt['checkpoint_path']), receipt['checkpoint_sha256'])
    identities = {}
    for name, (path, expected) in assets.items():
        if replay.sha256(path) != expected:
            raise ValueError('Changed checkpoint: ' + name)
        identities[name] = {'path': str(path), 'sha256': expected}
    rows = replay.clean_rows(args.clean_manifest)
    official = replay.import_file(replay.SCORER, 'b256_point_scoring')
    result = {'evidence': 'Explored clean716 teacher direction check; not independent, official, or deployment gain',
              'candidate_receipt_sha256': replay.sha256(args.candidate_receipt), 'assets': identities,
              'scorer_sha256': replay.sha256(replay.SCORER),
              'precision': 'all full native checkpoint states, FP16 prediction, FP32 u/v scoring; no INT8',
              'normalizer_buffers': 'preserved from strict checkpoint loading', 'models': {}}
    normalizers = []
    for name, (path, _) in assets.items():
        payload = torch.load(path, map_location='cpu', weights_only=False)
        config = dict(payload['model_config'])
        if config.pop('architecture') != 'ensemble_unet' or config['base'] != (256 if name == 'B256' else 210):
            raise ValueError('Wrong model architecture: ' + name)
        model = EnsembleUNet2d(EnsembleUNet2dConfig(**config))
        model.load_state_dict(payload['model_state_dict'], strict=True)
        parameters = sum(p.numel() for p in model.parameters())
        if parameters != (263574885 if name == 'B256' else 178705575):
            raise ValueError('Wrong parameter count: ' + name)
        stats = {key: torch.as_tensor(value, device='cuda:0', dtype=torch.float16)
                 for key, value in payload['normalizer'].items()}
        normalizers.append(hashlib.sha256(json.dumps(payload['normalizer'], sort_keys=True).encode()).hexdigest())
        model.cuda().half().eval()
        del payload
        errors, labels = [], []
        hashes = {key: hashlib.sha256() for key in ('input', 'target', 'prediction')}
        for inputs, truth, batch_labels in replay.tail_batches(rows, args.real_dir):
            if time.monotonic() - started > args.wall_budget_seconds:
                raise TimeoutError('Point evaluation budget expired')
            prediction = predict(model, stats, inputs)
            if prediction.shape != truth.shape:
                raise ValueError('Prediction and target shapes differ')
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite point prediction')
            for key, array in (('input', inputs), ('target', truth), ('prediction', prediction)):
                hashes[key].update(array.tobytes())
            errors.append(np.stack((official.rel_l2_per_sample(prediction, truth, 2),
                                    official.tke_rel_l2_per_sample(prediction, truth, 2),
                                    official.mvpe_rel_l2_per_sample(prediction, truth)), axis=1))
            labels.extend(batch_labels)
        errors = np.concatenate(errors)
        if len(errors) != 716 or not np.isfinite(errors).all():
            raise ValueError('Wrong evaluation population or invalid error')
        aoa_by_file = {row['file']: row['aoa'] for row in rows}
        aoas = np.array([aoa_by_file[label] for label in labels])
        result['models'][name] = {'metrics': summarize(official, errors), 'parameters': parameters,
            'by_aoa': {str(a): summarize(official, errors[aoas == a]) for a in (0, 5, 10, 15, 20)},
            'per_window_errors_rel_tke_mvpe': errors.tolist(),
            **{key + '_sha256': value.hexdigest() for key, value in hashes.items()}}
        print(json.dumps({'model': name, 'metrics': result['models'][name]['metrics']}), flush=True)
        del model, stats
        torch.cuda.empty_cache()
    if len(set(normalizers)) != 1:
        raise ValueError('Compared models have different Real normalization')
    for key in ('input_sha256', 'target_sha256'):
        if len({m[key] for m in result['models'].values()}) != 1:
            raise ValueError('Compared windows changed')
    candidate, baseline = result['models']['B256'], result['models']['old_C']
    key = 'point_only_fitted_final'
    delta = candidate['metrics'][key] - baseline['metrics'][key]
    aoa_deltas = {a: candidate['by_aoa'][a][key] - baseline['by_aoa'][a][key] for a in candidate['by_aoa']}
    rel_delta = candidate['metrics']['rel_l2_score'] - baseline['metrics']['rel_l2_score']
    passed = delta >= .10 and sum(x > 0 for x in aoa_deltas.values()) >= 3 and rel_delta >= 0
    result.update(gate={'baseline': 'old_C', 'min_delta': .10, 'min_positive_aoas': 3,
                        'no_overall_rel_regression': True, 'actual_delta': delta,
                        'actual_rel_delta': rel_delta, 'by_aoa_delta': aoa_deltas, 'pass': passed},
                  decision='TEACHER_SIGNAL_REQUIRES_INDEPENDENT_CONFIRMATION' if passed else 'STOP_B256_TEACHER_CONFIG',
                  fixed_v3_sps=FIXED_SPS, time_measured=False, sps_measured=False,
                  release_verified=False, elapsed_seconds=time.monotonic() - started)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return {'decision': result['decision'], 'gate': result['gate']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--candidate-receipt', type=Path)
    parser.add_argument('--reference-a', type=Path)
    parser.add_argument('--reference-old-c', type=Path)
    parser.add_argument('--clean-manifest', type=Path)
    parser.add_argument('--real-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--wall-budget-seconds', type=float, default=300)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'decision': 'PLAN_ONLY_NO_DATA_NO_GPU', 'candidate': 'B256', 'baseline': 'old_C',
                          'diagnostic_control': 'A', 'gate': 'point-only delta >= .10; >=3/5 AoA positive; Rel nonnegative'}))
    else:
        if any(getattr(args, key) is None for key in ('candidate_receipt', 'clean_manifest', 'real_dir', 'output')):
            parser.error('Execution requires candidate receipt, manifest, Real data and output')
        if args.output.exists() or not 0 < args.wall_budget_seconds <= 300:
            parser.error('Output must be new; budget must be within 300 seconds')
        print(json.dumps(execute(args), indent=2))
