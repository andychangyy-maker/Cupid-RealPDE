"""Compare two sealed same-source Real4000 endpoints on the fixed held15 tail."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/training'))
import train_b210_held_residual_uq as held


def main(args, *, member_teacher=False, canonical_stem=False, reversible_history=False):
    assert sum((member_teacher, canonical_stem, reversible_history)) <= 1
    torch, dense, native, prefix = held.runtime()
    import numpy as np
    import evaluate_b210_transfer as ev
    from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    ev.TIME_SCORE = 92.0326603523
    receipts, payloads, configs = {}, {}, {}
    for label in ('control', 'candidate'):
        receipt = json.loads((args.root / label / 'point_endpoint.json').read_text())
        assert held.sha256(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
        if (member_teacher or canonical_stem or reversible_history) and label == 'control':
            assert receipt['checkpoint_sha256'] == 'ce84999755b685bfc84061cec3367ee324f64d71661ee5fcfd33320182c5b5fa'
        payload = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)
        assert receipt['updates'] == payload['iteration'] == 4000
        assert receipt['held_aoa'] == payload['held_aoa'] == 15
        assert receipt['start_sim_sha256'] == '628860c7d82239d942895f08a6fd72ede81feb6a7cad4de4e32e110ed2742b9b'
        assert receipt['sample_stream_sha256'] == 'e00aeee3c15d51a48a5401f4c226c336649d0778234ae0fee71953d15dcc2319'
        if member_teacher and label == 'candidate':
            from train_real_member_teacher import INTERVENTION
            assert receipt['real_member_teacher'] == payload['start']['real_member_teacher'] == INTERVENTION
            assert payload['model_state_sha256'] == dense.dr1.model_state_sha256(payload['model_state_dict'])
            for member in range(4):
                for scale in range(3):
                    projection = payload['model_state_dict'][f'member_feature_adapters.{member}.{scale}.body.4.weight']
                    assert torch.isfinite(projection).all() and torch.count_nonzero(projection)>0
        if canonical_stem and label == 'candidate':
            from train_canonical_fluctuation_stem import INTERVENTION
            assert receipt['canonical_fluctuation_stem'] == payload['start']['canonical_fluctuation_stem'] == INTERVENTION
            assert payload['model_state_sha256'] == dense.dr1.model_state_sha256(payload['model_state_dict'])
            projection = payload['model_state_dict']['stem.fluctuation_projection.weight']
            assert torch.isfinite(projection).all() and torch.count_nonzero(projection) > 0
        if reversible_history and label == 'candidate':
            from train_reversible_history_coordinates import INTERVENTION
            assert receipt['reversible_history_coordinates'] == payload['start']['reversible_history_coordinates'] == INTERVENTION
            assert payload['model_state_sha256'] == dense.dr1.model_state_sha256(payload['model_state_dict'])
        config = dict(payload['model_config'])
        assert config.pop('architecture') == 'ensemble_unet'
        receipts[label], payloads[label], configs[label] = receipt, payload, EnsembleUNet2dConfig(**config)
    assert configs['control'].point_feature_refiner_channels == 0
    expected = (replace(configs['control'], member_adapter_channels=64) if member_teacher
                else replace(configs['control'], point_feature_refiner_channels=32))
    if canonical_stem or reversible_history:
        expected = configs['control']
    assert configs['candidate'] == expected
    assert receipts['control']['fit_files'] == receipts['candidate']['fit_files']
    assert payloads['control']['normalizer_sha256'] == payloads['candidate']['normalizer_sha256'] == '641e7fdfa05af5fefc195a8d7906e0cd44796e66b7d3cc4fac42ca99eabf7b1d'
    rows = ev.clean_rows(args.manifest)
    assert receipts['control']['fit_files'] == [r['file'] for r in rows if r['aoa'] != 15]
    test = [r for r in rows if r['aoa'] == 15]
    scorer = ev.import_file(ev.SCORER, 'shared_point_scorer')
    keys = ev.METRICS[:3]
    functions = (lambda p, y: scorer.rel_l2_per_sample(p, y, 2),
                 lambda p, y: scorer.tke_rel_l2_per_sample(p, y, 2), scorer.mvpe_rel_l2_per_sample)
    results = {}
    for label in ('control', 'candidate'):
        payload = payloads.pop(label)
        model = EnsembleUNet2d(configs[label]).float()
        if canonical_stem and label == 'candidate':
            from models.canonical_fluctuation_stem import install
            install(model)
        if reversible_history and label == 'candidate':
            from models.reversible_history_coordinates import install
            norm, _ = prefix.normalizer_from_point(payload, torch.device('cpu'))
            install(model, norm)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        norm, _ = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
        model.cuda().eval()
        hashes = {k: hashlib.sha256() for k in ('input', 'truth')}
        errors = {r['file']: [] for r in test}
        with torch.inference_mode():
            for history, truth, labels in ev.tail_batches(test, args.real_dir):
                hashes['input'].update(history.tobytes()); hashes['truth'].update(truth.tobytes())
                x, _ = norm.preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
                p = model(x)
                zeros = torch.zeros_like(p[..., :2])
                p = held.absolute_distribution(p, zeros, zeros, norm)[0].cpu().numpy()
                assert np.isfinite(p).all() and np.all(p[..., 2] == 0)
                values = np.stack([fn(p, truth) for fn in functions], axis=-1)
                for file, value in zip(labels, values):
                    errors[file].append(value.tolist())
        def metrics(values):
            means = np.asarray(values, dtype=np.float64).mean(axis=0)
            return {key: scorer.score_error(value) for key, value in zip(keys, means)}
        result = dict(point_metrics=metrics([v for values in errors.values() for v in values]),
                      trajectory_metrics={file: metrics(values) for file, values in errors.items()},
                      errors=errors, hashes={k:v.hexdigest() for k,v in hashes.items()})
        results[label] = result
        del payload, model, norm
        gc.collect(); torch.cuda.empty_cache()
    assert results['control']['hashes'] == results['candidate']['hashes']
    assert sum(len(x) for x in results['control']['errors'].values()) == 153
    if member_teacher or canonical_stem or reversible_history:
        expected_scores = (95.00859412090676,80.16879759178723,95.617979435169)
        assert max(abs(results['control']['point_metrics'][key]-value) for key,value in zip(keys,expected_scores))<=1e-5
    def contribution(point):
        return ev.fixed_time_final({**point, 'sps_score': 43.2953289552})
    changes = {k: results['candidate']['point_metrics'][k]-results['control']['point_metrics'][k] for k in keys}
    point_delta = contribution(results['candidate']['point_metrics']) - contribution(results['control']['point_metrics'])
    trajectory_deltas = {file: contribution(results['candidate']['trajectory_metrics'][file])-contribution(value)
                         for file,value in results['control']['trajectory_metrics'].items()}
    positive = sum(v >= 0 for v in trajectory_deltas.values())
    passed = point_delta >= .05 and changes[keys[1]] >= .20 and changes[keys[0]] >= -.03 and changes[keys[2]] >= -.03 and positive/len(trajectory_deltas) >= .70
    if member_teacher or reversible_history:
        passed = point_delta >= .08 and min(changes.values())>=-.03 and positive/17>=.70
    decision = (('PASS_' if passed else 'STOP_')+'REAL_MEMBER_TEACHER_HELD15' if member_teacher
                else 'PASS_POINT_HELD15' if passed else 'STOP_SHARED_POINT_HELD15')
    if canonical_stem:
        decision = ('PASS_' if passed else 'STOP_') + 'CANONICAL_STEM_HELD15'
    if reversible_history:
        decision = ('PASS_' if passed else 'STOP_') + 'REVERSIBLE_HISTORY_HELD15'
    result = dict(decision=decision,
                  changes=changes, point_fitted_delta=point_delta, nonnegative_trajectories=positive,
                  trajectory_count=len(trajectory_deltas), trajectory_deltas=trajectory_deltas,
                  results=results, receipts=receipts, source_sha256=held.sha256(__file__),
                  scope='Same-machine paired Real-stage held15; shared Sim saw held-angle prefixes. SPS and Time are fixed arithmetic inputs, not measured improvements.')
    (args.root / 'RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('decision','changes','point_fitted_delta','nonnegative_trajectories')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root','manifest','real-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
