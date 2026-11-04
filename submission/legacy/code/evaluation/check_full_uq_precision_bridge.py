"""Compare sealed full-data UQ pairs before and after deployment, without training."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile


def main(args):
    for folder in ('training', 'evaluation'):
        sys.path.insert(0, str(args.source / 'code' / folder))
    import train_b210_held_residual_uq as held
    torch, dense, native, prefix = held.runtime()
    import numpy as np
    import evaluate_b210_transfer as ev
    from models.ensemble_unet2d import EnsembleUNet2d

    begun = time.monotonic()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    ev.TIME_SCORE = 92.03266035232576
    sources = {
        args.baseline_checkpoint: '11ea6fe89ca23b2f05c91cd5d00e121db5313331e2d86fc484aecf8bb1027902',
        args.candidate_checkpoint: '26f56335b8287c2ca16d5ce759bc16fe28ca1791a50d3e1732f3cfc33c140fee',
        args.baseline_zip: 'b8938d06c100cc68bb4b7e310de112754a9053b4d2834134cb2b4e6f4b5c2a35',
        args.candidate_zip: '756b9328ba92d706fe8ec994d965f56191cd417fba287bf729304ae50cce94c7',
        args.manifest: ev.MANIFEST_SHA,
        ev.SCORER: 'a144853b1bc1ff79bb8d40601629f23460ac12af95678577e9a1b59949294d39',
    }
    assert all(ev.sha256(p) == sha for p, sha in sources.items())
    scorer = ev.import_file(ev.SCORER, 'full_precision_bridge_scorer')
    rows = ev.clean_rows(args.manifest)
    assert len(rows) == 81
    baseline = torch.load(args.baseline_checkpoint, map_location='cpu', weights_only=False)
    patch = torch.load(args.candidate_checkpoint, map_location='cpu', weights_only=False)
    assert baseline['compact_uq'] and patch['compact_uq']
    assert baseline['source_checkpoint_sha256'] == '46f493e84d991b94620b4caba4be6169d3e7d38032dd44a1f398cb20143b3ed8'
    assert patch['point_receipt']['checkpoint_sha256'] == 'dbe5d9514361a390a0523b330240f8bf2572fd51ef92a495d93e730f5abb4656'
    point_path = Path(baseline['point_receipt']['checkpoint_path'])
    point_file_sha = patch['point_receipt']['checkpoint_sha256']
    assert baseline['point_receipt']['checkpoint_sha256'] == ev.sha256(point_path) == point_file_sha
    sources[point_path] = point_file_sha
    point_payload = torch.load(point_path, map_location='cpu', weights_only=False)
    assert point_payload['model_config'] == baseline['model_config'] == patch['model_config']
    model = EnsembleUNet2d(dense.ft12.validate_b210_model_config(baseline['model_config'])).float()
    model.load_state_dict({**point_payload['model_state_dict'], **baseline['model_state_dict']}, strict=True)
    del point_payload
    point_sha = native.dr1.model_state_sha256(native.partition_state(model, uq=False))
    assert point_sha == 'd035c6b811d97f2009ffee67ec19169d28d30c2c459bbf6d42457bf5675bce23'
    assert set(patch['model_state_dict']) == {k for k in model.state_dict() if k.startswith(native.UQ_PREFIXES)}
    assert set(baseline['model_state_dict']) == set(patch['model_state_dict'])
    normalizer, report = prefix.normalizer_from_point(
        {**baseline, 'normalizer_sha256': 'c87f669fdf6cc6a67a610cae414c210e5927782dc252ce65f540c2241a454d3a'},
        torch.device('cuda:0'))
    _, patch_normalizer = prefix.normalizer_from_point(patch, torch.device('cpu'))
    assert report['sha256'] == patch_normalizer['sha256'] == 'c87f669fdf6cc6a67a610cae414c210e5927782dc252ce65f540c2241a454d3a'
    model.cuda().eval()
    results = {}
    for label in ('fp32_baseline', 'fp32_candidate'):
        if label.endswith('candidate'):
            model.load_state_dict({**model.state_dict(), **patch['model_state_dict']}, strict=True)
            assert native.dr1.model_state_sha256(native.partition_state(model, uq=False)) == point_sha
        raw = {r['file']: [] for r in rows}
        hashes = {key: hashlib.sha256() for key in ('input', 'target', 'prediction')}
        with torch.inference_mode():
            for history, truth, labels in ev.tail_batches(rows, args.real_dir):
                assert time.monotonic() - begun < 1000, 'evaluation time budget exceeded'
                hashes['input'].update(history.tobytes())
                hashes['target'].update(truth.tobytes())
                x, _ = normalizer.preprocess(torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda())
                point, lower, upper = model.forward_distribution(x)
                p, lo, hi = [v.cpu().numpy() for v in held.absolute_distribution(point, lower, upper, normalizer)]
                hashes['prediction'].update(p.tobytes())
                for filename in dict.fromkeys(labels):
                    selected = np.asarray([v == filename for v in labels])
                    raw[filename].append(ev.metric_state(scorer, p[selected], truth[selected], lo[selected], hi[selected]))
        raw = {k: ev.merge_states(v) for k, v in raw.items()}
        pooled = ev.merge_states(list(raw.values()))
        assert pooled['samples'] == 716
        results[label] = {'metrics': ev.finalize(scorer, pooled), 'pooled_state': pooled,
                          'trajectory_states': raw, 'scorer_sha256': ev.sha256(ev.SCORER),
                          **{k + '_sha256': v.hexdigest() for k, v in hashes.items()}}
        held.write_json(args.output / (label + '.json'), results[label])
        print(json.dumps({'sealed': label, 'seconds': time.monotonic() - begun}), flush=True)
    del model, baseline, patch, normalizer
    gc.collect()
    torch.cuda.empty_cache()
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
           'PYTHONDONTWRITEBYTECODE': '1'}
    for label, path in (('zip_baseline', args.baseline_zip), ('zip_candidate', args.candidate_zip)):
        with tempfile.TemporaryDirectory(prefix='track1_precision_bridge_', dir='/dev/shm') as runtime:
            with zipfile.ZipFile(path) as archive:
                assert set(archive.namelist()) == {*ev.RUNTIME_FILES, 'ensemble_unet2d.pth'}
                archive.extractall(runtime)
            output = args.output / (label + '.json')
            command = [sys.executable, str(Path(ev.__file__)), '--execute', '--worker-runtime', runtime,
                       '--worker-label', label, '--worker-output', str(output), '--device', 'cuda:0',
                       '--study', 'real-reflection', '--clean-manifest', str(args.manifest), '--real-dir', str(args.real_dir)]
            with (args.output / (label + '.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240, env=env)
            results[label] = json.loads(output.read_text())
        print(json.dumps({'sealed': label, 'seconds': time.monotonic() - begun}), flush=True)
    for key in ('input_sha256', 'target_sha256', 'scorer_sha256'):
        assert len({d[key] for d in results.values()}) == 1, key
    cases = {}
    for case, selected in (('all716', rows), ('same563_without15', [r for r in rows if r['aoa'] != 15])):
        metrics = {label: ev.finalize(scorer, ev.merge_states([d['trajectory_states'][r['file']] for r in selected]))
                   for label, d in results.items()}
        deltas = {precision: {k: metrics[precision + '_candidate'][k] - metrics[precision + '_baseline'][k]
                             for k in metrics[precision + '_baseline']} for precision in ('fp32', 'zip')}
        for precision, tolerance in (('fp32', 1e-5), ('zip', 1e-4)):
            assert max(abs(deltas[precision][k]) for k in ev.METRICS[:3]) <= tolerance
        cases[case] = {'metrics': metrics, 'deltas': deltas,
                       'sps_deployment_gain_loss': deltas['fp32']['sps_score'] - deltas['zip']['sps_score']}
    loss = cases['all716']['sps_deployment_gain_loss']
    result = {'decision': 'MATERIAL_DEPLOYMENT_GAIN_LOSS' if loss > .10 else 'NO_MATERIAL_DEPLOYMENT_GAIN_LOSS',
              'gate_sps_loss': .10, 'cases': cases, 'seconds': time.monotonic() - begun,
              'point_frozen_sha256': point_sha, 'sources': {str(p): sha for p, sha in sources.items()},
              'source_sha256': ev.sha256(__file__), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'float32_matmul_precision': torch.get_float32_matmul_precision(),
              'runtime_source_root': str(args.source),
              'runtime_source_sha256': {str(Path(m.__file__).relative_to(args.source)): ev.sha256(m.__file__)
                                        for m in (ev, held, native, prefix)},
              'scope': 'Same sealed full-data weights and clean716; overlaps training; no new model, tuning, timing claim, or official forecast.'}
    held.write_json(args.output / 'RESULT.json', result)
    print(json.dumps({'decision': result['decision'], 'sps_deployment_gain_loss': loss}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'baseline-checkpoint', 'candidate-checkpoint', 'baseline-zip', 'candidate-zip', 'manifest', 'real-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
