"""Replay fixed UQ confirmation weights through actual 0908-runtime ZIPs."""
import argparse
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
for directory in ('evaluation', 'submissions', 'training'):
    sys.path.insert(0, str(ROOT / 'code' / directory))
import evaluate_b210_transfer as ev
import train_b210_held_residual_uq as held
from build_track1_blended_ensemble_unet2d_package import compact_state_dict
from package_track1_combined_uq_quality import equal


def main(plan_path):
    plan = json.loads(plan_path.read_text())
    root = Path(plan['output'])
    root.mkdir(exist_ok=False)
    torch, _, native, _ = held.runtime()
    from models.ensemble_unet2d import EnsembleUNet2dConfig
    torch.set_num_threads(4)
    ev.TIME_SCORE = 92.03266035232576
    scorer = ev.import_file(ev.SCORER, 'uq_deployment_scorer')
    assert ev.sha256(plan['baseline_zip']) == 'b8938d06c100cc68bb4b7e310de112754a9053b4d2834134cb2b4e6f4b5c2a35'
    with zipfile.ZipFile(plan['baseline_zip']) as archive:
        infos = archive.infolist()
        files = {i.filename: archive.read(i) for i in infos}
    template = torch.load(io.BytesIO(files['ensemble_unet2d.pth']), map_location='cpu', weights_only=False)
    assert set(files) == {*ev.RUNTIME_FILES, 'ensemble_unet2d.pth'}
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
    begun = time.monotonic()
    for pair in plan['pairs']:
        angle = pair['angle']
        folder = root / f'held{angle}'
        (folder / 'uq').mkdir(parents=True)
        control = held.read_bound(pair['control_receipt'], pair['control_receipt_sha256'])
        uq = held.read_bound(pair['uq_receipt'], pair['uq_receipt_sha256'])
        assert uq['held_aoa'] == control['held_aoa'] == angle and uq['compact_uq'] and uq['point_state_bitwise']
        assert uq['point_receipt_sha256'] == control['point_receipt_sha256']
        control_path = pair.get('control_checkpoint_copy', control['checkpoint_path'])
        uq_path = pair.get('uq_checkpoint_copy', uq['checkpoint_path'])
        assert ev.sha256(uq_path) == uq['checkpoint_sha256']
        assert ev.sha256(control_path) == control['checkpoint_sha256']
        base = torch.load(control_path, map_location='cpu', weights_only=False)
        patch = torch.load(uq_path, map_location='cpu', weights_only=False)
        assert equal(base['normalizer'], patch['normalizer'])
        def config(value):
            value = dict(value)
            assert value.pop('architecture') == 'ensemble_unet'
            return dataclasses.asdict(EnsembleUNet2dConfig(**value))
        assert config(base['model_config']) == config(patch['model_config']) == config(template['model_config'])
        compact, _ = compact_state_dict(base['model_state_dict'], 1024)
        uq_keys = {k for k in compact if k.startswith(native.UQ_PREFIXES)}
        assert uq_keys == set(patch['model_state_dict'])
        candidate = {**compact, **{k: v.half() for k, v in patch['model_state_dict'].items()}}
        assert all(equal(v, candidate[k]) for k, v in compact.items() if k not in uq_keys)
        states, zips, outputs = {}, {}, {}
        with tempfile.TemporaryDirectory(prefix=f'fold{angle}_', dir=root) as scratch:
            scratch = Path(scratch)
            for label, state in [('baseline', compact), ('candidate', candidate)]:
                payload = {**template, 'normalizer': base['normalizer'], 'quantized_state_dict': state}
                buffer = io.BytesIO()
                torch.save(payload, buffer)
                path = scratch / (label + '.zip')
                with zipfile.ZipFile(path, 'x') as archive:
                    for info in infos:
                        archive.writestr(info, buffer.getvalue() if info.filename == 'ensemble_unet2d.pth' else files[info.filename])
                runtime = scratch / label
                runtime.mkdir()
                with zipfile.ZipFile(path) as archive:
                    assert archive.testzip() is None
                    assert all(archive.read(name) == files[name] for name in ev.RUNTIME_FILES)
                    archive.extractall(runtime)
                zips[label] = {'sha256': ev.sha256(path), 'bytes': path.stat().st_size}
                output = folder / (label + '.quality.json')
                command = [sys.executable, str(Path(ev.__file__)), '--execute', '--worker-runtime', str(runtime),
                           '--worker-label', label, '--worker-output', str(output), '--device', 'cuda:0',
                           '--study', 'real-reflection', '--held-aoa', str(angle),
                           '--clean-manifest', plan['clean_manifest'], '--real-dir', plan['real_dir']]
                assert time.monotonic() - begun < 580
                assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
                with (folder / (label + '.log')).open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120, env=env)
                data = json.loads(output.read_text())
                outputs[label] = data
                states[label] = {'by_trajectory': data['trajectory_states'],
                                 'metrics': ev.finalize(scorer, data['pooled_state']),
                                 'hashes': {k: data[k] for k in ('input_sha256', 'target_sha256', 'scorer_sha256')}}
                del buffer, payload
                path.unlink()
                shutil.rmtree(runtime)
        assert states['baseline']['hashes'] == states['candidate']['hashes']
        point_metric_delta = {k: states['candidate']['metrics'][k] - states['baseline']['metrics'][k] for k in ev.METRICS[:3]}
        assert max(abs(v) for v in point_metric_delta.values()) <= 1e-4
        shutil.copyfile(pair['uq_receipt'], folder / 'uq/uq_endpoint.json')
        result = {'held_aoa': angle, 'decision': 'COHORT_FOLD_RECORDED', 'confirmation_cohort': True,
                  'uq_reflection_combination': True, 'deployment_confirmation': True, 'results': states,
                  'candidate_receipt_sha256': pair['uq_receipt_sha256'], 'zips': zips,
                  'prediction_hashes': {label: data['prediction_sha256'] for label, data in outputs.items()},
                  'non_uq_packed_state_bitwise': True, 'source_sha256': ev.sha256(__file__)}
        result['point_metric_delta'] = point_metric_delta
        result['deployment_point_metric_tolerance'] = 1e-4
        result['execution_gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
        (folder / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
        del base, patch, compact, candidate, state
        print(json.dumps({'sealed_angle': angle, 'elapsed_seconds': time.monotonic() - begun}), flush=True)
    (root / 'COHORT_DONE.json').write_text(json.dumps({'sealed_angles': [p['angle'] for p in plan['pairs']],
        'source_commit': '160c229d007ebb39a70800fd4762c637fb7d0557', 'deployment_confirmation': True,
        'finished_epoch': time.time()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    main(parser.parse_args().plan)
