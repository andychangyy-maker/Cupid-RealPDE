"""Actual 0908/candidate ZIP quality replay and fixed ABBA cost guard."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import zipfile

import evaluate_b210_transfer as ev


def main(args):
    assert ev.sha256(args.baseline) == 'b8938d06c100cc68bb4b7e310de112754a9053b4d2834134cb2b4e6f4b5c2a35'
    package = json.loads(args.candidate.with_suffix('.package.json').read_text())
    assert package['zip_sha256'] == ev.sha256(args.candidate)
    assert package['runtime_members_byte_identical'] and package['non_uq_packed_state_bitwise']
    ev.TIME_SCORE = 92.03266035232576
    scorer = ev.import_file(ev.SCORER, 'combined_release_scorer')
    args.output.mkdir(exist_ok=False)
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
    deadline = time.monotonic() + 1000

    def run(command, name, cap):
        assert time.monotonic() < deadline
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        with (args.output / (name + '.log')).open('w') as log:
            cp = subprocess.run([sys.executable] + command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=cap)
        assert cp.returncode == 0, name
        print(json.dumps({'completed': name}), flush=True)

    qualities, zips = {}, {'baseline': args.baseline, 'candidate': args.candidate}
    for label, path in zips.items():
        runtime = args.output / label
        runtime.mkdir()
        with zipfile.ZipFile(path) as archive:
            assert set(archive.namelist()) == {*ev.RUNTIME_FILES, 'ensemble_unet2d.pth'}
            archive.extractall(runtime)
        target = args.output / (label + '.quality.json')
        run([str(Path(ev.__file__)), '--execute', '--worker-runtime', str(runtime), '--worker-label', label,
             '--worker-output', str(target), '--device', 'cuda:0', '--study', 'real-reflection',
             '--clean-manifest', str(args.clean_manifest), '--real-dir', str(args.real_dir)], label + '.quality', 240)
        qualities[label] = json.loads(target.read_text())
    baseline, candidate = qualities['baseline'], qualities['candidate']
    assert all(baseline[k] == candidate[k] for k in ('input_sha256', 'target_sha256', 'scorer_sha256'))
    assert baseline['pooled_state']['samples'] == candidate['pooled_state']['samples'] == 716
    metrics = {label: ev.finalize(scorer, data['pooled_state']) for label, data in qualities.items()}
    delta = {k: metrics['candidate'][k] - metrics['baseline'][k] for k in metrics['baseline']}
    assert all(abs(delta[k]) <= 1e-4 for k in ev.METRICS[:3])
    by_angle = {a: candidate['by_aoa'][a]['sps_score'] - baseline['by_aoa'][a]['sps_score'] for a in baseline['by_aoa']}
    quality_pass = delta['sps_score'] > 0 and sum(d >= 0 for d in by_angle.values()) >= 4
    timing = []
    for index, label in enumerate(('baseline', 'candidate', 'candidate', 'baseline')):
        target = args.output / f'timing{index}_{label}.json'
        run([str(Path(__file__).with_name('profile_track1_zip_runtime.py')), '--zip', str(zips[label]),
             '--input', str(args.input), '--out', str(target), '--single-call', '--samples', '716'], target.stem, 120)
        timing.append({**json.loads(target.read_text()), 'label': label})
    medians = {label: statistics.median(r['all_predict_seconds'] for r in timing if r['label'] == label) for label in zips}
    cost_change = medians['candidate'] / medians['baseline'] - 1
    passed = quality_pass and cost_change <= .02
    report = {'decision': 'PASS_QUALITY_ZIP_READY_FOR_USER_SUBMISSION' if passed else 'STOP_QUALITY_ZIP_DEPLOYMENT',
              'quality_pass': quality_pass, 'cost_pass': cost_change <= .02, 'delta': delta, 'by_angle_sps_delta': by_angle,
              'metrics': metrics, 'median_predict_seconds': medians, 'relative_cost_change': cost_change,
              'point_output_bitwise': baseline['prediction_sha256'] == candidate['prediction_sha256'],
              'point_metric_tolerance': 1e-4, 'non_uq_packed_state_bitwise': True,
              'timing_prediction_hashes': [r['output_sha256']['prediction'] for r in timing],
              'zips': {label: {'path': str(p), 'sha256': ev.sha256(p)} for label, p in zips.items()},
              'evidence_scope': 'clean716 overlaps full training; deployment diagnostic only. Primary evidence is separately sealed 0/5/10/20 confirmation.',
              'official_score_prediction': False, 'timing_scope': 'A800, fixed716, fresh processes, ABBA single-call; no official hidden-evaluator timing',
              'source_sha256': ev.sha256(__file__)}
    (args.output / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('baseline', 'candidate', 'clean-manifest', 'real-dir', 'input', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
