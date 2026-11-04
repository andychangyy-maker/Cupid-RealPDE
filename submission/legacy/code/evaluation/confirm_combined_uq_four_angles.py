"""Read the fixed four-angle confirmation only after both queues are sealed."""
import argparse
import json
from pathlib import Path

import evaluate_b210_transfer as ev

TRAINING_COMMIT = '160c229d007ebb39a70800fd4762c637fb7d0557'


def main(root, manifest, deployment=False):
    ev.TIME_SCORE = 92.03266035232576
    scorer = ev.import_file(ev.SCORER, 'combined_uq_confirmation_scorer')
    rows = ev.clean_rows(manifest)
    inputs, folds, states = {}, {}, {'baseline': {}, 'candidate': {}}
    queues = [('a800', [0, 20]), ('4080', [5, 10])]
    for card, angles in queues:
        marker = root / card / 'COHORT_DONE.json'
        done = json.loads(marker.read_text())
        assert done['sealed_angles'] == angles and done['source_commit'] == TRAINING_COMMIT
        inputs[str(marker)] = ev.sha256(marker)
    for card, angles in queues:
        for angle in angles:
            folder = root / card / f'held{angle}'
            path, receipt_path = folder / 'RESULT.json', folder / 'uq/uq_endpoint.json'
            result, receipt = [json.loads(p.read_text()) for p in (path, receipt_path)]
            assert result['held_aoa'] == angle and result['decision'] == 'COHORT_FOLD_RECORDED'
            assert result['confirmation_cohort'] and result['uq_reflection_combination']
            if deployment:
                assert result['deployment_confirmation'] and result['non_uq_packed_state_bitwise']
            assert receipt['held_aoa'] == angle and receipt['compact_uq'] and receipt['point_state_bitwise']
            assert result['candidate_receipt_sha256'] == ev.sha256(receipt_path)
            assert result['results']['baseline']['hashes'] == result['results']['candidate']['hashes']
            inputs[str(path)], inputs[str(receipt_path)] = ev.sha256(path), ev.sha256(receipt_path)
            metrics = {}
            for label in states:
                raw = result['results'][label]['by_trajectory']
                assert set(raw) == {r['file'] for r in rows if r['aoa'] == angle}
                assert not set(raw).intersection(states[label])
                states[label].update(raw)
                metrics[label] = ev.finalize(scorer, ev.merge_states(list(raw.values())))
                assert max(abs(metrics[label][k] - result['results'][label]['metrics'][k]) for k in metrics[label]) < 1e-10
            delta = {k: metrics['candidate'][k] - metrics['baseline'][k] for k in metrics['baseline']}
            assert max(abs(delta[k]) for k in ev.METRICS[:3]) <= (1e-4 if deployment else 1e-5)
            folds[angle] = delta
    assert len(states['baseline']) == len(states['candidate']) == 64
    pooled = {label: ev.finalize(scorer, ev.merge_states(list(raw.values()))) for label, raw in states.items()}
    delta = {k: pooled['candidate'][k] - pooled['baseline'][k] for k in pooled['baseline']}
    positive = sum(ev.finalize(scorer, states['candidate'][name])['sps_score'] >=
                   ev.finalize(scorer, states['baseline'][name])['sps_score'] for name in states['baseline'])
    passed = (delta['sps_score'] >= .20 and delta['fixed_time_fitted_final'] >= .045
              and all(d['sps_score'] > 0 for d in folds.values()) and positive >= .70 * 64)
    report = {'decision': 'PASS_COMBINED_UQ_INDEPENDENT_CONFIRMATION_NEEDS_RELEASE' if passed else 'STOP_COMBINED_UQ_INDEPENDENT_CONFIRMATION',
              'delta': delta, 'pooled': pooled, 'by_fold': folds, 'positive_trajectories': positive,
              'trajectory_count': 64, 'samples': ev.merge_states(list(states['baseline'].values()))['samples'],
              'discovery_held15_excluded': True, 'fixed_time_score': ev.TIME_SCORE, 'inputs': inputs,
              'training_commit': TRAINING_COMMIT, 'source_sha256': ev.sha256(__file__),
              'manifest_sha256': ev.sha256(manifest),
              'deployment_confirmation': deployment,
              'scope': 'Real/UQ stage LOAO; shared Sim has Real-teacher ancestry; local fitted score, not official gain'}
    (root / 'CONFIRMATION_RESULT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('decision', 'delta', 'by_fold', 'positive_trajectories', 'samples')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--clean-manifest', type=Path, required=True)
    parser.add_argument('--deployment', action='store_true')
    args = parser.parse_args()
    main(args.root, args.clean_manifest, args.deployment)
