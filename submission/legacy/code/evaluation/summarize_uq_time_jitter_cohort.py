"""Merge the preregistered three-angle or five-angle UQ sampling cohort."""
import argparse
import json
from pathlib import Path

import evaluate_b210_transfer as ev


def summarize(root, manifest, scope):
    angles = (0, 5, 10) if scope == 'three' else (0, 5, 10, 15, 20)
    ev.TIME_SCORE = 92.03266035232576
    scorer = ev.import_file(ev.SCORER, 'jitter_cohort_scorer')
    rows = [r for r in ev.clean_rows(manifest) if r['aoa'] in angles]
    states = {'baseline': {}, 'candidate': {}}
    inputs, by_fold = {}, {}
    for angle in angles:
        folder = root / f'fulltime_jitter_stage{angle}'
        path = folder / 'RESULT.json'
        result = json.loads(path.read_text())
        receipt = folder / 'uq/uq_endpoint.json'
        assert ev.sha256(receipt) == result['candidate_receipt_sha256']
        bound = json.loads(receipt.read_text())
        assert bound['point_state_bitwise'] and bound['access_audit']['held_arrays_read'] == 0
        assert bound['sampling_intervention']['support'] == 'full_fit_trajectories'
        assert result['held_aoa'] == angle and result['full_time']
        assert result['point_score_max_abs'] <= 1e-5
        for name in ('input', 'truth'):
            assert result['results']['baseline']['hashes'][name] == result['results']['candidate']['hashes'][name]
        expected = {r['file']: len(r['val_starts']) for r in rows if r['aoa'] == angle}
        metrics = {}
        for label in states:
            raw = result['results'][label]['by_trajectory']
            assert {name: value['samples'] for name, value in raw.items()} == expected
            assert not set(raw) & set(states[label])
            states[label].update(raw)
            metrics[label] = ev.finalize(scorer, ev.merge_states(list(raw.values())))
            assert max(abs(metrics[label][k] - result['results'][label]['metrics'][k]) for k in metrics[label]) <= 1e-9
        by_fold[str(angle)] = {k: metrics['candidate'][k] - metrics['baseline'][k] for k in metrics['baseline']}
        inputs[str(path)] = ev.sha256(path)
        inputs[str(receipt)] = ev.sha256(receipt)
    metrics = {label: ev.finalize(scorer, ev.merge_states(list(raw.values()))) for label, raw in states.items()}
    delta = {k: metrics['candidate'][k] - metrics['baseline'][k] for k in metrics['baseline']}
    positive = sum(ev.finalize(scorer, states['candidate'][name])['fixed_time_fitted_final'] >=
                   ev.finalize(scorer, states['baseline'][name])['fixed_time_fitted_final'] for name in states['baseline'])
    finals = [v['fixed_time_fitted_final'] for v in by_fold.values()]
    if scope == 'three':
        passed = delta['sps_score'] >= .15 and delta['fixed_time_fitted_final'] >= .035 and min(finals) > 0 and positive >= .7 * len(rows)
    else:
        passed = delta['sps_score'] >= .20 and delta['fixed_time_fitted_final'] >= .045 and sum(v > 0 for v in finals) >= 4 and min(finals) >= -.01
    return {'decision': ('PASS_' if passed else 'STOP_') + scope.upper() + '_ANGLE_JITTER_CONFIRMATION',
            'scope': scope, 'delta': delta, 'metrics': metrics, 'by_fold': by_fold,
            'positive_trajectories': positive, 'trajectories': len(rows),
            'samples': sum(s['samples'] for s in states['baseline'].values()), 'inputs': inputs,
            'evidence_class': 'Real/UQ stage holdout; shared Sim has Real-teacher ancestry',
            'selection_note': 'Three new angles confirm a modest signal; old large-effect STOPs remain. Five-angle result includes exploratory angle15.',
            'official_gain': None, 'source_sha256': ev.sha256(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--scope', choices=('three', 'five'), required=True)
    args = parser.parse_args()
    result = summarize(args.root, args.manifest, args.scope)
    (args.root / f'{args.scope}_angle_confirmation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('decision', 'delta', 'by_fold', 'positive_trajectories', 'trajectories', 'samples')}, indent=2))
