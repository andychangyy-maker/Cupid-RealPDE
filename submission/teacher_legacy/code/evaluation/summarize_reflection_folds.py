"""Apply the preregistered reflection gates to pooled raw scorer states."""
import argparse
import json
from pathlib import Path
import evaluate_b210_transfer as replay

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation', type=Path, nargs='+', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
folds = [json.loads(path.read_text()) for path in a.evaluation]
angles = [x['held_aoa'] for x in folds]
assert len(set(angles)) == len(angles)
assert set(angles) in ({5, 20}, {0, 5, 10, 15, 20})
replay.TIME_SCORE = 92.032660
official = replay.import_file(replay.SCORER, 'reflection_official')
for fold in folds:
    assert fold['study'] == 'real-reflection'
    assert fold['fixed_time_score'] == replay.TIME_SCORE
    assert fold['point_pre_gate_used'] and not fold['official_improvement_claimed']
    for key in ('input_sha256', 'target_sha256', 'scorer_sha256'):
        assert len({v[key] for v in fold['arms'].values()}) == 1
    assert fold['arms']['control']['scorer_sha256'] == replay.sha256(replay.SCORER)
metrics = {arm: replay.finalize(official, replay.merge_states(
    [fold['arms'][arm]['pooled_state'] for fold in folds]))
    for arm in ('control', 'reflection')}
keys = (*replay.METRICS, 'fixed_time_fitted_final')
delta = {k: metrics['reflection'][k] - metrics['control'][k] for k in keys}
by_fold = {str(x['held_aoa']): x['contrasts']['reflection-control'] for x in folds}
finals = [v['fixed_time_fitted_final'] for v in by_fold.values()]
if len(folds) == 2:
    passed = min(finals) > 0 and delta['fixed_time_fitted_final'] >= .08
    passed = passed and min(v['sps_score'] for v in by_fold.values()) >= -.10
    decision = 'PASS_REFLECTION_TWO_FOLD_PROMOTE' if passed else 'STOP_REFLECTION_TWO_FOLD'
else:
    passed = delta['fixed_time_fitted_final'] >= .06 and sum(v > 0 for v in finals) >= 4 and min(finals) >= -.03
    decision = 'PASS_REFLECTION_FIVE_FOLD_FINAL_TRAINING_PENDING' if passed else 'STOP_REFLECTION_FIVE_FOLD'
trajectory_positive = sum(
    f['arms']['reflection']['by_trajectory'][name]['fixed_time_fitted_final'] > v['fixed_time_fitted_final']
    for f in folds for name, v in f['arms']['control']['by_trajectory'].items())
result = {'decision': decision, 'inputs': {str(path): replay.sha256(path) for path in a.evaluation},
          'fixed_time_score': replay.TIME_SCORE, 'metrics': metrics, 'delta': delta, 'by_fold': by_fold,
          'positive_trajectories': trajectory_positive,
          'total_trajectories': sum(len(f['arms']['control']['by_trajectory']) for f in folds),
          'holdout_scope': 'Real/UQ stages only; shared Sim ancestors not held out',
          'official_gain': None, 'release_verified': False}
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
