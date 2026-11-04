"""Pool Q4's own five folds; older reflection results cannot enter this gate."""
import argparse
import json
from pathlib import Path
import evaluate_b210_transfer as replay

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation', type=Path, nargs=5, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
folds = [json.loads(path.read_text()) for path in a.evaluation]
assert sorted(f['held_aoa'] for f in folds) == [0, 5, 10, 15, 20]
replay.TIME_SCORE = 92.032660
official = replay.import_file(replay.SCORER, 'q4_official')
names = []
for path, fold in zip(a.evaluation, folds):
    assert fold['study'] == 'real-reflection-exposure' and fold['fixed_time_score'] == replay.TIME_SCORE
    assert fold['full_fit_development_screen_used'] and not fold['release_verified']
    for key in ('input_sha256', 'target_sha256', 'scorer_sha256'):
        assert len({v[key] for v in fold['arms'].values()}) == 1
    assert fold['arms']['control']['scorer_sha256'] == replay.sha256(replay.SCORER)
    for arm, updates in (('control',4000),('reflection',5334)):
        receipt_path = path.parent/arm/'uq/uq_endpoint.json'
        assert replay.sha256(receipt_path) == fold['bindings'][arm]['receipt_sha256']
        receipt = json.loads(receipt_path.read_text())
        assert receipt['point_receipt']['updates'] == updates and receipt['held_aoa'] == fold['held_aoa']
        assert set(fold['arms'][arm]['by_trajectory']) == set(fold['arms']['control']['by_trajectory'])
    names.extend(fold['arms']['control']['by_trajectory'])
assert len(names) == len(set(names)) == 81
metrics = {arm: replay.finalize(official, replay.merge_states([f['arms'][arm]['pooled_state'] for f in folds]))
           for arm in ('control','reflection')}
assert all(sum(f['arms'][arm]['pooled_state']['samples'] for f in folds)==716 for arm in metrics)
delta = {k:metrics['reflection'][k]-metrics['control'][k] for k in (*replay.METRICS,'fixed_time_fitted_final')}
by_fold = {str(f['held_aoa']):f['delta'] for f in folds}
finals = [d['fixed_time_fitted_final'] for d in by_fold.values()]
passed = delta['fixed_time_fitted_final'] >= .06 and sum(v>0 for v in finals) >= 4 and min(finals) >= -.03
result = {'decision':'PASS_EXPOSURE_FIVE_FOLD_NEEDS_RELEASE' if passed else 'STOP_EXPOSURE_FIVE_FOLD',
          'metrics':metrics,'delta':delta,'by_fold':by_fold,'fixed_time_score':92.032660,
          'samples':716,'trajectories':81,'inputs':{str(path):replay.sha256(path) for path in a.evaluation},
          'devices':{str(f['held_aoa']):{'device':f['device'],'torch_version':f['torch_version']} for f in folds},
          'source_commits':{str(f['held_aoa']):f['source_commit'] for f in folds},
          'scope':'Real/UQ stage confirmation after development screens; shared Sim Real ancestry; same-device pairs',
          'official_gain':None,'release_verified':False}
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
