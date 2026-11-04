"""Decide the preregistered two-angle PC3D confirmation from pooled raw errors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import evaluate_b210_transfer as ev


def main(root):
    ev.TIME_SCORE = 92.0326603523
    results, files, states = {}, {}, {'control': [], 'teacher': []}
    def metrics(errors):
        return dict(zip(ev.METRICS[:3], (100 / (1 + .5 * np.asarray(errors).mean(axis=0))).tolist()))
    def final(values):
        return ev.fixed_time_final({**values, 'sps_score': 43.2953289552})
    for angle in (20, 0):
        path = root / f'held{angle}' / 'RESULT.json'
        value = json.loads(path.read_text())
        assert value['decision'] == f'SEALED_PC3D_CONFIRM_HELD{angle}' and value['held_aoa'] == angle
        files[str(angle)] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for label in states:
            errors = value['results'][label]['errors']
            assert all(name.endswith(f'_{angle}.h5') for name in errors)
            states[label].extend(row for trajectory in errors.values() for row in trajectory)
        delta = final(value['results']['teacher']['point_metrics']) - final(value['results']['control']['point_metrics'])
        assert abs(delta - value['point_fitted_delta']) <= 1e-9
        fraction = value['nonnegative_trajectories'] / value['trajectory_count']
        results[str(angle)] = dict(point_delta=delta, component_deltas=value['changes'],
                                   nonnegative_fraction=fraction,
                                   passed=delta > 0 and min(value['changes'].values()) >= -.03 and fraction >= .60)
    pooled = {label: metrics(values) for label, values in states.items()}
    delta = final(pooled['teacher']) - final(pooled['control'])
    passed = all(v['passed'] for v in results.values()) and delta >= .02
    result = dict(decision=('PASS_' if passed else 'STOP_') + 'PC3D_TWO_ANGLE_CONFIRMATION',
                  fold_results=results, pooled_metrics=pooled, pooled_point_delta=delta,
                  pooled_component_deltas={k: pooled['teacher'][k] - pooled['control'][k] for k in ev.METRICS[:3]},
                  samples=len(states['control']), source_files=files,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='New fixed Real-stage held20 and held0 confirmation; shared C Sim has Real ancestry. The old held15 STOP is unchanged. No student, UQ, Time or official gain is established.')
    (root / 'RESULT.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
