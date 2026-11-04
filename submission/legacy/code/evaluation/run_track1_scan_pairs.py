"""Run fixed-order, fresh-process comparisons of the 917211 scan candidate."""
from pathlib import Path
import argparse
import json
import statistics
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--benchmark', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    runs = []
    schedule = [(16, arm) for arm in ('baseline', 'candidate', 'candidate',
                                      'baseline', 'baseline', 'candidate')]
    schedule += [(batch, arm) for batch in (1, 8) for arm in ('baseline', 'candidate')]
    for i, (batch, arm) in enumerate(schedule):
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                          '--format=csv,noheader'], text=True).strip()
        assert not active, 'GPU is occupied: ' + active
        path = out / f'{i:02d}_{arm}_b{batch}.json'
        with path.with_suffix('.log').open('w') as log:
            subprocess.run([sys.executable, a.benchmark, '--zip', getattr(a, arm),
                            '--input', a.input, '--out', str(path),
                            '--batch-size', str(batch)], stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=120)
        result = json.loads(path.read_text())
        runs.append({'arm': arm, 'batch': batch, 'result': result})
        same_batch = [r for r in runs if r['batch'] == batch]
        assert all(r['result']['output_sha256'] == result['output_sha256']
                   for r in same_batch), 'Output mismatch; stop promotion'
        print(json.dumps({'run': i, 'arm': arm, 'batch': batch,
                          'seconds': result['all_predict_seconds'],
                          'hashes_match_same_batch': True}), flush=True)
    median = {arm: statistics.median(r['result']['all_predict_seconds'] for r in runs
                                    if r['arm'] == arm and r['batch'] == 16)
              for arm in ('baseline', 'candidate')}
    summary = {'runs': runs, 'median_b16_seconds': median,
               'latency_reduction': 1 - median['candidate'] / median['baseline'],
               'all_outputs_exact_within_batch': True, 'official_score': None}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'runs'}), flush=True)


if __name__ == '__main__':
    main()
