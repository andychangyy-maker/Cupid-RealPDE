"""Fixed historical ZIP replay, with one scored call per fresh process."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--benchmark', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    manifest = json.loads(Path(a.manifest).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    for item in manifest['zips'].values():
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
    schedule = [(716, arm) for order in (
        ('v2', 'v3', 'baseline', 'candidate'),
        ('candidate', 'baseline', 'v3', 'v2'),
        ('v2', 'v3', 'baseline', 'candidate')) for arm in order]
    schedule += [(n, arm) for n in (16, 63, 64) for arm in (
        'baseline', 'candidate', 'candidate', 'baseline', 'baseline', 'candidate')]
    schedule += [(1, arm) for arm in ('baseline', 'candidate')]
    runs = []
    started = time.monotonic()
    for i, (n, arm) in enumerate(schedule):
        assert time.monotonic() - started < 1800, 'Replay wall budget exhausted'
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                           '--format=csv,noheader'], text=True).strip()
        path = out / f'{i:02d}_{arm}_n{n}.json'
        with path.with_suffix('.log').open('w') as log:
            subprocess.run([sys.executable, a.benchmark, '--zip', manifest['zips'][arm]['path'],
                            '--input', manifest['input'], '--out', str(path),
                            '--single-call', '--samples', str(n)], check=True, timeout=180,
                           stdout=log, stderr=subprocess.STDOUT)
        result = json.loads(path.read_text())
        assert result['single_call'] and len(result['call_seconds']) == 1
        assert result['torch'].startswith('2.2.2') and result['cuda'] == '12.1'
        equivalent = ('baseline', 'candidate') if arm in ('baseline', 'candidate') else (arm,)
        peers = [r for r in runs if r['samples'] == n and r['arm'] in equivalent]
        assert all(r['result']['output_sha256'] == result['output_sha256'] for r in peers), 'Output mismatch'
        runs.append({'arm': arm, 'samples': n, 'result': result})
        (out / 'progress.json').write_text(json.dumps({'runs': runs}, indent=2) + '\n')
        print(json.dumps({'run': i, 'arm': arm, 'N': n, 'predict_seconds': result['all_predict_seconds'],
                          'import_seconds': result['import_seconds']}), flush=True)
    summaries = {}
    for n in (1, 16, 63, 64, 716):
        medians = {arm: statistics.median(r['result']['all_predict_seconds'] for r in runs
                                         if r['samples'] == n and r['arm'] == arm)
                   for arm in sorted({r['arm'] for r in runs if r['samples'] == n})}
        summaries[str(n)] = {'median_predict_seconds': medians,
                             'candidate_latency_reduction': 1 - medians['candidate'] / medians['baseline']}
    result = {'runs': runs, 'summary_by_call_size': summaries, 'all_three_outputs_exact': True,
              'elapsed_seconds': time.monotonic() - started, 'official_score': None,
              'scope': 'Official-version software stack; public input and PCIe hardware, not exact official evaluator'}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}), flush=True)


if __name__ == '__main__':
    main()
