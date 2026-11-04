"""Finish the three preregistered UQ folds and pool all five sealed results."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/root/track1_reflection_confirm_20260909')
first = Path('/root/autodl-tmp/track1_reflection_20260909')
source = Path(__file__).resolve().parents[2]
assert json.loads((root/'two_fold_decision.json').read_text())['decision'] == 'PASS_REFLECTION_TWO_FOLD_PROMOTE'
env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}
negative = 0
evaluations = [first/'uq_stage20/evaluation.json', first/'uq_stage5/evaluation.json']
for aoa in (0, 10, 15):
    deadline = time.monotonic() + 600
    while not (root/f'transfer{aoa}_direct.json').is_file():
        assert time.monotonic() < deadline, 'sealed transfer wait expired'
        time.sleep(5)
    with (root/f'uq{aoa}_queue.log').open('x') as log:
        subprocess.run(['timeout', '--signal=TERM', '--kill-after=15s', '1200s', sys.executable,
            str(source/'code/training/run_reflection_uq_stage.py'), '--root', str(root), '--held-aoa', str(aoa),
            '--source-commit', '923761c4ff0d41ce96386c49107b685e1bf63a01'],
            check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    result = json.loads((root/f'uq_stage{aoa}/decision.json').read_text())
    gain = result['delta']['fixed_time_fitted_final']
    negative += gain <= 0
    if gain < -.03 or negative >= 2:
        (root/'confirmation_status.json').write_text(json.dumps({
            'status': 'TERMINAL', 'decision': 'STOP_REFLECTION_FIVE_FOLD_GATE_UNREACHABLE',
            'held': aoa, 'last_result': result, 'official_gain': None}, indent=2))
        sys.exit(0)
    evaluations.append(root/f'uq_stage{aoa}/evaluation.json')
subprocess.run([sys.executable, str(source/'code/evaluation/summarize_reflection_folds.py'),
    '--evaluation', *map(str, evaluations), '--output', str(root/'five_fold_decision.json')], check=True, env=env)
(root/'confirmation_status.json').write_text(json.dumps({'status': 'TERMINAL',
    **json.loads((root/'five_fold_decision.json').read_text())}, indent=2))
