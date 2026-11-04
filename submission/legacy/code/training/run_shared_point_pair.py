"""Run the fixed paired endpoint queue; inspect the holdout only after both seal."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def main(root):
    plan = json.loads((root / 'launch_plan.json').read_text())
    assert plan['budget_seconds'] == 5500
    assert json.loads((root / 'engineering.json').read_text())['decision'] == 'PASS_TO_REAL4000'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    try:
        for label, cap in (('control', 2450), ('candidate', 2450), ('screen', 300)):
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            (root/'progress.json').write_text(json.dumps(dict(phase=label,started_epoch=time.time())))
            with (root/(label+'.log')).open('x') as log:
                subprocess.run(plan['commands'][label],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=cap)
        (root/'completed.json').write_text(json.dumps(dict(finished_epoch=time.time(),decision=json.loads((root/'RESULT.json').read_text())['decision'])))
    except Exception:
        (root/'FAILED.json').write_text(json.dumps(dict(finished_epoch=time.time(),traceback=traceback.format_exc())))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
