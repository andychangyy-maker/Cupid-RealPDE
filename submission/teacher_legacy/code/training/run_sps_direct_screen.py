"""Execute the preregistered two-angle continuation screen and seal its decision."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/sps_direct_threads_20260909')
STAGE = Path('/root/autodl-tmp/track1_reflection_exposure_stage_20260909')
SCRIPT = ROOT/'train_sps_direct_continuation.py'
begin = time.monotonic()


def run(arguments, log):
    with log.open('x') as handle:
        subprocess.run([sys.executable,str(SCRIPT),*arguments],stdout=handle,
            stderr=subprocess.STDOUT,check=True,timeout=max(1,2400-(time.monotonic()-begin)))


if __name__ == '__main__':
    folds = {}
    try:
        for held in (15,20):
            base = STAGE/f'held{held}'/'reflection'
            folder = ROOT/f'held{held}'
            folder.mkdir()
            for arm in ('control','direct'):
                run(['--base',str(base),'--held',str(held),'--arm',arm,'--output',str(folder/arm)],
                    folder/f'{arm}_train.log')
            receipts = [json.loads((folder/arm/'receipt.json').read_text()) for arm in ('control','direct')]
            assert receipts[0]['sample_stream_sha256'] == receipts[1]['sample_stream_sha256']
            assert receipts[0]['source_checkpoint_sha256'] == receipts[1]['source_checkpoint_sha256']
            values = {}
            for arm in ('baseline','control','direct'):
                runtime = base/'runtime' if arm=='baseline' else folder/arm/'runtime'
                run(['--runtime',str(runtime),'--held',str(held),'--output',str(folder/f'{arm}_scores.json')],
                    folder/f'{arm}_eval.log')
                values[arm] = json.loads((folder/f'{arm}_scores.json').read_text())
            for key in ('input_sha256','truth_sha256'):
                assert len({v[key] for v in values.values()}) == 1
            point = ('rel_l2_score','tke_rel_l2_score','mvpe_rel_l2_score')
            assert all(abs(values[a]['scores'][k]-values['baseline']['scores'][k])<=1e-4
                       for a in ('control','direct') for k in point)
            delta = {a:values['direct']['scores']['sps_score']-values[a]['scores']['sps_score']
                     for a in ('baseline','control')}
            folds[held] = {'values':values,'direct_minus':delta}
            (ROOT/'quality_progress.json').write_text(json.dumps(folds,indent=2))
            print(json.dumps({'held':held,'direct_minus':delta}),flush=True)
            if any(v<=0 for v in delta.values()):
                break
        sys.path.insert(0,'/root/autodl-tmp/q4_stage_source/code/evaluation')
        import evaluate_b210_transfer as ev
        scorer=ev.import_file(Path('/root/autodl-tmp/time_audit_919767_20260909/source/scoring.py'),'direct_merge')
        pooled={a:ev.finalize(scorer,ev.merge_states([f['values'][a]['merged'] for f in folds.values()]))
                for a in ('baseline','control','direct')}
        passed = (len(folds)==2 and all(all(d>0 for d in f['direct_minus'].values()) for f in folds.values())
                  and pooled['direct']['sps_score']-pooled['baseline']['sps_score']>=.30)
        result={'decision':'PROMOTE_STAGE_CONFIRMATION' if passed else 'STOP_DIRECT_SPS_CONTINUATION',
                'folds':folds,'pooled':pooled,'seconds':time.monotonic()-begin}
        (ROOT/'QUALITY_RESULT.json').write_text(json.dumps(result,indent=2))
        print(result['decision'],flush=True)
    except Exception as error:
        (ROOT/'QUALITY_ERROR.json').write_text(json.dumps({'error':repr(error),'seconds':time.monotonic()-begin}))
        raise
