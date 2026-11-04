"""Verify the exact best-weight threads4 ZIP against its unmodified source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/sps_direct_threads_20260909')
BASE=Path('/root/autodl-tmp/time_audit_919767_20260909/official')
INPUT='/root/autodl-tmp/spuq2_a800/artifacts/clean716_input_cache.npy'
TRUTH='/root/autodl-tmp/real_time_release_20260908/time_replay_truth_20260907.npz'


def worker(args):
    import numpy as np
    import torch
    runtime=BASE if args.arm=='baseline' else ROOT/(args.variant+'_runtime')
    sys.path.insert(0,str(runtime))
    import submission
    x=np.load(INPUT,mmap_mode='r')
    if args.mode=='quality':
        sys.path.insert(0,'/root/autodl-tmp/time_audit_919767_20260909/source')
        import evaluate_b210_transfer as ev
        official=ev.import_file(ROOT.parent/'time_audit_919767_20260909/source/scoring.py','thread_scorer')
        with np.load(TRUTH) as truth:
            y,labels=truth['truth'],truth['labels'].astype(str)
        output=submission.predict(x.copy())
        states={s:ev.metric_state(official,output['prediction'][labels==s],y[labels==s],
                output['lower'][labels==s],output['upper'][labels==s]) for s in dict.fromkeys(labels)}
        print(json.dumps({'scores':ev.finalize(official,ev.merge_states(list(states.values()))),
                          'states':states,'threads':torch.get_num_threads()}),flush=True)
        return
    if args.mode=='cpu':
        assert not torch.cuda.is_available()
        calls=(1,)
    else:
        warm=submission.predict(x[:16].copy())
        torch.cuda.synchronize()
        del warm
        calls=(args.n,)
    batch=x[:calls[0]].copy()
    start=time.perf_counter()
    output=submission.predict(batch)
    if torch.cuda.is_available():torch.cuda.synchronize()
    done=time.perf_counter()
    assert set(output)=={'prediction','lower','upper'}
    assert all(v.shape==batch.shape and v.dtype==np.float32 and np.isfinite(v).all()
               and np.all(v[...,2]==0) for v in output.values())
    assert np.all(output['lower']<=output['upper'])
    checks={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in output.items()}
    checksum=sum(float(np.sum(v,dtype=np.float64)) for v in output.values())
    consumed=time.perf_counter()
    later=submission.predict(x[:1].copy())
    assert all(checks[k]==hashlib.sha256(v.tobytes()).hexdigest() for k,v in output.items())
    assert np.isfinite(checksum)
    print(json.dumps({'arm':args.arm,'n':calls[0],'threads':torch.get_num_threads(),
        'predict_seconds':done-start,'consume_seconds':consumed-done,'contract':True,
        'retained_arrays_unchanged':True,'cuda':torch.cuda.is_available()}),flush=True)


def main(variant):
    out=ROOT/('release_check' if variant=='threads4' else 'cap4_release_check');out.mkdir()
    env={k:v for k,v in os.environ.items() if k not in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}
    begun=time.monotonic()
    def call(arm,mode,n,name,extra=None):
        cp=subprocess.run([sys.executable,__file__,'--arm',arm,'--mode',mode,'--n',str(n),'--variant',variant],
            capture_output=True,text=True,env={**env,**(extra or {})},timeout=min(120,max(1,1200-(time.monotonic()-begun))))
        (out/(name+'.log')).write_text(cp.stdout+'\n'+cp.stderr)
        assert cp.returncode==0,cp.stderr
        value=json.loads(cp.stdout)
        if arm=='candidate':assert value['threads']==4
        return value
    cpu=call('candidate','cpu',1,'cpu',{'CUDA_VISIBLE_DEVICES':''})
    quality={arm:[call(arm,'quality',716,f'quality_{arm}_{r}') for r in range(2)]
             for arm in ('baseline','candidate')}
    keys=('rel_l2_score','tke_rel_l2_score','mvpe_rel_l2_score','sps_score')
    deltas=[{k:quality['candidate'][r]['scores'][k]-quality['baseline'][r]['scores'][k] for k in keys} for r in range(2)]
    numerical=all(abs(v)<=1e-4 for row in deltas for v in row.values())
    (out/'quality.json').write_text(json.dumps({'results':quality,'deltas':deltas,'pass':numerical},indent=2))
    assert numerical,'quality numerical gate failed'
    timing=[]
    for forced in (False,True):
        for n in ((1,16,64,716) if not forced else (716,)):
            for repeat in range(3):
                for arm in (('baseline','candidate') if repeat%2==0 else ('candidate','baseline')):
                    row=call(arm,'time',n,f'time_{forced}_{n}_{repeat}_{arm}',
                             {'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'} if forced else None)
                    row.update(forced_environment_threads4=forced,repeat=repeat)
                    timing.append(row)
                    print(json.dumps(row),flush=True)
    ratios={}
    for forced in (False,True):
        for n in ((1,16,64,716) if not forced else (716,)):
            med={a:statistics.median(r['predict_seconds'] for r in timing if r['arm']==a
                and r['n']==n and r['forced_environment_threads4']==forced) for a in ('baseline','candidate')}
            ratios[f'{forced}_{n}']={**med,'candidate_over_baseline':med['candidate']/med['baseline']}
    passed=all(v['candidate_over_baseline'] <= (.95 if k in ('False_64','False_716') else 1.05)
               for k,v in ratios.items())
    result={'decision':'PASS_LOCAL_THREADS4_RELEASE' if passed else 'STOP_THREADS4_RELEASE','variant':variant,
            'quality_deltas':deltas,'timing':timing,'ratios':ratios,'cpu':cpu,'seconds':time.monotonic()-begun,
            'scope':'Exact 918343 weights; local performance only, no official Time guarantee'}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2))
    print(result['decision'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm');p.add_argument('--mode');p.add_argument('--n',type=int)
    p.add_argument('--variant',choices=('threads4','cap4'),default='threads4')
    a=p.parse_args()
    worker(a) if a.arm else main(a.variant)
