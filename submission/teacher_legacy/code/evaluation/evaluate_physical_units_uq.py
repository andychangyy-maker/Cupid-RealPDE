"""Same-device quality comparison of the sealed physical point plus matched q."""
import argparse,gc,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace


def main(a):
    sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'training')]
    import numpy as np
    import train_b210_held_residual_uq as held
    import evaluate_b210_transfer as ev
    from models.ensemble_unet2d import EnsembleUNet2d,EnsembleUNet2dConfig
    from models.physical_units_b210 import install
    torch,_,_,prefix=held.runtime();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    device=torch.device('cuda:0');ev.TIME_SCORE=91.8606318828
    stem='matched_uq_reference' if a.physical_reference_repair else 'matched_uq'
    uq_folder='uq_reference' if a.physical_reference_repair else 'uq'
    assert json.loads((a.root/(stem+'_status.json')).read_text())['status']=='BOTH_MATCHED_UQ_ENDPOINTS_SEALED'
    plan=json.loads((a.root/(stem+'_plan.json')).read_text());reference=json.loads((a.root/'reference_quality.json').read_text())
    point_reference=json.loads((a.root/'POINTS.json').read_text());assert point_reference['arm']=='candidate'
    scorer=ev.import_file(ev.SCORER,'physical_units_q_scorer');allrows=ev.clean_rows(Path(plan['manifest']))
    seals={}
    for angle in (20,0):
        c=plan['controls'][str(angle)];assert ev.sha256(c['receipt'])==c['receipt_sha256']
        for arm,path in [('control',Path(c['receipt'])),('candidate',a.root/f'candidate/held{angle}/{uq_folder}/uq_endpoint.json')]:
            s=json.loads(path.read_text());assert s['updates']==3200 and s['held_aoa']==angle and s['point_state_bitwise']
            assert ev.sha256(s['checkpoint_path'])==s['checkpoint_sha256'];seals[angle,arm]=s
    folds={};pooled_states={k:{} for k in ('control','candidate','candidate_old_bounds')};positive=total=0;max_point_delta=0.
    for angle in (20,0):
        models={};normalizers={}
        for arm in ('control','candidate'):
            s=seals[angle,arm];p=s['point_receipt'];assert ev.sha256(p['checkpoint_path'])==p['checkpoint_sha256']
            payload=torch.load(p['checkpoint_path'],map_location='cpu',weights_only=False)
            uq=torch.load(s['checkpoint_path'],map_location='cpu',weights_only=False)
            config=dict(payload['model_config']);assert config.pop('architecture')=='ensemble_unet'
            model=EnsembleUNet2d(EnsembleUNet2dConfig(**config)).float()
            if arm=='candidate':
                assert s['representation']=='physical_units_b210_v1'
                assert bool(s.get('uq_physical_reference',False))==a.physical_reference_repair
                install(model,SimpleNamespace(**payload['normalizer']),uq_global=a.physical_reference_repair)
            model.load_state_dict({**payload['model_state_dict'],**uq['model_state_dict']},strict=True)
            normalizer,nr=prefix.normalizer_from_point(payload,device);assert nr['sha256']==s['normalizer_sha256']
            models[arm]=model.cuda().eval();normalizers[arm]=normalizer;del payload,uq
        sa,sb=seals[angle,'control'],seals[angle,'candidate']
        for k in ('sample_stream_sha256','sampling_intervention','physical_reflection'):
            assert sa[k]==sb[k],k
        rows=[r for r in allrows if r['aoa']==angle]
        states={arm:{r['file']:[] for r in rows} for arm in pooled_states};hashes={k:hashlib.sha256() for k in ('input','truth')}
        with torch.inference_mode():
            for history,truth,files in ev.tail_batches(rows,Path(plan['real_dir'])):
                hashes['input'].update(history.tobytes());hashes['truth'].update(truth.tobytes());predictions={}
                for arm,model in models.items():
                    normalizer=normalizers[arm]
                    x,_=normalizer.preprocess(torch.from_numpy(history).cuda(),torch.from_numpy(truth).cuda())
                    predictions[arm]=tuple(t.cpu().numpy() for t in held.absolute_distribution(*model.forward_distribution(x),normalizer))
                predictions['candidate_old_bounds']=(predictions['candidate'][0],*predictions['control'][1:])
                for arm,values in predictions.items():
                    for name in dict.fromkeys(files):
                        take=np.asarray([f==name for f in files]);states[arm][name].append(ev.metric_state(scorer,values[0][take],truth[take],values[1][take],values[2][take]))
        hashes={k:v.hexdigest() for k,v in hashes.items()};assert hashes==reference['folds'][str(angle)]['hashes']
        states={arm:{name:ev.merge_states(v) for name,v in chunks.items()} for arm,chunks in states.items()}
        metrics={arm:ev.finalize(scorer,ev.merge_states(list(s.values()))) for arm,s in states.items()}
        deltas={arm:{k:metrics[arm][k]-metrics['control'][k] for k in metrics['control']} for arm in ('candidate','candidate_old_bounds')}
        point_delta={k:metrics['candidate'][k]-point_reference['folds'][str(angle)]['metrics'][k] for k in ev.METRICS[:3]}
        max_point_delta=max(max_point_delta,*map(abs,point_delta.values()))
        td={n:ev.finalize(scorer,states['candidate'][n])['fixed_time_fitted_final']-ev.finalize(scorer,states['control'][n])['fixed_time_fitted_final'] for n in states['control']}
        positive+=sum(v>=0 for v in td.values());total+=len(td)
        folds[str(angle)]=dict(metrics=metrics,deltas=deltas,trajectory_quality_delta=td,states=states,hashes=hashes,
            candidate_point_replay_delta=point_delta,control_historical_reference_delta={k:metrics['control'][k]-reference['folds'][str(angle)]['metrics']['control'][k] for k in metrics['control']})
        for arm in states:pooled_states[arm].update(states[arm])
        del models,model,normalizers,normalizer;gc.collect();torch.cuda.empty_cache()
    pooled={arm:ev.finalize(scorer,ev.merge_states(list(s.values()))) for arm,s in pooled_states.items()}
    delta={arm:{k:pooled[arm][k]-pooled['control'][k] for k in pooled['control']} for arm in ('candidate','candidate_old_bounds')}
    gates=dict(each_fold=all(f['deltas']['candidate']['rel_l2_score']>0 and f['deltas']['candidate']['fixed_time_fitted_final']>0 for f in folds.values()),
        pooled_sps=delta['candidate']['sps_score']>=0,pooled_quality=delta['candidate']['fixed_time_fitted_final']>=.12,
        trajectories=positive/total>=.6,point_replay=max_point_delta<=1e-5)
    result=dict(status='PASS_UNITS_MATCHED_Q_CONFIRM_5_10_NEXT' if all(gates.values()) else 'STOP_UNITS_MATCHED_Q_GATE',
        gates=gates,folds=folds,pooled=pooled,pooled_delta=delta,positive_trajectories=positive,total_trajectories=total,
        max_candidate_point_delta=max_point_delta,source_sha256=ev.sha256(__file__),plan_sha256=ev.sha256(a.root/(stem+'_plan.json')),
        uq_physical_reference_repair=a.physical_reference_repair,
        official_gain=None,evidence='Held0/20 reused discovery; same-device C vs candidate; historical other-GPU C reference reported separately')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','gates','pooled_delta','positive_trajectories','max_candidate_point_delta')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--physical-reference-repair',action='store_true')
    main(p.parse_args())
