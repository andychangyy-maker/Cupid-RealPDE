"""Fixed held20/0 point comparison after all four full training endpoints seal."""
import argparse,gc,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace


def main(a):
    sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'training')]
    import numpy as np
    import torch
    import evaluate_b210_transfer as ev
    import train_b210_full_prefix_uq_refit as prefix
    from models.ensemble_unet2d import EnsembleUNet2d,EnsembleUNet2dConfig
    from models.physical_units_b210 import install
    plan=json.loads((a.root/'units_plan.json').read_text())
    assert ev.sha256(a.reference)==plan['mature_c_reference']['sha256']
    reference=json.loads(a.reference.read_text())
    seal=json.loads(a.seals.read_text())
    assert set(seal)=={'control_20','control_0','candidate_20','candidate_0'}
    for name,r in seal.items():
        arm,angle=name.split('_')
        assert r['schema']=='physical_units_b210_real4000_v1' and r['held_aoa']==int(angle)
        assert r['updates']==4000 and r['seed']==4 and not r['held_data_read']
        assert r['representation']==('physical_units_b210_v1' if arm=='candidate' else 'original_global_coordinates')
    for angle in ('20','0'):
        control,candidate=seal['control_'+angle],seal['candidate_'+angle]
        for key in ('fit_files','sample_stream_sha256','normalizer_statistics_sha256'):
            assert control[key]==candidate[key],key
    assert json.loads((a.root/'queue_status.json').read_text())['status']=='ALL_POINT_ENDPOINTS_SEALED_AWAIT_PAIRED_EVALUATION'
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high');ev.TIME_SCORE=91.8606318828
    scorer=ev.import_file(ev.SCORER,'physical_units_point_scorer')
    allrows=ev.clean_rows(Path(plan['manifest']))
    results={}
    for angle in ('20','0'):
        r=seal[a.arm+'_'+angle]
        own=a.root/a.arm/('held'+angle)/'point_endpoint.json'
        assert json.loads(own.read_text())==r
        assert ev.sha256(r['checkpoint_path'])==r['checkpoint_sha256']
        payload=torch.load(r['checkpoint_path'],map_location='cpu',weights_only=False)
        config=dict(payload['model_config']);assert config.pop('architecture')=='ensemble_unet'
        model=EnsembleUNet2d(EnsembleUNet2dConfig(**config)).float()
        if a.arm=='candidate':install(model,SimpleNamespace(**payload['normalizer']))
        model.load_state_dict(payload['model_state_dict'],strict=True)
        normalizer,normal_report=prefix.normalizer_from_point(payload,torch.device('cuda:0'))
        model.cuda().eval();del payload
        rows=[r for r in allrows if r['aoa']==int(angle)]
        sums={r['file']:dict(samples=0,rel_error_sum=0.,tke_error_sum=0.,mvpe_error_sum=0.) for r in rows}
        hashes={k:hashlib.sha256() for k in ('input','truth')}
        anatomy=dict(samples=0,normalized_squared_mean_error=0.,normalized_squared_fluctuation_error=0.,
                     normalized_squared_total_error=0.)
        with torch.inference_mode():
            for history,truth,files in ev.tail_batches(rows,Path(plan['real_dir'])):
                hashes['input'].update(history.tobytes());hashes['truth'].update(truth.tobytes())
                x,_=normalizer.preprocess(torch.from_numpy(history).cuda(),torch.from_numpy(truth).cuda())
                pred=normalizer.postprocess_prediction(model(x).float()).cpu().numpy().astype(np.float32)
                pred[...,2]=0
                assert np.isfinite(pred).all() and pred.shape==truth.shape
                residual=pred[...,:2].astype(np.float64)-truth[...,:2]
                mean_error=residual.mean(1,keepdims=True)
                denominator=np.maximum(np.square(truth[...,:2].astype(np.float64)).sum((1,2,3,4)),1e-16)
                anatomy['samples']+=len(truth)
                anatomy['normalized_squared_mean_error']+=float((20*np.square(mean_error).sum((1,2,3,4))/denominator).sum())
                anatomy['normalized_squared_fluctuation_error']+=float((np.square(residual-mean_error).sum((1,2,3,4))/denominator).sum())
                anatomy['normalized_squared_total_error']+=float((np.square(residual).sum((1,2,3,4))/denominator).sum())
                errors=dict(rel_error_sum=scorer.rel_l2_per_sample(pred,truth,2),
                            tke_error_sum=scorer.tke_rel_l2_per_sample(pred,truth,2),
                            mvpe_error_sum=scorer.mvpe_rel_l2_per_sample(pred,truth))
                for name in dict.fromkeys(files):
                    take=np.asarray([v==name for v in files]);sums[name]['samples']+=int(take.sum())
                    for key,values in errors.items():sums[name][key]+=float(np.sum(values[take],dtype=np.float64))
        hashes={k:v.hexdigest() for k,v in hashes.items()}
        assert hashes==reference['folds'][angle]['hashes'],'Different held inputs/targets'
        states={}
        for name,s in sums.items():
            c=reference['folds'][angle]['states']['control'][name]
            assert s['samples']==c['samples']
            states[name]={**c,**s} # Hold C SPS constant: this is point-only contribution, not new UQ scoring.
        results[angle]=dict(states=states,metrics=ev.finalize(scorer,ev.merge_states(list(states.values()))),
                            hashes=hashes,normalizer=normal_report,point_sha256=r['checkpoint_sha256'],
                            error_decomposition=anatomy)
        del model,normalizer;gc.collect();torch.cuda.empty_cache()
    result=dict(arm=a.arm,folds=results,source_sha256=ev.sha256(__file__),seals_sha256=ev.sha256(a.seals),
                reference_sha256=ev.sha256(a.reference),evidence='Fixed SPS/Time point-only, reused held 20/0',official_gain=None)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(arm=a.arm,folds={k:v['metrics'] for k,v in results.items()})))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','seals','reference','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--arm',choices=('control','candidate'),required=True)
    main(p.parse_args())
