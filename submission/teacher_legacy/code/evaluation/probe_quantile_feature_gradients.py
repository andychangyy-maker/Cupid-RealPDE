"""Fixed training-side feature-gradient diagnostic; no optimizer or holdout score."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


def main(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/training'))
    import train_b210_held_residual_uq as held
    torch, dense, native, prefix = held.runtime()
    import evaluate_b210_transfer as ev
    from models.ensemble_unet2d import EnsembleUNet2d, EnsembleUNet2dConfig
    train = dense.ft12.npt1.ensemble_train
    joint = dense.ft12.npt1.joint_real
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    started = time.monotonic()
    receipt = json.loads(args.control_receipt.read_text())
    assert receipt['checkpoint_sha256'] == 'ce84999755b685bfc84061cec3367ee324f64d71661ee5fcfd33320182c5b5fa'
    assert held.sha256(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
    payload = torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=False)
    assert payload['normalizer_sha256'] == '641e7fdfa05af5fefc195a8d7906e0cd44796e66b7d3cc4fac42ca99eabf7b1d'
    norm, _ = prefix.normalizer_from_point(payload, torch.device('cuda:0'))
    all_rows = ev.clean_rows(args.manifest)
    fit = [r for r in all_rows if r['aoa'] != 15]
    assert receipt['fit_files'] == [r['file'] for r in fit]
    selected = []
    for aoa in (0,5,10,20):
        rows = sorted((r for r in fit if r['aoa']==aoa), key=lambda r:r['re'])
        for index in (0,len(rows)//3,2*len(rows)//3,len(rows)-1):
            row = rows[index]
            selected.append({**row,'val_starts':row['val_starts'][:1]})
    history, truth, files = next(ev.tail_batches(selected,args.real_dir))
    assert len(files) == len(set(files)) == 16 and all(r['aoa']!=15 for r in selected)
    histories, truths = torch.from_numpy(history).cuda(), torch.from_numpy(truth).cuda()
    hashes = {key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in [('input',history),('truth',truth)]}
    loss_args = joint.real_loss_args()

    def case(model,index,iteration):
        model.train()
        torch.manual_seed(31012+index)
        raw, target = histories[index:index+1], truths[index:index+1]
        augmented = train.add_training_input_gain(raw,.01)
        x, y = norm.preprocess(augmented,target)
        x = train.masked_velocity_blur(x,raw,.15,.20)
        x = train.add_training_input_noise(x,raw,.01)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        prediction,members,quantiles,lower,upper,features,_ = model.forward_uq_training(x)
        prediction_reference, quantile_reference = prediction.detach().clone(), quantiles.detach().clone()
        point,_ = train.point_loss(loss_args,prediction,y,norm,(5.,3.,2.),iteration)
        member = torch.stack([train.point_loss(loss_args,members[:,j],y,norm,(5.,3.,2.),iteration)[0] for j in range(4)]).mean()
        p = .25*point+member
        q = .05*train.residual_quantile_loss(model,prediction,quantiles,y,norm,physical_target=target)
        pg = torch.autograd.grad(p,tuple(features),retain_graph=True,allow_unused=True)
        qg = torch.autograd.grad(q,tuple(features),allow_unused=True)
        def stats(p2,q2,dot):
            assert p2>0 and q2>=0
            return dict(point_norm=p2**.5,quantile_norm=q2**.5,ratio=(q2/p2)**.5,
                        cosine=dot/max((p2*q2)**.5,1e-30),projection_ratio=dot/p2)
        levels=[];total_p2=total_q2=total_dot=0.
        for first,second,feature in zip(pg,qg,features):
            first = torch.zeros_like(feature) if first is None else first
            second = torch.zeros_like(feature) if second is None else second
            p2,q2,dot = [float(value) for value in ((first*first).sum(),(second*second).sum(),(first*second).sum())]
            levels.append(stats(p2,q2,dot));total_p2+=p2;total_q2+=q2;total_dot+=dot
        row=dict(file=files[index],point_loss=float(p.detach()),quantile_loss=float(q.detach()),
                 levels=levels,combined=stats(total_p2,total_q2,total_dot))
        if index==0:
            model.config=replace(model.config,detach_uq_context=True)
            torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
            pred2,_,quant2,*_ = model.forward_uq_training(x)
            q2=.05*train.residual_quantile_loss(model,pred2,quant2,y,norm,physical_target=target)
            uq_ids={id(v) for v in train.uq_trainable_parameters(model)}
            point_parameters=tuple(v for v in model.parameters() if id(v) not in uq_ids)
            detached_gradients=torch.autograd.grad(q2,point_parameters,allow_unused=True)
            maximum=max(float((pred2-prediction_reference).abs().max()),float((quant2-quantile_reference).abs().max()))
            row['isolation']=dict(output_max_abs_difference=maximum,
                                  point_gradient_zero=all(v is None or not torch.count_nonzero(v) for v in detached_gradients))
            row['isolation']['passed']=maximum<=1e-6 and row['isolation']['point_gradient_zero']
            model.config=replace(model.config,detach_uq_context=False)
        return row

    results={}
    for label,iteration in [('sim_start',1),('real_endpoint',4000)]:
        if label=='sim_start':
            assert held.sha256(args.sim_metadata)=='282307574088299de967a7d64597a417150dd81cb18ea7b8d2897ee29b00e96f'
            binding=json.loads(args.sim_metadata.read_text())
            model,_=dense.ft12.validate_start_checkpoint(args.sim,'628860c7d82239d942895f08a6fd72ede81feb6a7cad4de4e32e110ed2742b9b',norm,torch.device('cuda:0'),expected_status=binding['status'])
        else:
            config=dict(payload['model_config']);assert config.pop('architecture')=='ensemble_unet'
            model=EnsembleUNet2d(EnsembleUNet2dConfig(**config)).float()
            model.load_state_dict(payload['model_state_dict'],strict=True);model.cuda()
        assert model.config.detach_uq_context is False and model.config.point_feature_refiner_channels==0
        rows=[case(model,index,iteration) for index in range(16)]
        median=statistics.median(r['combined']['projection_ratio'] for r in rows)
        negative=sum(r['combined']['cosine']<0 for r in rows)
        results[label]=dict(rows=rows,median_projection_ratio=median,negative_windows=negative,
                            mechanism_gate=median<=-.01 and negative>=8)
        del model;gc.collect();torch.cuda.empty_cache()
    isolation=all(v['rows'][0]['isolation']['passed'] for v in results.values())
    passed=isolation and any(v['mechanism_gate'] for v in results.values())
    result=dict(decision='PASS_QUANTILE_CONFLICT_TO_REAL_TRAINING_RESEARCH' if passed else 'STOP_QUANTILE_CONFLICT_DIAGNOSTIC',
                results=results,isolation_passed=isolation,selected_rows=selected,hashes=hashes,
                source_sha256=held.sha256(__file__),wall_seconds=time.monotonic()-started,
                scope='Training-side feature-gradient diagnostic with full-member averaging; no optimizer update, no held15 data, no quality/official gain.')
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['decision','isolation_passed','wall_seconds']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('control-receipt','sim','sim-metadata','manifest','real-dir','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
