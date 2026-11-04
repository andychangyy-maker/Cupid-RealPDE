"""Fine-tune existing UQ through deployed point shape; keep the point backbone frozen."""
import argparse
import json
import math
import time
import numpy as np
import torch
import joint_deployed as j


def main(angle,adaptive_tau=False,baseline='320',point_eval=False,energy_joint=False,steps_override=None):
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(20260923)
    out=j.ROOT/f'joint_uq{"_energy" if energy_joint else ""}{"_adaptive" if adaptive_tau else ""}{"_point_eval" if point_eval else ""}{"080" if baseline=="080" else ""}_held{angle}';out.mkdir(exist_ok=False)
    model,stats,teacher,calibration,names,identity=j.setup(angle,baseline)
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    prefixes=('uq_projections.','uq_fusion.','quantile_head.','utility_refinement_head.')
    if energy_joint:
        from energy_priority import objective
        assert point_eval and model.energy_controller is not None
        prefixes+=('energy_controller.',)
        j.objective=objective
    steps=steps_override if steps_override is not None else (2400 if energy_joint else 1600)
    model.train().requires_grad_(False)
    train_names=[]
    for name,param in model.named_parameters():
        if name.startswith(prefixes):param.requires_grad_(True);train_names.append(name)
    assert train_names and len(train_names)<len(list(model.parameters()))
    parameters=[p for p in model.parameters() if p.requires_grad]
    fixed_blocks=[]
    if point_eval:
        for name,module in model.named_modules():
            if getattr(module,'residual_dropout',0)>0:
                assert all(not p.requires_grad for p in module.parameters())
                module.eval();fixed_blocks.append(name)
        assert fixed_blocks
    identity.update(train_names=train_names,train_parameters=sum(p.numel() for p in parameters),steps=steps,
                    tau_policy='min(global tau, 0.05 input dynamic RMS), floor 1e-5' if adaptive_tau else 'fixed global tau',
                    method='Existing energy controller and UQ jointly learn; frozen point encoder/decoder' if energy_joint else 'Only existing UQ learns through deployed point and interval; unchanged inference topology',
                    energy_joint=energy_joint,
                    point_score_weights=[.1,.7,.1] if energy_joint else [.5,.1,.1],
                    soft_sps_weight=.8 if energy_joint else .25,
                    frozen_dropout_disabled=fixed_blocks)
    (out/'identity.json').write_text(json.dumps(identity,indent=2))
    ev=j.load(j.BASE/'src/code/evaluation/evaluate_b210_transfer.py','joint_uq_data')
    rows=[r for r in ev.clean_rows(j.BASE/'manifest.json') if r['aoa']!=angle]
    assert sorted(r['file'] for r in rows)==sorted(identity['fit_files'])
    trajectories,windows=j.data(rows)
    if point_eval:
        with torch.no_grad():
            check=trajectories[0][:20].unsqueeze(0)
            aligned=j.bundle(check,model,stats,teacher,calibration)[0]
            for name,module in model.named_modules():
                if name in fixed_blocks:module.train()
            noisy=j.bundle(check,model,stats,teacher,calibration)[0]
            model.eval();deployed=j.bundle(check,model,stats,teacher,calibration)[0]
            model.train()
            for name,module in model.named_modules():
                if name in fixed_blocks:module.eval()
            comparison=dict(frozen_dropout_blocks=fixed_blocks,training_relative_shift=float((noisy-deployed).norm()/deployed.norm()),
                            aligned_relative_shift=float((aligned-deployed).norm()/deployed.norm()),aligned_maxabs=float((aligned-deployed).abs().max()))
            assert comparison['aligned_maxabs']<1e-6,comparison
            (out/'mode_alignment.json').write_text(json.dumps(comparison,indent=2))
            print(json.dumps(dict(event='mode_alignment',angle=angle,**comparison)),flush=True)
    opt=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=1e-5)
    ema={k:v.detach().clone() for k,v in model.state_dict().items() if k.startswith(prefixes)}
    rng=np.random.default_rng(20260923);train_start=time.monotonic()
    for step in range(1,steps+1):
        assert time.monotonic()-train_start<(900 if angle is None else 600)
        chosen=[windows[i] for i in rng.integers(0,len(windows),8)]
        raw=torch.stack([trajectories[i][s:s+20] for i,s in chosen])
        truth=torch.stack([trajectories[i][s+20:s+40] for i,s in chosen])
        flip=torch.tensor(rng.random(8)<.25,device='cuda');sign=raw.new_tensor([1.,-1.,1.])
        raw=torch.where(flip[:,None,None,None,None],raw.flip(2)*sign,raw)
        truth=torch.where(flip[:,None,None,None,None],truth.flip(2)*sign,truth)
        outputs=j.bundle(raw,model,stats,teacher,calibration)
        tau=None
        if adaptive_tau:
            rms=(raw[...,:2]-raw[...,:2].mean(1,keepdim=True)).square().mean((1,2,3),keepdim=True).sqrt()
            tau=torch.minimum(.05*rms,raw.new_tensor([.01075,.00550])*.05).clamp_min(1e-5)
        loss,detail=j.objective(outputs,truth,tau=tau);assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        assert all(p.grad is None for name,p in model.named_parameters() if name not in train_names)
        if step==1:
            assert grad>0
            energy_grad=sum(float(p.grad.square().sum()) for name,p in model.named_parameters() if name.startswith('energy_controller.') and p.grad is not None)**.5
            if energy_joint:assert energy_grad>0
            (out/'preflight.json').write_text(json.dumps(dict(point_backbone_frozen=True,teacher_frozen=True,gradient_norm=float(grad),
                energy_controller_training=energy_joint,energy_gradient_norm=energy_grad,
                tau_min=None if tau is None else float(tau.min()),tau_max=None if tau is None else float(tau.max()))))
        lr=2e-6+.5*(1e-5-2e-6)*(1+math.cos(math.pi*step/steps))
        for group in opt.param_groups:group['lr']=lr
        opt.step()
        with torch.no_grad():
            for k,v in model.state_dict().items():
                if k in ema:ema[k].lerp_(v.detach(),.001)
        if step==1 or step%200==0:print(json.dumps(dict(event='training',angle=angle,step=step,loss=float(loss),seconds=time.monotonic()-train_start,**detail)),flush=True)
    assert all(torch.equal(v.cpu(),original[k]) for k,v in model.state_dict().items() if not k.startswith(prefixes))
    compact={k:v.cpu() for k,v in ema.items()};endpoint=dict(original);endpoint.update(compact)
    path=out/'joint_uq.pth';torch.save(dict(model_state_dict=compact,identity=identity,normalizer=identity['normalizer']),path)
    (out/'endpoint.json').write_text(json.dumps(dict(sha256=j.sha(path),bytes=path.stat().st_size,frozen_bitwise=True,seconds=time.monotonic()-started)))
    del trajectories,ema,opt,outputs,loss;torch.cuda.empty_cache()
    j.evaluate(angle,model,stats,teacher,calibration,names,original,endpoint,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--angle',type=int,required=True);p.add_argument('--adaptive-tau',action='store_true')
    p.add_argument('--baseline',choices=['320','080'],default='320')
    p.add_argument('--point-eval',action='store_true')
    p.add_argument('--energy-joint',action='store_true')
    args=p.parse_args();main(args.angle,args.adaptive_tau,args.baseline,args.point_eval,args.energy_joint)
