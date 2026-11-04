"""Train the unchanged one-pass network against its frozen two-view teacher."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,'/dev/shm/dual_gain_20260923')
import joint_deployed as j
j.ROOT=Path(__file__).resolve().parent


def energy(p):
    uv=p[...,:2]
    return .5*(uv-uv.mean(1,keepdim=True)).square().mean(1).sum(-1)


def reflect_quantiles(q):
    q=q.flip(2)
    return torch.stack((q[...,0,:],-q[...,1,:].flip(-1)),-2)


def reflect_bounds(lo,hi):
    lo,hi=lo.flip(2),hi.flip(2)
    return torch.stack((lo[...,0],-hi[...,1]),-1),torch.stack((hi[...,0],-lo[...,1]),-1)


@torch.no_grad()
def targets(model,stats,teacher,cal,names,trajectories,out):
    started=time.monotonic()
    fixed=copy.deepcopy(model).half().eval().requires_grad_(False)
    fixed.load_state_dict(j.quantize(model.state_dict(),names),strict=True)
    windows=[(i,s) for i,t in enumerate(trajectories) for s in range(0,len(t)-39,20)]
    cache={k:[] for k in ('point','lower','upper','q','k')}
    sums={a:dict(errors=np.zeros(3),sps=0.,valid=0,n=0) for a in ('baseline','teacher')}
    for offset in range(0,len(windows),16):
        assert time.monotonic()-started<400
        chosen=windows[offset:offset+16]
        raw=torch.stack([trajectories[i][s:s+20] for i,s in chosen])
        truth=torch.stack([trajectories[i][s+20:s+40] for i,s in chosen]);sign=raw.new_tensor([1.,-1.,1.])
        p,lo,hi,q,native=j.bundle(raw,fixed,stats,teacher,cal)
        rp,rl,rh,rq,rnative=j.bundle(raw.flip(2)*sign,fixed,stats,teacher,cal)
        q=q+native[...,None];rq=rq+rnative[...,None]
        rp=rp.flip(2)*sign;rl,rh=reflect_bounds(rl[...,:2],rh[...,:2]);rq=reflect_quantiles(rq)
        pu,ru=p[...,:2],rp[...,:2]
        pm,rm=pu.mean(1,keepdim=True),ru.mean(1,keepdim=True)
        pd,rd=pu-pm,ru-rm;dynamic=.5*(pd+rd)
        k=.5*(energy(p)+energy(rp))
        power=dynamic.square().sum(-1).mean(1)
        ep=.5*(pm+rm)+dynamic*(2*k/power.clamp_min(1e-12)).sqrt()[:,None,:,:,None]
        el,eh=.5*(lo[...,:2]+rl),.5*(hi[...,:2]+rh)
        target=dict(point=ep,lower=el,upper=eh,q=.5*(q+rq),k=k)
        for key,value in target.items():
            assert torch.isfinite(value).all()
            cache[key].append(value.clone())
        assert (el<=eh).all()
        for arm,point,lower,upper in [('baseline',pu,lo[...,:2],hi[...,:2]),('teacher',ep,el,eh)]:
            y=truth[...,:2];valid=y!=0;errors=j.errors(point,y)
            factor=(.5/(.5+errors)*errors.new_tensor([.5,.3,.2])).sum(-1)
            reward=((y>=lower)&(y<=upper))*torch.exp(-(upper-lower)/.0563870259)*valid
            sums[arm]['errors']+=errors.sum(0).cpu().numpy()
            sums[arm]['sps']+=float((reward*factor[:,None,None,None,None]).sum())
            sums[arm]['valid']+=int(valid.sum());sums[arm]['n']+=len(raw)
    scores={a:dict(zip(('rel_l2_score','tke_score','mvpe_score'),(100/(1+.5*v['errors']/v['n'])).tolist()),sps_score=100*v['sps']/v['valid']) for a,v in sums.items()}
    result=dict(windows=len(windows),scores=scores,delta={k:scores['teacher'][k]-v for k,v in scores['baseline'].items()},seconds=time.monotonic()-started,
                evidence='Fit-support diagnostic only; the frozen teacher already has separate held-angle evidence')
    (out/'teacher.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(event='teacher',**result)),flush=True)
    del fixed;torch.cuda.empty_cache()
    return windows,{k:torch.cat(v) for k,v in cache.items()}


def masked_mse(a,b,valid):
    while valid.ndim<a.ndim:valid=valid.unsqueeze(-1)
    return ((a-b).square()*valid).sum()/valid.expand_as(a).sum().clamp_min(1)


def main(angle):
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(20260924)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=True
    out=j.ROOT/f'reflection_distill_held{angle}';out.mkdir(exist_ok=False)
    model,stats,teacher,cal,names,identity=j.setup(angle,'080')
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    ev=j.load(j.BASE/'src/code/evaluation/evaluate_b210_transfer.py','reflection_distill_data')
    rows=[r for r in ev.clean_rows(j.BASE/'manifest.json') if r['aoa']!=angle]
    assert sorted(r['file'] for r in rows)==sorted(identity['fit_files'])
    trajectories,_=j.data(rows)
    windows,cache=targets(model,stats,teacher,cal,names,trajectories,out)
    prefixes=('uq_projections.','uq_fusion.','quantile_head.','utility_refinement_head.')
    model.train().requires_grad_(True)
    groups=[dict(params=[p for n,p in model.named_parameters() if n.startswith(prefixes)==uq],lr=lr,peak=lr,floor=floor) for uq,lr,floor in ((False,3e-6,3e-7),(True,1e-5,2e-6))]
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-5)
    ema={k:v.detach().clone() for k,v in model.state_dict().items()}
    identity.update(method='Single-pass student of physical reflection ensemble',steps=4000,teacher_uses_current20_only=True,
                    loss_teacher_weights=dict(point=.2,bounds=.2,quantiles=.05,energy=.5),point_and_uq_training=True,inference_topology_unchanged=True)
    (out/'identity.json').write_text(json.dumps(identity,indent=2))
    rng=np.random.default_rng(20260924);train_start=time.monotonic();parameters=list(model.parameters())
    for step in range(1,4001):
        assert time.monotonic()-train_start<1200 and time.monotonic()-started<1700
        indexes=rng.integers(0,len(windows),8);chosen=[windows[n] for n in indexes]
        raw=torch.stack([trajectories[i][s:s+20] for i,s in chosen]);truth=torch.stack([trajectories[i][s+20:s+40] for i,s in chosen])
        target={k:v[indexes] for k,v in cache.items()}
        flip=torch.tensor(rng.random(8)<.25,device='cuda');sign=raw.new_tensor([1.,-1.,1.])
        raw=torch.where(flip[:,None,None,None,None],raw.flip(2)*sign,raw)
        truth=torch.where(flip[:,None,None,None,None],truth.flip(2)*sign,truth)
        target['point']=torch.where(flip[:,None,None,None,None],target['point'].flip(2)*sign[:2],target['point'])
        rl,rh=reflect_bounds(target['lower'],target['upper'])
        target['lower']=torch.where(flip[:,None,None,None,None],rl,target['lower'])
        target['upper']=torch.where(flip[:,None,None,None,None],rh,target['upper'])
        target['q']=torch.where(flip[:,None,None,None,None,None],reflect_quantiles(target['q']),target['q'])
        target['k']=torch.where(flip[:,None,None],target['k'].flip(1),target['k'])
        outputs=j.bundle(raw,model,stats,teacher,cal)
        scale=raw.new_tensor([.01075,.00550]);valid=truth[...,:2]!=0
        rms=(raw[...,:2]-raw[...,:2].mean(1,keepdim=True)).square().mean((1,2,3),keepdim=True).sqrt()
        tau=torch.minimum(.05*rms,.05*scale).clamp_min(1e-5)
        loss,detail=j.objective(outputs,truth,tau)
        point_loss=masked_mse(outputs[0][...,:2]/scale,target['point']/scale,valid)
        bounds_loss=.5*(masked_mse(outputs[1][...,:2]/scale,target['lower']/scale,valid)+masked_mse(outputs[2][...,:2]/scale,target['upper']/scale,valid))
        quantile_loss=masked_mse((outputs[3]+outputs[4][...,None])/scale[:,None],target['q']/scale[:,None],valid)
        predicted_k=energy(outputs[0]);target_k=target['k']
        energy_loss=((predicted_k-target_k).square().sum((1,2))/target_k.square().sum((1,2)).clamp_min(1e-16)).mean()
        loss=loss+.2*point_loss+.2*bounds_loss+.05*quantile_loss+.5*energy_loss
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        if step==1:
            assert grad>0 and all(p.grad is None for p in teacher.parameters())
            (out/'preflight.json').write_text(json.dumps(dict(gradient_norm=float(grad),T_teacher_frozen=True,student_dropout_preserved=True,teacher_samples=len(windows))))
        ratio=.5*(1+math.cos(math.pi*step/4000))
        for group in optimizer.param_groups:group['lr']=group['floor']+(group['peak']-group['floor'])*ratio
        optimizer.step()
        with torch.no_grad():
            for k,v in model.state_dict().items():
                if v.is_floating_point():ema[k].lerp_(v.detach(),.001)
                else:ema[k].copy_(v)
        if step==1 or step%400==0:
            print(json.dumps(dict(event='train',angle=angle,step=step,seconds=time.monotonic()-train_start,loss=float(loss),point_teacher=float(point_loss),bounds_teacher=float(bounds_loss),quantile_teacher=float(quantile_loss),energy_teacher=float(energy_loss),**detail)),flush=True)
    endpoint={k:v.cpu() for k,v in ema.items()};path=out/'model.pth'
    torch.save(dict(model_state_dict=endpoint,normalizer=identity['normalizer'],model_config=identity['model_config'],identity=identity),path)
    (out/'endpoint.json').write_text(json.dumps(dict(sha256=j.sha(path),bytes=path.stat().st_size,seconds=time.monotonic()-started)))
    del cache,trajectories,ema,optimizer,outputs,loss;torch.cuda.empty_cache()
    j.evaluate(angle,model,stats,teacher,cal,names,original,endpoint,out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--angle',type=int);group.add_argument('--full',action='store_true');args=parser.parse_args();main(args.angle)
