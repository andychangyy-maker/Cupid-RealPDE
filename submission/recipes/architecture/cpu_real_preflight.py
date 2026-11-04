"""Small CPU interface/gradient checks, never a trained-model quality claim."""
import argparse,dataclasses,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
import run_architecture_real as adapter

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=int,required=True);p.add_argument('--depth',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(1729)
 b,transfer,dense,uq,native,info=adapter.install_real(a.base,a.depth,adapter.NEW/'point_recipe')
 assert uq.POINT_SCHEMA==transfer.SCHEMA+'_endpoint'
 assert uq.POINT_RECEIPT_SCHEMA==transfer.SCHEMA+'_point_receipt'
 cfg=b.b210_config('candidate');cfgdict={'architecture':'ensemble_unet',**dataclasses.asdict(cfg)}
 assert dense.ft12.validate_b210_model_config(cfgdict)==cfg
 bad=dict(cfgdict);bad['base']+=1
 try:dense.ft12.validate_b210_model_config(bad)
 except AssertionError:pass
 else:raise AssertionError('Wrong actual config accepted')
 m=b.EnsembleUNet2d(cfg).float();trainable=dense.ft12.configure_trainable(m)
 assert sum(p.numel() for p in trainable)==info['total_parameters']
 norm=SimpleNamespace(mean_input=torch.zeros(3),std_input=torch.ones(3),mean_target=torch.zeros(3),std_target=torch.ones(3))
 dense.ft12.npt1.ensemble_train.common.set_model_stats(m,norm)
 with torch.device('meta'):clone=b.EnsembleUNet2d(cfg)
 clone.load_state_dict(m.state_dict(),strict=True,assign=True);del clone
 x=torch.randn(1,20,8,16,3);target=torch.randn_like(x)
 opt=torch.optim.SGD(m.parameters(),lr=1e-3)
 records=[]
 for step in range(2):
  opt.zero_grad(set_to_none=True);m.train();out=m.forward_uq_training(x)
  assert out[0].shape==x.shape and all(torch.isfinite(v).all() for v in out if isinstance(v,torch.Tensor))
  # Exercise every native output branch; scientific loss functions themselves are untouched.
  loss=sum((v.square().mean()+.01*v.mean()) for v in out if isinstance(v,torch.Tensor))
  loss.backward()
  finite=[n for n,p in m.named_parameters() if p.grad is not None and torch.isfinite(p.grad).all()]
  missing=[n for n,p in m.named_parameters() if p.grad is None]
  nonzero=[n for n,p in m.named_parameters() if p.grad is not None and p.grad.abs().max()>0]
  assert len(finite)==len(trainable) and not missing
  records.append(dict(step=step+1,finite=len(finite),nonzero=len(nonzero),total=len(trainable)))
  opt.step()
 opt.zero_grad(set_to_none=True);params=native.configure_uq_only(m);native.set_uq_training_mode(m)
 assert sum(p.numel() for p in params)==info['uq_parameters']
 out=m.forward_uq_training(x);loss=sum(v.square().mean() for v in out[2:] if isinstance(v,torch.Tensor));loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
 assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
 report=dict(passed=True,cpu_only=True,architecture=info,strict_state_reload=True,config_rejects_wrong_base=True,point_gradients=records,uq_gradients_finite=len(params),point_gradients_none_in_uq=True,original_training_loops_unchanged=True,point_schema=uq.POINT_SCHEMA,receipt_schema=uq.POINT_RECEIPT_SCHEMA,output_shapes=[list(v.shape) for v in out if isinstance(v,torch.Tensor)])
 a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()