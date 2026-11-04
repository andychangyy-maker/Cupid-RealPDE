"""Export complete new EnsembleUNet state using the final H51 per-name INT8/FP16 policy.
No baseline tensor values are copied; only explicit storage-category metadata is read. Native-only mode is an interface artifact, not a submission.
"""
import argparse,dataclasses,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import torch
UQ=('uq_projections.','uq_fusion.','quantile_head.','utility_refinement_head.','temporal_uq_refiner.')
BLEND=UQ[:4]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def canon(v):
 if isinstance(v,(tuple,list)):return [canon(x) for x in v]
 if isinstance(v,dict):return {k:canon(x) for k,x in v.items()}
 return v
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def compact(state,policy):
 out={};quantized=[]
 for key,value in state.items():
  value=value.detach().cpu()
  if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('Nonfinite '+key)
  if key not in policy:raise ValueError('New architecture name needs an explicit storage policy: '+key)
  category=policy[key]
  if category['kind']=='int8_per_channel':
   if not(value.is_floating_point() and value.ndim>=2):raise ValueError('INT8 policy/type mismatch: '+key)
   v=value.float();scale=(v.abs().amax(tuple(range(1,v.ndim)),keepdim=True)/127).clamp_min(torch.finfo(torch.float32).tiny)
   out[key]={'quantized':(v/scale).round().clamp(-127,127).to(torch.int8),'scale':scale};quantized.append(key)
  else:
   dtype=getattr(torch,category['dtype'].removeprefix('torch.'))
   out[key]=value.to(dtype)
 return out,quantized
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--point',type=Path,required=True);p.add_argument('--uq',type=Path);p.add_argument('--blend',type=Path)
 p.add_argument('--shape',type=Path);p.add_argument('--mean',type=Path);p.add_argument('--native-only',action='store_true')
 p.add_argument('--storage-policy',type=Path,default=Path(__file__).with_name('h51_storage_policy.json'))
 p.add_argument('--runtime',type=Path,required=True);p.add_argument('--training-code',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-forward',action='store_true');p.add_argument('--threads',type=int,default=2)
 a=p.parse_args();torch.set_num_threads(a.threads)
 if not a.native_only and not(a.shape and a.mean):p.error('Full package needs explicit --shape and --mean sources')
 if a.native_only and (a.shape or a.mean):p.error('native-only must not imply a full T/mean package')
 if a.blend and not a.uq:p.error('blend requires explicit matched UQ source')
 if a.output.exists():raise FileExistsError(a.output)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 script='import sys,json,dataclasses;sys.path.insert(0,sys.argv[1]);from models.ensemble_unet2d import EnsembleUNet2dConfig as C;print(json.dumps(dataclasses.asdict(C())))'
 defaults=json.loads(subprocess.check_output([sys.executable,'-c',script,str(a.training_code)],env=env,text=True))
 sys.path.insert(0,str(a.runtime));import base_submission as stock
 C=stock.EnsembleUNet2dConfig;Model=stock.EnsembleUNet2d
 point=load(a.point);cfg=dict(point['model_config']);arch=cfg.pop('architecture','ensemble_unet')
 if arch!='ensemble_unet':raise ValueError(arch)
 fields={f.name for f in dataclasses.fields(C)};dropped={}
 for key in sorted(set(cfg)-fields):
  if key not in defaults or canon(cfg[key])!=canon(defaults[key]):raise ValueError('Unsupported nondefault training option: '+key)
  dropped[key]=cfg.pop(key)
 model=Model(C(**cfg));state=dict(point['model_state_dict']);expected=model.state_dict()
 if set(state)!=set(expected):raise ValueError('Point must provide the complete actual architecture state')
 sources={'point':{'path':str(a.point),'sha256':sha(a.point)},'uq':None,'blend':None,'shape':None,'mean':None}
 for label,path,prefixes in [('uq',a.uq,UQ),('blend',a.blend,BLEND)]:
  if path:
   payload=load(path);patch=payload['model_state_dict']
   if not patch or not all(k.startswith(prefixes) for k in patch):raise ValueError(label+' must contain only its declared UQ tensors')
   if canon(payload['normalizer'])!=canon(point['normalizer']):raise ValueError(label+' normalizer mismatch')
   if set(patch)-set(expected):raise ValueError(label+' contains unknown state keys')
   state.update(patch);sources[label]={'path':str(path),'sha256':sha(path),'tensor_count':len(patch)}
 for key,value in state.items():
  if value.shape!=expected[key].shape:raise ValueError('Actual-config state shape mismatch: '+key)
 model.load_state_dict(state,strict=True)
 storage=json.loads(a.storage_policy.read_text());policy=storage['entries'];packed,quantized=compact(state,policy)
 tensor_bytes=sum(sum(x.numel()*x.element_size() for x in v.values()) if isinstance(v,dict) else v.numel()*v.element_size() for v in packed.values())
 category_mismatch=[k for k,v in packed.items() if (isinstance(v,dict))!=(policy[k]['kind']=='int8_per_channel') or (not isinstance(v,dict) and str(v.dtype)!=policy[k]['dtype'])]
 assert not category_mismatch
 # Copy code/runtime dependencies only. Never copy a baseline model or mean file.
 shutil.copytree(a.runtime,a.output,ignore=shutil.ignore_patterns('*.pth','*.pt','*.npz','*.ckpt','__pycache__'))
 payload={'model_config':dict(cfg,architecture='ensemble_unet'),'normalizer':point['normalizer'],'inference_dtype':'float16','quantized_state_dict':packed,'recipe':'Complete new architecture state; final H51 per-name INT8/FP16 categories','component_sources':sources}
 torch.save(payload,a.output/'ensemble_unet2d.pth')
 if not a.native_only:
  t=load(a.shape);torch.save(dict(t,inference_dtype='float16'),a.output/'shape_model.pth');shutil.copy2(a.mean,a.output/'mean_calibration.npz')
  sources['shape']={'path':str(a.shape),'sha256':sha(a.shape)};sources['mean']={'path':str(a.mean),'sha256':sha(a.mean)}
  # Store complete source metadata after adding explicitly supplied auxiliary artifacts.
  payload['component_sources']=sources;torch.save(payload,a.output/'ensemble_unet2d.pth')
 uq_n=sum(v.numel() for k,v in model.named_parameters() if k.startswith(UQ));total=sum(v.numel() for v in model.parameters())
 check_script='''import sys,json,torch
sys.path.insert(0,sys.argv[1]);import base_submission as s
torch.set_num_threads(int(sys.argv[3]));m,d,_,_=s._load_model(sys.argv[1],torch.device('cpu'));r=dict(stock_cpu_load=True,parameters=sum(p.numel() for p in m.parameters()),dtype=str(d))
if sys.argv[2]=='1':
 torch.manual_seed(1042)
 with torch.inference_mode():p,l,u=m.forward_distribution(torch.randn(1,20,32,64,3)*.05)
 r.update(forward_shapes=[list(x.shape) for x in (p,l,u)],finite=all(torch.isfinite(x).all().item() for x in (p,l,u)),bounds_ordered=bool((l+u>=0).all()))
 assert r['finite'] and r['bounds_ordered']
print(json.dumps(r))'''
 check=json.loads(subprocess.check_output([sys.executable,'-c',check_script,str(a.output.resolve()),str(int(a.cpu_forward)),str(a.threads)],env=env,text=True).strip().splitlines()[-1])
 report=dict(status='PASS',native_only=a.native_only,model_config=cfg,parameters=total,point_parameters=total-uq_n,uq_parameters=uq_n,state_tensors=len(state),quantized_tensors=len(quantized),float_or_integer_tensors=len(state)-len(quantized),quantization='Final H51 per-name categories on intersection; new scales/quantized values recomputed; unclassified names explicitly rejected',storage_policy_source=str(a.storage_policy),tensor_bytes=tensor_bytes,storage_category_mismatch=category_mismatch,quantized_names=quantized,baseline_tensor_values_copied=False,component_sources=sources,dropped_training_defaults=dropped,stock_check=check,checkpoint_bytes=(a.output/'ensemble_unet2d.pth').stat().st_size,checkpoint_sha256=sha(a.output/'ensemble_unet2d.pth'),quality_claim=False,speed_claim=False)
 (a.output/'export_receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
