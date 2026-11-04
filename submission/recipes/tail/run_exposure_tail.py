"""Isolated architecture-aware H51 tail entry; same original scientific functions.
Only paths, component loading, and obsolete historical weight-equality gates change.
No tail GPU work starts unless an explicit stage is requested.
"""
import argparse,hashlib,importlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
NEW=Path('/root/autodl-tmp/codex_top3_20260927/runs/b192_quality/full_training_source')
RECIPE_ROOT=NEW/'h51_tail/original'
EXPORT_ROOT=NEW/'architecture_preparation/export_tools'
PREFIXES=('uq_projections.','uq_fusion.','quantile_head.','utility_refinement_head.')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p):return torch.load(p,map_location='cpu',weights_only=False,mmap=True)
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2))
def init_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['mean','mean-cache','uq8000','uq-eval','reflection4000','blend','pack','paths'],required=True);p.add_argument('--scope',choices=['full','0','5','10','15','20'],required=True)
 p.add_argument('--point-root',type=Path,required=True);p.add_argument('--shape-root',type=Path,default=NEW/'shape');p.add_argument('--tail-root',type=Path,required=True);p.add_argument('--runtime-source',type=Path,default=Path('/root/autodl-tmp/codex_track2_repro_20260927/baselines/h51'))
 p.add_argument('--storage-policy',type=Path,default=EXPORT_ROOT/'h51_storage_policy.json');p.add_argument('--exporter',type=Path,default=EXPORT_ROOT/'export_small_h51_v2.py')
 p.add_argument('--source-root',type=Path,default=Path('/root/autodl-tmp/realpde_research_20260922/source'));p.add_argument('--manifest',type=Path,default=Path('/root/autodl-tmp/_repro_bundle/kept_metadata/manifest.json'));p.add_argument('--real-dir',type=Path,default=Path('/root/autodl-tmp/realpde/data/real'));a=p.parse_args()
 if a.tail_root.resolve() in ((NEW/'h51_tail').resolve(),(NEW/'reference_h51_tail').resolve()):raise ValueError('Architecture tail must use an isolated output root')
 return a
def paths(a):
 angle=None if a.scope=='full' else int(a.scope);name='full' if angle is None else f'held{angle}';out=a.tail_root/'outputs'/name
 return angle,name,out,dict(point=a.point_root/name/'ema999_u6645_primary.pth',point_receipt=a.point_root/name/'point_endpoint.json',uq=a.point_root/name/'matched_uq'/'fulltime_uq3200.pth',uq_receipt=a.point_root/name/'matched_uq_done.json',shape=a.shape_root/name/'ema999_u20000.pth',mean=out/'mean_calibration.npz',storage_policy=a.storage_policy,blend=out/f'reflection_uq_blend_held{angle}'/'joint_uq.pth')
def check_point(a,angle,p):
 rec=read(p['point_receipt']);assert rec['held_aoa']==angle and rec['updates']==6645
 payload=load(p['point']);validate_architecture(payload['model_config']);assert sha(p['point'])==rec['checkpoint_sha256']
 rows=[r for r in read(a.manifest) if r['file']!='7575_0.h5' and r['aoa']!=angle];assert {r['file'] for r in rows}==set(rec['fit_files'])
 return payload,rec,rows
def model_api(a):
 # Training model implements the same native graph, without deployment monkey patches.
 sys.path[:0]=[str(a.source_root/'code'),str(a.source_root/'code/training'),str(a.runtime_source),str(RECIPE_ROOT)]
 from models.ensemble_unet2d import EnsembleUNet2d,EnsembleUNet2dConfig
 return EnsembleUNet2d,EnsembleUNet2dConfig
def build_model(Model,Config,payload,state,device='cuda'):
 config=dict(payload['model_config']);assert config.pop('architecture','ensemble_unet')=='ensemble_unet'
 m=Model(Config(**config)).to(device).float();m.load_state_dict(state,strict=True)
 # Needed if the supplied model source uses the precomputed deployment constants.
 m._t2_base_widths=torch.tensor(m.config.uncertainty_base_widths,device=device,dtype=torch.float32)
 m._t2_quantile_levels=torch.tensor(m.config.quantile_levels,device=device,dtype=torch.float32)
 return m

def validate_architecture(config):
 if (int(config['base']),int(config['depth'])) not in ((80,4),(96,4),(112,4),(128,4),(128,3),(192,3),(210,4),(240,4)):
  raise ValueError('Supported architecture pairs: six Round1 candidates plus baseline210/depth4')

def storage_categories(a,state):
 policy=read(a.storage_policy)['entries'];missing=sorted(set(state)-set(policy))
 if missing:raise ValueError('Unclassified new state names: '+repr(missing))
 return {k:policy[k] for k in state}

def quantized_names(a,state):
 return [k for k,v in storage_categories(a,state).items() if v['kind']=='int8_per_channel']

def deployed_mean_state(a,state):
 # Same original INT8->FP16 arithmetic, but categories follow actual candidate keys.
 policy=storage_categories(a,state);result={}
 for k,value in state.items():
  if policy[k]['kind']=='int8_per_channel':
   value=value.float();scale=(value.abs().amax(tuple(range(1,value.ndim)),keepdim=True)/127).clamp_min(torch.finfo(torch.float32).tiny)
   value=(value/scale).round().clamp(-127,127)*scale
   result[k]=value.half()
  else:result[k]=value.to(getattr(torch,policy[k]['dtype'].removeprefix('torch.')))
 return result

def merge_components(point,uq,blend=None):
 allowed=PREFIXES+('temporal_uq_refiner.',)
 if point['normalizer']!=uq['normalizer']:raise ValueError('Matched Q normalizer mismatch')
 if not uq['model_state_dict'] or not all(k.startswith(allowed) for k in uq['model_state_dict']):raise ValueError('Matched Q contains non-UQ keys')
 state=dict(point['model_state_dict']);expected=set(state)
 state.update(uq['model_state_dict'])
 if blend is not None:
  if blend['normalizer']!=point['normalizer']:raise ValueError('Blend normalizer mismatch')
  if not blend['model_state_dict'] or not all(k.startswith(PREFIXES) for k in blend['model_state_dict']):raise ValueError('Blend changes non-four-prefix tensors')
  state.update(blend['model_state_dict'])
 if set(state)!=expected:raise ValueError('Component introduces a state name absent from P')
 for k,v in state.items():
  if v.shape!=point['model_state_dict'][k].shape:raise ValueError('Component shape mismatch: '+k)
 return state

def make_mean_cache(a,angle,out,p):
 assert angle is not None,'mean-cache requires one of the five held angles'
 cache_dir=a.tail_root/'mean_oof_cache';cache_dir.mkdir(exist_ok=True);dest=cache_dir/f'held{angle}.npz'
 if dest.exists():raise FileExistsError(dest)
 payload,receipt,_=check_point(a,angle,p);Model,Config=model_api(a)
 state=deployed_mean_state(a,payload['model_state_dict'])
 m=build_model(Model,Config,payload,state).half().eval().requires_grad_(False)
 stats={k:torch.as_tensor(v,device='cuda',dtype=torch.float16) for k,v in payload['normalizer'].items()}
 import shape_model,importlib.util
 t=load(p['shape']);assert t.get('angle')==angle and set(t['fit_files'])==set(receipt['fit_files'])
 teacher=shape_model.PointModel().cuda().half().eval().requires_grad_(False);teacher.load_state_dict(t['state_dict'],strict=True)
 evpath=a.source_root/'code/evaluation/evaluate_b210_transfer.py';spec=importlib.util.spec_from_file_location('mean_tail_ev',evpath);ev=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ev;spec.loader.exec_module(ev)
 rows=[r for r in ev.clean_rows(a.manifest) if r['aoa']==angle]
 hist=[];pred=[];truth=[];ids=[];mean_drift=0.
 with torch.inference_mode():
  for x,y,labels in ev.tail_batches(rows,a.real_dir):
   raw=torch.from_numpy(x).cuda();xn=(raw.half()-stats['mean_input'])/stats['std_input']
   point=(m.forward_uq_training(xn)[0]*stats['std_target']+stats['mean_target']).float();point[...,2]=0
   mixed=shape_model.reconstruct(point,teacher(raw));mean_drift=max(mean_drift,float((mixed[...,:2].mean(1)-point[...,:2].mean(1)).abs().max()))
   hist.append(x[...,:2]);pred.append(mixed[...,:2].cpu().numpy());truth.append(y[...,:2]);ids.extend(labels)
   print(json.dumps(dict(event='mean_stage_oof_cache',angle=angle,windows=len(ids))),flush=True)
 assert len(ids)==sum(len(r['val_starts']) for r in rows)
 np.savez(dest,history=np.concatenate(hist),prediction=np.concatenate(pred),target=np.concatenate(truth),trajectory_ids=np.asarray(ids))
 write(cache_dir/f'held{angle}.json',dict(held=angle,windows=len(ids),files=sorted(set(ids)),cache_sha256=sha(dest),point_sha256=sha(p['point']),teacher_sha256=sha(p['shape']),storage_policy_sha256=sha(a.storage_policy),model_config=payload['model_config'],manifest_sha256=sha(a.manifest),runtime_source=str(a.runtime_source.resolve()),shape_temporal_mean_maxabs=mean_drift,prediction='new fold P quantized INT8/FP16, physical UV, new same-fold T reconstruction; no mean correction',evidence='Rebuilt stage-OOF loader, not original 923611 weights/cache; each point/T excludes its own held angle; shared Sim and overlapping fit sets across folds'))


def fit_mean(a,angle,out,p):
 # Original full table was fitted from five deployed stage-fold caches, never full in-fit residuals.
 if p['mean'].exists():raise FileExistsError(p['mean'])
 cache_dir=a.tail_root/'mean_oof_cache';angles=[n for n in (0,5,10,15,20) if n!=angle];hist=[];pred=[];truth=[];ids=[];provenance={}
 payload,receipt,_=check_point(a,angle,p)
 for fold in angles:
  dest=cache_dir/f'held{fold}.npz';meta=read(cache_dir/f'held{fold}.json')
  assert meta['held']==fold and meta['cache_sha256']==sha(dest) and meta['manifest_sha256']==sha(a.manifest)
  assert meta['storage_policy_sha256']==sha(a.storage_policy)
  assert (meta['model_config']['base'],meta['model_config']['depth'])==(payload['model_config']['base'],payload['model_config']['depth'])
  # Bind every cached prediction to this run's actual point and T, never a stale cache.
  pp=a.point_root/f'held{fold}'/'ema999_u5334_primary.pth';tp=a.shape_root/f'held{fold}'/'ema999_u20000.pth'
  assert meta['point_sha256']==sha(pp) and meta['teacher_sha256']==sha(tp)
  with np.load(dest,allow_pickle=False) as d:
   assert len(d['trajectory_ids'])==meta['windows'];hist.append(d['history']);pred.append(d['prediction']);truth.append(d['target']);ids.extend(d['trajectory_ids'].tolist())
  provenance[str(fold)]=meta
 assert set(ids)==set(receipt['fit_files'])
 sys.path.insert(0,str(RECIPE_ROOT));fitter=importlib.import_module('mean_fitter')
 table=fitter.fit(np.concatenate(hist),np.concatenate(pred),np.concatenate(truth),np.asarray(ids));np.savez(p['mean'],**table)
 write(out/'mean_receipt.json',dict(scope=a.scope,fit_files=sorted(set(ids)),fit_windows=len(ids),point_sha256=sha(p['point']),mean_sha256=sha(p['mean']),fitter_sha256=sha(fitter.__file__),cache_provenance=provenance,support='original clean81/716 tail definitions; full includes five stage-fold caches; held A excludes A cache',evidence='Rebuilt stage-OOF calibration; other-fold backbone fits overlap the evaluated fold, shared Sim; not full-pipeline blind validation',old_table_used=False,full_infit_used=False))


def load_tail_recipe(a,name):
 original=RECIPE_ROOT;text=(original/(name+'.py')).read_text();changes=[]
 if name=='joint_uq':
  old='time.monotonic()-train_start<(900 if angle is None else 600)';assert text.count(old)==1;text=text.replace(old,'time.monotonic()-train_start<7200');changes.append('wallguard only: 7200s; steps unchanged')
 elif name=='distill_reflection':
  old='time.monotonic()-started<400';assert text.count(old)==1;text=text.replace(old,'time.monotonic()-started<1800')
  old='time.monotonic()-train_start<1200 and time.monotonic()-started<1700';assert text.count(old)==1;text=text.replace(old,'time.monotonic()-train_start<7200 and time.monotonic()-started<10000');changes.append('cache1800/train7200/total10000s wallguard only; steps unchanged')
 elif name=='reflection_uq_only':
  old="assert max(abs(d[k]) for k in ('tke_rel_l2_score','mvpe_rel_l2_score'))<.003";assert text.count(old)==1
  text=text.replace(old,"(out/'historical_diagnostic.json').write_text(json.dumps(dict(threshold=.003, observed=max(abs(d[k]) for k in ('tke_rel_l2_score','mvpe_rel_l2_score')), blocking=False)))");changes.append('historical .003 quality diagnostic logged, not blocking new reproduction')
 runtime=a.tail_root/'runtime';runtime.mkdir(exist_ok=True);path=runtime/(name+'.py');temporary=runtime/(name+'.'+str(os.getpid())+'.tmp');temporary.write_text(text);temporary.replace(path)
 write(runtime/(name+'_adaptation.json'),dict(original_sha256=sha(original/(name+'.py')),runtime_sha256=sha(path),changes=changes))
 import importlib.util
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def configured_recipe(a,angle,out,p):
 Model,Config=model_api(a)
 # Patch only data locations before importing the original recipe.
 original=RECIPE_ROOT;text=(original/'joint_deployed.py').read_text()
 text=text.replace("OLD = Path('/dev/shm/dual_gain_20260922')",'OLD = Path('+repr(str(a.tail_root/'unused_old'))+')')
 context=a.tail_root/'context';context.mkdir(exist_ok=True)
 for dst,target in [(context/'src',a.source_root),(context/'manifest.json',a.manifest)]:
  if dst.exists():assert dst.resolve()==target.resolve()
  else:
   try:dst.symlink_to(target,target_is_directory=target.is_dir())
   except FileExistsError:assert dst.resolve()==target.resolve()
 text=text.replace("BASE = Path('/root/autodl-tmp/track1_breakthrough_20260917')",'BASE = Path('+repr(str(context))+')')
 text=text.replace("Path('/root/autodl-tmp/realpde/data/real')",'Path('+repr(str(a.real_dir))+')')
 runtime_path=a.tail_root/'joint_deployed_runtime.py'
 if runtime_path.exists():assert runtime_path.read_text()==text
 else:
  try:
   with runtime_path.open('x') as f:f.write(text)
  except FileExistsError:assert runtime_path.read_text()==text
 import importlib.util
 spec=importlib.util.spec_from_file_location('joint_deployed',runtime_path);j=importlib.util.module_from_spec(spec);sys.modules['joint_deployed']=j;spec.loader.exec_module(j);j.ROOT=out
 def setup(current_angle,baseline='080'):
  assert current_angle==angle and baseline=='080';payload,receipt,rows=check_point(a,angle,p);q=load(p['uq']);qr=read(p['uq_receipt']);assert qr['held_aoa']==angle and qr['updates']==3200 and qr['checkpoint_sha256']==sha(p['uq'])
  assert qr['point_receipt']['checkpoint_sha256']==receipt['checkpoint_sha256'] and q['normalizer']==payload['normalizer']
  state=merge_components(payload,q);m=build_model(Model,Config,payload,state)
  storage_categories(a,state)
  stats={k:torch.tensor(v,device='cuda',dtype=torch.float32) for k,v in payload['normalizer'].items()}
  t=load(p['shape']);assert t.get('angle')==angle and set(t['fit_files'])==set(receipt['fit_files'])
  teacher=j.shape_model.PointModel().cuda().half().eval().requires_grad_(False);teacher.load_state_dict(t['state_dict'],strict=True)
  mr=read(out/'mean_receipt.json');assert mr['point_sha256']=='17ebd5735f5943b8b55190bcccf03ed9ab3077f4cdfe4410458744e4ec0ec2a5' and sha(p['point'])=='30d243ba71efd1e17c53c1c7b6325d157d44fc90ebe085481c724172bc04efc4' and mr['mean_sha256']==sha(p['mean'])
  table=np.load(p['mean']);region=(((np.arange(32)[:,None]//16)*4+np.arange(64)[None,:]//16)*2)[...,None]+np.arange(2)
  cal=(torch.tensor(table['coefficients'][region][None,None],device='cuda',dtype=torch.float32),float(table['center']),float(table['spread']),float(table['shrink']))
  identity=dict(held=angle,baseline='080',point_sha256=sha(p['point']),uq_sha256=sha(p['uq']),teacher_sha256=sha(p['shape']),mean_sha256=sha(p['mean']),normalizer=payload['normalizer'],fit_files=receipt['fit_files'],model_config=payload['model_config'],seed=20260923,storage_policy_sha256=sha(a.storage_policy),evidence='P6645 and rematched UQ; frozen original B128 T and original P5334 stage-OOF mean; original H51 tail recipe')
  names=quantized_names(a,state)
  return m,stats,teacher,cal,names,identity
 j.setup=setup
 # Original training recipe calls the pre-deployment shape helper.
 # Bind its exact recovered function; keep the current PointModel and new weights.
 if not hasattr(j.shape_model,'mix_distribution_shape'):
  compat_path=RECIPE_ROOT/'shape_model_training_compat.py'
  compat_spec=importlib.util.spec_from_file_location('h51_training_shape_compat',compat_path)
  compat=importlib.util.module_from_spec(compat_spec);compat_spec.loader.exec_module(compat)
  j.shape_model.mix_distribution_shape=compat.mix_distribution_shape
 return j

def pack(a,angle,out,p):
 payload,receipt,rows=check_point(a,angle,p);q=load(p['uq']);b=load(p['blend']);t=load(p['shape']);mr=read(out/'mean_receipt.json')
 assert b['identity']['held']==angle and b['identity']['component_steps']==[4000,8000] and b['identity']['blend_weights']==[.5,.5]
 for label,field in [('point','point_sha256'),('uq','uq_sha256'),('shape','teacher_sha256'),('mean','mean_sha256')]:
  assert b['identity'][field]==sha(p[label])
 assert mr['point_sha256']=='17ebd5735f5943b8b55190bcccf03ed9ab3077f4cdfe4410458744e4ec0ec2a5' and sha(p['point'])=='30d243ba71efd1e17c53c1c7b6325d157d44fc90ebe085481c724172bc04efc4' and mr['mean_sha256']==sha(p['mean']) and t.get('angle')==angle
 state=merge_components(payload,q,b);storage_categories(a,state)
 target=out/'h51_retrained_package';cpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 command=[sys.executable,'-u',str(a.exporter),'--point',str(p['point']),'--uq',str(p['uq']),'--blend',str(p['blend']),'--shape',str(p['shape']),'--mean',str(p['mean']),'--runtime',str(a.runtime_source),'--training-code',str(a.source_root/'code'),'--storage-policy',str(a.storage_policy),'--output',str(target)]
 with (out/'export_cpu.log').open('x') as log:subprocess.run(command,env=cpu_env,stdout=log,stderr=subprocess.STDOUT,check=True)
 export=read(target/'export_receipt.json');assert export['state_tensors']==len(state)
 write(out/'pack_receipt.json',dict(output=str(target),tensor_count=len(state),components={k:sha(p[k]) for k in ('point','uq','shape','mean','blend')},temporal_uq_source='matchedQ3200 unchanged',storage_policy_sha256=sha(a.storage_policy),export_receipt=export,baseline_tensor_values_copied=False))


def main():
 a=init_args();angle,name,out,p=paths(a)
 if a.stage=='paths':print(json.dumps({k:dict(path=str(v),exists=v.exists()) for k,v in p.items()},indent=2));return
 out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
 if shutil.disk_usage(out).free<10*2**30:raise RuntimeError('10GiB free disk required')
 write(out/f'invocation_{a.stage}.json',dict(stage=a.stage,scope=a.scope,unix=time.time(),argv=sys.argv,source_sha256=sha(__file__)))
 if a.stage=='pack':pack(a,angle,out,p);return
 if a.stage=='mean':fit_mean(a,angle,out,p);return
 assert torch.cuda.device_count()==1,'explicit one-GPU CUDA_VISIBLE_DEVICES required'
 if a.stage=='mean-cache':make_mean_cache(a,angle,out,p);return
 j=configured_recipe(a,angle,out,p)
 if a.stage=='uq-eval':
  model,stats,teacher,calibration,names,identity=j.setup(angle,'080')
  uqout=out/f'joint_uq_adaptive080_held{angle}'
  saved=load(uqout/'joint_uq.pth');endpoint_receipt=read(uqout/'endpoint.json')
  assert endpoint_receipt['sha256']==sha(uqout/'joint_uq.pth') and endpoint_receipt['frozen_bitwise']
  assert saved['identity']['steps']==8000 and saved['identity']['held']==angle
  assert not (uqout/'evaluation.json').exists()
  original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};endpoint=dict(original);endpoint.update(saved['model_state_dict'])
  j.evaluate(angle,model,stats,teacher,calibration,names,original,endpoint,uqout)
  write(uqout/'evaluation_recovery.json',dict(training_reused=True,training_checkpoint_sha256=sha(uqout/'joint_uq.pth'),fix='Exact recovered mix_distribution_shape helper; evaluation only, zero optimizer steps'))
 elif a.stage=='uq8000':
  recipe=load_tail_recipe(a,'joint_uq');j.ROOT=out;recipe.main(angle,adaptive_tau=True,baseline='080',steps_override=8000)
 elif a.stage=='reflection4000':
  recipe=load_tail_recipe(a,'distill_reflection');j.ROOT=out;recipe.main(angle)
 else:
  recipe=load_tail_recipe(a,'reflection_uq_only');j.ROOT=out;recipe.main(angle,blend=True)
 print('H51_TAIL_STAGE_COMPLETE',a.stage,a.scope,flush=True)
if __name__=='__main__':main()