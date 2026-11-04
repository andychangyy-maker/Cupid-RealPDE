"""Architecture-aware entry into original P5334 / matched-Q3200 scientific loops."""
import argparse,dataclasses,hashlib,importlib,json,math,os,re,shutil,sys
from pathlib import Path
from types import SimpleNamespace
from parametric_original import SOURCE,import_original,install_architecture
NEW=Path('/root/autodl-tmp/realpde/track2_repro_20260927')
SCHEMA='architecture_variant_real5334_v1'
FAMILY='causal4k_standard20_architecture_variant'
CACHE_SHA='8f73ecbfb792101ce7b7f900947326f8946263be4ce30b5aab0fb8c1fd063301'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def validate_sim(binding,base,depth,sim_sha):
 assert binding['schema']=='architecture_variant_causal_sim_binding_v1'
 assert binding['family']==FAMILY and binding['phase']=='sim'
 assert (binding['base'],binding['depth'])==(base,depth)
 assert binding['iteration']==10000 and binding['checkpoint_sha256']==sim_sha
 assert [x['updates'] for x in binding['lineage']]==[4000,10000,10000]
 assert binding['lineage'][-1]['checkpoint_sha256']==sim_sha
 assert binding['teacher_cache_sha256']==CACHE_SHA
 assert binding['historical_C_identity_claim'] is False
 assert binding['status'] and binding['training_visibility'].startswith('shared clean81 Real prefix')

def install_real(base,depth,runtime_root):
 import torch
 b=import_original();info=install_architecture(b,base,depth)
 sys.path.insert(0,str(runtime_root))
 transfer=importlib.import_module('point_runtime')
 dense=transfer.load_runtime();ft=dense.ft12
 ft.EXPECTED_PARAMETERS=info['total_parameters'];ft.EXPECTED_CONTROLLER_PARAMETERS=info['controller_parameters']
 # expected_b210_config resolves the already parameterized original config.
 def exact_config(config,expected_config=None):
  stored=dict(config);assert stored.pop('architecture')=='ensemble_unet'
  parsed=b.EnsembleUNet2dConfig(**stored)
  assert parsed==(expected_config if expected_config is not None else b.b210_config('candidate'))
  return parsed
 ft.validate_b210_model_config=exact_config
 transfer.SCHEMA=SCHEMA;transfer.METADATA_SCHEMA='architecture_variant_causal_sim_binding_v1'
 transfer.ARMS={**transfer.ARMS,'C':(FAMILY,False)}
 def binding_validator(args):
  assert not transfer.plan(args)['missing_execution_arguments']
  assert args.arm=='C' and re.fullmatch(r'[0-9a-f]{40}',args.source_commit)
  assert math.isfinite(args.wall_budget_seconds) and 0<args.wall_budget_seconds<=3600
  transfer.require_sha(args.start_sim,args.start_sim_sha256)
  transfer.require_sha(args.start_metadata,args.start_metadata_sha256)
  value=json.loads(args.start_metadata.read_text())
  validate_sim(value,base,depth,args.start_sim_sha256.lower())
  return value
 transfer.validate_binding=binding_validator
 original_fields=transfer.intervention_fields
 def fields(args):
  return {**original_fields(args),'architecture':info,'historical_C_identity_claim':False,
          'shared_sim_stage_cv':True}
 transfer.intervention_fields=fields
 uq=importlib.import_module('train_b210_transfer_uq')
 native=importlib.import_module('train_b210_small4k_uq_release')
 native.EXPECTED_PARAMETERS=info['total_parameters'];native.EXPECTED_CONTROLLER_PARAMETERS=info['controller_parameters']
 with torch.device('meta'):
  model=b.EnsembleUNet2d(b.b210_config('candidate'))
 uq.UQ_PARAMETERS=sum(p.numel() for n,p in model.named_parameters() if any(n.startswith(prefix) for prefix in native.UQ_PREFIXES))
 info['uq_parameters']=uq.UQ_PARAMETERS
 uq.SCHEMA='architecture_variant_matched_uq3200_v1'
 uq.POINT_SCHEMA=SCHEMA+'_endpoint';uq.POINT_RECEIPT_SCHEMA=SCHEMA+'_point_receipt'
 return b,transfer,dense,uq,native,info

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--base',type=int,required=True);p.add_argument('--depth',type=int,required=True)
 p.add_argument('--sim',type=Path,required=True);p.add_argument('--binding',type=Path,required=True)
 p.add_argument('--scope',choices=['full','0','5','10','15','20'],required=True)
 p.add_argument('--stage',choices=['point','uq'],required=True);p.add_argument('--output-root',type=Path,required=True)
 a=p.parse_args();root=a.output_root.resolve();root.mkdir(parents=True,exist_ok=True)
 origin=NEW/'point_recipe'
 copied=['train.py','coupling.py','point_runtime.py','train_matched_uq.py','control_uq_receipts.json']
 for n in copied:
  target=root/n
  if target.exists():assert sha(target)==sha(origin/n),'Private scientific source differs: '+n
  else:shutil.copy2(origin/n,target)
 sim_sha=sha(a.sim);binding_sha=sha(a.binding);binding=json.loads(a.binding.read_text())
 validate_sim(binding,a.base,a.depth,sim_sha)
 b,transfer,dense,uq,native,info=install_real(a.base,a.depth,root)
 plan=dict(schema='architecture_variant_original_real_plan_v1',source_root=str(SOURCE),
  source_commit_label='5b28cf57f609fc2fe481d8f08818a5a0a7d9ba76',
  source_commit_label_note='Original recipe label; exact source identities are recorded separately.',
  original_scientific_sources={n:sha(origin/n) for n in copied},adapter_sha256=sha(__file__),architecture=info,
  sim=str(a.sim.resolve()),sim_sha256=sim_sha,sim_metadata=str(a.binding.resolve()),sim_metadata_sha256=binding_sha,
  manifest='/root/autodl-tmp/rp_repro_20260917/track1_gain_20260916_h1/manifest.json',
  real_dir='/root/autodl-tmp/realpde/data/real',point_wall_budget_seconds=3600,
  reproduction_note='Architecture variant from its own causal4k+standard20k Sim; original P5334/Q3200 loops, losses, folds, normalizers and sampling. Shared-Sim stage CV, not blind held-out Sim.')
 plan=json.loads(json.dumps(plan))  # canonical JSON, including tuple-valued config fields
 path=root/'plan.json'
 if path.exists():assert json.loads(path.read_text())==plan,'Existing plan differs'
 else:
  tmp=root/('plan.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(plan,indent=2));tmp.replace(path)
 full=a.scope=='full';angle=None if full else int(a.scope)
 if a.stage=='point':
  import train
  train.main(SimpleNamespace(plan=path,full_fit=full,angle=angle))
 else:
  import train_matched_uq
  train_matched_uq.main(root,angle,False,full)
if __name__=='__main__':main()