"""Private architecture-only adapter around the unchanged original causal Sim trainers."""
import argparse
import dataclasses
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

SOURCE = Path('/root/autodl-tmp/realpde_research_20260922/source')
ORIGINAL_METHODS = (
 'b210_causal_denoising_future_sim_real_prefix_pretrain_v1',
 'b210_causal_denoising_pretrain_then_standard_stride20_v1',
 'b210_causal_denoising_pretrain_then_standard_stride20x2_v1')
def validate_architecture(base, depth):
 if base <= 0 or depth < 2: raise ValueError('Positive base and depth >= 2 are required by the original multiscale pretext readout')

def methods(base, depth):
 if (base, depth) == (210, 4): return ORIGINAL_METHODS
 prefix=f'architecture_variant_b{base}_d{depth}_'
 return tuple(prefix+x.removeprefix('b210_') for x in ORIGINAL_METHODS)

def import_original(source=SOURCE):
 for p in (source/'code', source/'code/training', source/'runs/private_audit'):
  sys.path.insert(0,str(p))
 return importlib.import_module('train_b210_np_functional_sim')

def install_architecture(b210, base, depth):
 import torch
 validate_architecture(base,depth)
 if os.environ.get('B210_ENERGY_CONTEXT_HEAD') == '1': raise ValueError('Unexpected architecture environment override')
 original_config=b210.b210_config
 def config(*args,**kwargs):
  value=original_config(*args,**kwargs)
  # This adapter changes only base/depth. The entry point exposes no optional architecture arms.
  return dataclasses.replace(value,base=base,depth=depth)
 # Meta construction counts real modules without allocating weights or advancing CPU RNG.
 with torch.random.fork_rng(devices=[]),torch.device('meta'):
  meta=b210.EnsembleUNet2d(config('candidate'))
  count=sum(p.numel() for p in meta.parameters())
  controller=sum(p.numel() for p in meta.energy_controller.parameters())
 original_controller_count=b210.EXPECTED_CONTROLLER_PARAMETERS
 if (base,depth)==(210,4):
  assert config('candidate')==original_config('candidate')
  assert count==b210.EXPECTED_PARAMETERS
  assert controller==original_controller_count
 b210.b210_config=config
 b210.EXPECTED_PARAMETERS=count
 b210.EXPECTED_CONTROLLER_PARAMETERS=controller
 if (base,depth)!=(210,4):
  for name in ['PASS_STATUS','HEALTH_PASS_STATUS','HEALTH_STOP_STATUS','RUNTIME_STOP_STATUS','WALL_STOP_STATUS']:
   setattr(b210,name,getattr(b210,name).replace('B210',f'ARCH_B{base}_D{depth}'))
 from architecture_health_policy import install as install_health_policy
 policy_active=install_health_policy(b210,base,depth)
 return dict(base=base,depth=depth,total_parameters=count,controller_parameters=controller,
             architecture_health_policy_active=policy_active,
             only_config_changes=['base','depth'],original_config=dataclasses.asdict(original_config('candidate')),
             actual_config=dataclasses.asdict(config('candidate')))

def configure_stage(stage, base, depth):
 names=methods(base,depth)
 if stage=='pretext':
  module=importlib.import_module('train_b210_causal_denoising_pretrain');module.METHOD=names[0]
 elif stage=='standard1':
  module=importlib.import_module('train_b210_causal_pretrained_standard_sim');module.PRETEXT_METHOD=names[0];module.METHOD=names[1]
 elif stage=='standard2':
  module=importlib.import_module('train_b210_causal_standard_continuation_sim');module.STAGE1_METHOD=names[1];module.METHOD=names[2]
 else: raise ValueError(stage)
 return module

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--base',required=True,type=int);p.add_argument('--depth',required=True,type=int)
 p.add_argument('--stage',choices=['pretext','standard1','standard2'],required=True)
 p.add_argument('--source',type=Path,default=SOURCE)
 a,rest=p.parse_known_args()
 if rest and rest[0]=='--':rest=rest[1:]
 # Do not silently enable an unrelated optional scientific or architecture variant.
 allowed={'--output-dir','--sim-dir','--real-dir','--real-stats','--clean-manifest','--device','--num-workers',
          '--pretrain-checkpoint','--pretrain-checkpoint-sha256','--teacher-cache','--teacher-cache-sha256',
          '--real-closure-bank','--real-closure-bank-sha256','--research-equivalent-cache-sha256','--acknowledge-research-cache'}
 for token in rest:
  if token.startswith('--') and token not in allowed:raise ValueError('Unprepared override '+token)
 b210=import_original(a.source);info=install_architecture(b210,a.base,a.depth)
 module=configure_stage(a.stage,a.base,a.depth)
 out=Path(rest[rest.index('--output-dir')+1])
 receipt=out.parent/(out.name+'.architecture.json');receipt.parent.mkdir(parents=True,exist_ok=True)
 info.update(stage=a.stage,scientific_recipe='original causal4k plus restarted standard10k twice',
             architecture_variant=(a.base,a.depth)!=(210,4),initialization='random in pretext; own previous stage thereafter',
             historical_C_identity_claim=False,method=methods(a.base,a.depth)[['pretext','standard1','standard2'].index(a.stage)])
 receipt.write_text(json.dumps(info,indent=2))
 sys.argv=[str(module.__file__)]+rest
 module.main()
 # Preserve original summaries, but make variant labels in copied narrative explicit.
 summary=out/'summary.json'
 if summary.exists() and info['architecture_variant']:
  s=json.loads(summary.read_text());s['architecture_variant']=info
  if isinstance(s.get('status'),str):s['status']=s['status'].replace('B210',f'ARCH_B{a.base}_D{a.depth}')
  if isinstance(s.get('evidence_class'),str):s['evidence_class']=s['evidence_class'].replace('B210',f'B{a.base}/D{a.depth} architecture variant')
  summary.write_text(json.dumps(s,indent=2))
if __name__=='__main__':main()
