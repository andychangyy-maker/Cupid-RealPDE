"""Prepare or run an isolated architecture variant of the original 4k+10k+10k causal Sim chain."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from parametric_original import SOURCE,methods,validate_architecture
from architecture_health_policy import POLICY
NEW=Path('/root/autodl-tmp/realpde/track2_repro_20260927')
CACHE_SHA='8f73ecbfb792101ce7b7f900947326f8946263be4ce30b5aab0fb8c1fd063301'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=int,required=True);p.add_argument('--depth',type=int,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
 validate_architecture(a.base,a.depth)
 root=a.output_root.resolve();root.mkdir(parents=True,exist_ok=True)
 cache=NEW/'sim/p1_teacher_cache.pt';bank=NEW/'sim/clean81_closure.pt'
 assert sha(cache)==CACHE_SHA
 bank_sha=sha(bank)
 wrapper=Path(__file__).with_name('parametric_original.py')
 shared=[sys.executable,'-u',str(wrapper),'--base',str(a.base),'--depth',str(a.depth)]
 data=['--sim-dir','/root/autodl-tmp/realpde/data/sim','--real-stats','/root/autodl-tmp/realpde/kits/track2/realpde_t2_starting_kit_v6/example_data/mean_std_real.pt','--device','cuda:0','--num-workers','0']
 stage_specs=[('pretext',4000,'causal_pretrain_4000.pth'),('standard1',10000,'model_10000.pth'),('standard2',10000,'model_10000.pth')]
 stages=[]
 for i,(stage,updates,filename) in enumerate(stage_specs):
  cmd=shared+['--stage',stage,'--']+data+['--output-dir',str(root/stage)]
  if stage=='pretext':cmd+=['--real-dir','/root/autodl-tmp/realpde/data/real','--clean-manifest','/root/autodl-tmp/rp_repro_20260917/track1_gain_20260916_h1/manifest.json']
  else:
   parent=root/stage_specs[i-1][0]/stage_specs[i-1][2]
   cmd+=['--pretrain-checkpoint',str(parent),'--pretrain-checkpoint-sha256','SHA_OF_OWN_PREVIOUS_STAGE','--teacher-cache',str(cache),'--teacher-cache-sha256',CACHE_SHA,'--research-equivalent-cache-sha256',CACHE_SHA,'--acknowledge-research-cache','B210_EQUIVALENT_P1_CACHE_RESEARCH_ONLY','--real-closure-bank',str(bank),'--real-closure-bank-sha256',bank_sha]
  stages.append(dict(name=stage,updates=updates,checkpoint=str(root/stage/filename),command=cmd,method=methods(a.base,a.depth)[i]))
 sources=['code/training/train_b210_causal_denoising_pretrain.py','code/training/train_b210_masked_reconstruction_pretrain.py','code/training/train_b210_np_functional_sim.py','code/models/ensemble_unet2d.py','runs/private_audit/train_b210_causal_pretrained_standard_sim.py','runs/private_audit/train_b210_causal_standard_continuation_sim.py','runs/private_audit/train_b210_dense_phase_pretrained_standard_sim.py','runs/private_audit/train_b210_standard_pretrained_standard_sim.py']
 plan=dict(source_root=str(SOURCE),source_sha256={n:sha(SOURCE/n) for n in sources},schema='architecture_variant_causal_sim_plan_v1',base=a.base,depth=a.depth,initialization='random pretext then own-stage strict reload',historical_C_identity_claim=False,gpu_uuid=a.gpu_uuid,teacher_cache_sha256=CACHE_SHA,closure_bank_sha256=bank_sha,original_scientific_recipe=True,wrapper_sha256=sha(wrapper),health_policy=POLICY if (a.base,a.depth)!=(210,4) else 'original unchanged',health_policy_source_sha256=sha(wrapper.with_name('architecture_health_policy.py')),stages=stages)
 (root/'architecture_plan.json').write_text(json.dumps(plan,indent=2))
 if a.plan_only:print(json.dumps(plan,indent=2));return
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu_uuid,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 lineage=[]
 for stage in stages:
  if shutil.disk_usage(root).free<20*2**30:raise RuntimeError('Less than 20 GiB free')
  log=root/(stage['name']+'.log')
  if log.exists() or (root/stage['name']).exists():raise FileExistsError('Refusing to overwrite or silently restart '+stage['name'])
  command=list(stage['command'])
  if 'SHA_OF_OWN_PREVIOUS_STAGE' in command:command[command.index('SHA_OF_OWN_PREVIOUS_STAGE')]=lineage[-1]['checkpoint_sha256']
  with log.open('x') as f:
   child=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT)
   (root/(stage['name']+'.pid')).write_text(str(child.pid));print(json.dumps(dict(stage=stage['name'],pid=child.pid)),flush=True)
   code=child.wait()
  if code:raise SystemExit(code)
  cp=Path(stage['checkpoint']);lineage.append(dict(name=stage['name'],updates=stage['updates'],checkpoint_path=str(cp),checkpoint_sha256=sha(cp)))
 import torch
 x=torch.load(stages[-1]['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
 assert x['phase']=='sim' and x['iteration']==10000 and x['training_config']['method']==methods(a.base,a.depth)[2]
 assert x['model_config']['base']==a.base and x['model_config']['depth']==a.depth
 binding=dict(schema='architecture_variant_causal_sim_binding_v1',family='causal4k_standard20_architecture_variant',base=a.base,depth=a.depth,phase='sim',iteration=10000,lineage=lineage,checkpoint_path=lineage[-1]['checkpoint_path'],checkpoint_sha256=lineage[-1]['checkpoint_sha256'],status=x['metrics']['status'],teacher_cache_sha256=CACHE_SHA,historical_C_identity_claim=False,training_visibility='shared clean81 Real prefix and released Sim; subsequent held-angle tests are stage CV')
 temporary=root/'architecture_sim_binding.json.tmp';temporary.write_text(json.dumps(binding,indent=2));temporary.replace(root/'architecture_sim_binding.json')
 print(json.dumps(dict(event='architecture_causal_sim_complete',binding=str(root/'architecture_sim_binding.json'))),flush=True)
if __name__=='__main__':main()
