import argparse,json,sys,hashlib,shutil,os
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser()
p.add_argument('--sim',type=Path,required=True);p.add_argument('--binding',type=Path,required=True)
p.add_argument('--scope',choices=['full','0','5','10','15','20'],required=True)
p.add_argument('--stage',choices=['point','uq'],required=True)
p.add_argument('--output-root',type=Path,required=True)
a=p.parse_args()
origin=Path(__file__).parent
root=a.output_root;root.mkdir(parents=True,exist_ok=True)
for n in ['train.py','coupling.py','point_runtime.py','train_matched_uq.py','control_uq_receipts.json']:
 dst=root/n
 if not dst.exists():shutil.copy2(origin/n,dst)
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
sim_sha,binding_sha=sha(a.sim),sha(a.binding)
binding=json.loads(a.binding.read_text())
if binding['checkpoint_sha256'] != sim_sha:raise ValueError('binding does not identify selected newly trained Sim endpoint')
plan=dict(source_root='/root/autodl-tmp/realpde_research_20260922/source',
source_commit_label='5b28cf57f609fc2fe481d8f08818a5a0a7d9ba76',
sim=str(a.sim.resolve()),sim_sha256=sim_sha,sim_metadata=str(a.binding.resolve()),sim_metadata_sha256=binding_sha,
manifest='/root/autodl-tmp/rp_repro_20260917/track1_gain_20260916_h1/manifest.json',
real_dir='/root/autodl-tmp/realpde/data/real',point_wall_budget_seconds=3600,
reproduction_note='Original .80 scientific recipe; start SHA constants bind the explicitly selected Sim identity. Historical-reference and newly-trained-Sim outputs live in separate roots.')
tmp=root/('plan.'+str(os.getpid())+'.tmp')
tmp.write_text(json.dumps(plan,indent=2));tmp.replace(root/'plan.json')
sys.path[:0]=[str(root),plan['source_root']+'/code/training',plan['source_root']+'/code']
full=a.scope=='full';angle=None if full else int(a.scope)
if a.stage=='point':
 import point_runtime
 point_runtime.C_START_SIM_SHA256=sim_sha
 point_runtime.C_START_METADATA_SHA256=binding_sha
 import train
 train.main(SimpleNamespace(plan=root/'plan.json',full_fit=full,angle=angle))
else:
 import train_matched_uq
 train_matched_uq.main(root,angle,False,full)
