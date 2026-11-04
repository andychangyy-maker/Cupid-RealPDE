import argparse,json,hashlib,importlib.util,sys,time,shutil
from pathlib import Path
from types import SimpleNamespace
import torch
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--manifest',required=True);a=p.parse_args()
base=Path(__file__).resolve().parent;root=base.parent;out=root/a.name;out.mkdir(exist_ok=False)
sys.path.insert(0,str(base))
import train_runtime as recipe
import metric_source as metric
torch.set_num_threads(4)
angle=None if a.name=='full' else int(a.name[4:]);allrows=json.loads(Path(a.manifest).read_text());rows=[r for r in allrows if r['file']!='7575_0.h5' and r['aoa']!=angle]
assert len(rows)==({None:81,0:65,20:65,5:67}[angle]);assert torch.cuda.device_count()==1
runtime=SimpleNamespace(angle=angle,plan={'manifest':a.manifest,'real_dir':'/root/autodl-tmp/realpde/data/real'})
plan=dict(angle=angle,fit_files=[r['file'] for r in rows],steps=20000,batch=32,seed=18618,batch_seed=18619,ema=.999,optimizer='AdamW3e-4 wd1e-4 cosine eta_min3e-5',reflection=.25,initialization='random Real only; no P/Sim dependency',source_sha256={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in ['train.py','train_runtime.py','network.py','metric_source.py']},engineering_changes='7200s training timeout+5GiB disk guard only; original per-iteration BF16 context retained')
(out/'plan.json').write_text(json.dumps(plan,indent=2));started=time.monotonic();recipe.fit(runtime,rows,metric.errors,out)
endpoint=torch.load(out/'ema999_u20000.pth',map_location='cpu',weights_only=False);torch.save(dict(endpoint,inference_dtype='float16'),out/'shape_model.pth')
(out/'complete.json').write_text(json.dumps(dict(steps=20000,seconds=time.monotonic()-started,shape_sha256=hashlib.sha256((out/'shape_model.pth').read_bytes()).hexdigest()),indent=2))
print('SHAPE_COMPLETE',a.name,flush=True)
