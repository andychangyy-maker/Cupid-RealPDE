import argparse,json,os,subprocess,time,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--name',required=True);p.add_argument('--old-pid',type=int,required=True);a=p.parse_args()
n=Path(__file__).resolve().parent.parent
start=time.monotonic()
while Path('/proc',str(a.old_pid)).exists():
 if time.monotonic()-start>7500:raise TimeoutError('predecessor did not exit')
 time.sleep(5)
c=json.loads((n/a.previous/'complete.json').read_text());assert c['steps']==20000
assert not (n/a.name).exists();assert shutil.disk_usage(n).free>10*2**30
cmd=['/root/miniconda3/bin/python','-u',str(n/'source/run_remaining_fit.py'),'--name',a.name,'--manifest','/root/autodl-tmp/_repro_bundle/kept_metadata/manifest.json']
print(json.dumps({'event':'conditional_start','previous':a.previous,'scope':a.name,'command':cmd}),flush=True)
r=subprocess.run(['timeout','7500']+cmd,check=True)
