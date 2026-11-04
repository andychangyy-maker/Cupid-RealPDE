"""Combine the sealed quality checkpoint with verified scan3, then one copy-order candidate."""
import argparse,hashlib,json,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--quality',type=Path,required=True);p.add_argument('--scan3',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(a.quality)=='01c3af6059c9d63138091ba7332560de62260c93d46e3990536274bdb93d469f'
assert sha(a.scan3)=='bf146947f77d96bb08af82e522e7c4b1a518a7edfe24b6d7f6c5d89f871d3107'
a.out.mkdir(exist_ok=True)
with zipfile.ZipFile(a.quality) as q,zipfile.ZipFile(a.scan3) as s:
 files={n:q.read(n) for n in q.namelist()}
 for n in ('submission.py','models/__init__.py','models/unet2d.py'):assert files[n]==s.read(n)
 files['models/ensemble_unet2d.py']=s.read('models/ensemble_unet2d.py')
 results={}
 for arm in ('combined','halfcopy'):
  if arm=='halfcopy':
   text=files['submission.py'].decode();assert text.count('.float().cpu().numpy()')==1
   text=text.replace('.float().cpu().numpy()','.cpu().float().numpy()');compile(text,'submission.py','exec');files['submission.py']=text.encode()
  dst=a.out/(arm+'.zip')
  with zipfile.ZipFile(dst,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
   for n,data in files.items():z.writestr(n,data)
  with zipfile.ZipFile(dst) as z:assert z.testzip() is None and z.read('ensemble_unet2d.pth')==q.read('ensemble_unet2d.pth')
  results[arm]={'path':str(dst),'sha256':sha(dst),'bytes':dst.stat().st_size,'member_sha256':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}}
(a.out/'combinations.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
