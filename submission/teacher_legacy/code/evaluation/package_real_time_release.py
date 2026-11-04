"""Seal final full-Real ZIP; clean716 replay is overlapping deployment diagnostics."""
import argparse,dataclasses,io,json,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT/'code/evaluation',ROOT/'code/submissions',ROOT/'code/training',ROOT/'code'):sys.path.insert(0,str(p))
import torch
import evaluate_b210_transfer as replay
import train_b210_transfer_uq as uq
from build_track1_blended_ensemble_unet2d_package import compact_state_dict
from models.ensemble_unet2d import EnsembleUNet2dConfig

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
 a=p.parse_args();cfg=json.loads(a.config.read_text());run=Path(cfg['output']);out=run.parent/'package'
 out.mkdir(exist_ok=False);started=time.monotonic();torch.set_num_threads(4)
 receipt=json.loads((run/'uq/uq_endpoint.json').read_text());point=receipt['point_receipt']
 uq.validate_point_receipt(point,'C',True,True)
 promotion=None
 if cfg.get('vertical_reflection'):
  assert replay.sha256(cfg['promotion_result'])==cfg['promotion_sha256']
  promotion=json.loads(Path(cfg['promotion_result']).read_text())
  assert promotion['decision']=='PASS_REFLECTION_FIVE_FOLD_FINAL_TRAINING_PENDING'
  assert point['vertical_reflection']=={'probability':.25,'isolated_cpu_seed':29009,'space':'physical_uvp_before_normalization'}
 assert receipt['point_state_bitwise'] and receipt['tail_arrays_read']==0
 assert receipt['sample_stream_sha256']==uq.UQ_STREAM_SHA and receipt['fit_windows']==2544
 assert receipt['updates']==3200 and receipt['seed']==18 and receipt['initialization']=='point_checkpoint'
 assert replay.sha256(receipt['checkpoint_path'])==receipt['checkpoint_sha256']
 assert replay.sha256(cfg['baseline_zip'])==cfg['baseline_sha256']
 payload=torch.load(receipt['checkpoint_path'],map_location='cpu',weights_only=False)
 with zipfile.ZipFile(cfg['baseline_zip']) as z:
  assert set(z.namelist())==set(replay.RUNTIME_FILES)|{'ensemble_unet2d.pth'}
  files={n:z.read(n) for n in z.namelist()}
 template=torch.load(io.BytesIO(files['ensemble_unet2d.pth']),map_location='cpu',weights_only=False)
 def config(c):
  c=dict(c);assert c.pop('architecture')=='ensemble_unet';return dataclasses.asdict(EnsembleUNet2dConfig(**c))
 assert config(payload['model_config'])==config(template['model_config'])
 assert replay.canonical_sha256(payload['normalizer'])==replay.canonical_sha256(template['normalizer'])
 compact,stats=compact_state_dict(payload['model_state_dict'],1024)
 state={'model_config':template['model_config'],'quantized_state_dict':compact,'normalizer':payload['normalizer'],
        'phase':'real','inference_dtype':'float16','storage_format':'symmetric_per_dim0_int8_v1'}
 buffer=io.BytesIO();torch.save(state,buffer);checkpoint=buffer.getvalue()
 dst=out/'quality.zip'
 with zipfile.ZipFile(dst,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for n,data in files.items():z.writestr(n,checkpoint if n=='ensemble_unet2d.pth' else data)
 del payload,compact,state,template,buffer
 with zipfile.ZipFile(dst) as z:
  assert z.testzip() is None
  assert z.read('ensemble_unet2d.pth')==checkpoint
  assert all(z.read(n)==files[n] for n in replay.RUNTIME_FILES)
 assert dst.stat().st_size<256*1024**2
 results={}
 for label,path in [('baseline1',Path(cfg['baseline_zip'])),('candidate1',dst),('baseline2',Path(cfg['baseline_zip'])),('candidate2',dst)]:
  runtime=out/label;runtime.mkdir()
  with zipfile.ZipFile(path) as z:z.extractall(runtime)
  result=out/(label+'.json')
  with (out/(label+'.log')).open('w') as log:
   subprocess.run([sys.executable,str(ROOT/'code/evaluation/evaluate_b210_transfer.py'),'--execute',
    '--worker-runtime',str(runtime),'--worker-label',label,'--worker-output',str(result),
    '--device','cuda:0','--clean-manifest',cfg['clean-manifest'],'--real-dir',cfg['real-dir'],
    *(['--study','real-reflection'] if cfg.get('vertical_reflection') else [])],
    check=True,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,900-(time.monotonic()-started)))
  results[label]=json.loads(result.read_text())
 for key in ('input_sha256','target_sha256','scorer_sha256'):assert len({r[key] for r in results.values()})==1
 b,c=results['baseline1'],results['candidate1']
 metrics=list(b['metrics']);delta={k:c['metrics'][k]-b['metrics'][k] for k in metrics}
 aoa={k:c['by_aoa'][k]['fixed_time_fitted_final']-b['by_aoa'][k]['fixed_time_fitted_final'] for k in b['by_aoa']}
 jitter={k:abs(results['baseline2']['metrics'][k]-b['metrics'][k]) for k in metrics}
 repeat={k:abs(results['candidate2']['metrics'][k]-c['metrics'][k]) for k in metrics}
 stable=all(repeat[k]<=max(2*jitter[k],1e-4) for k in metrics)
 passed=stable and delta['fixed_time_fitted_final']>=0 and sum(v>=0 for v in aoa.values())>=4
 identity={'decision':'PASS_QUALITY_ZIP_DEPLOYMENT_DIAGNOSTICS' if passed else 'STOP_FINAL_QUALITY_ZIP',
  'zip_path':str(dst),'zip_sha256':replay.sha256(dst),'zip_bytes':dst.stat().st_size,
  'source_commit':cfg['source_commit'],'uq_checkpoint_sha256':receipt['checkpoint_sha256'],
  'same_runtime_as_baseline':True,'runtime_baseline_sha256':cfg['baseline_sha256'],
  'same_prefix_normalizer':True,'same_uq_stream':True,
  'quality_delta_overlapping_diagnostic':delta,'by_aoa_overlapping_diagnostic':aoa,
  'baseline_repeat_jitter':jitter,'candidate_repeat_jitter':repeat,'stable':stable,
  'clean716_overlaps_point_training':True,'independent_validation':False,
  'promotion_basis':(promotion if promotion else 'previous five-fold Real/UQ-stage LOAO +0.09223115; shared Sim ancestry'),
  'official_gain':None,'elapsed_seconds':time.monotonic()-started,'results':results,'quantization':stats}
 (out/'RESULT.json').write_text(json.dumps(identity,indent=2)+'\n')
 print(json.dumps({k:v for k,v in identity.items() if k!='results'},indent=2))

if __name__=='__main__':main()
