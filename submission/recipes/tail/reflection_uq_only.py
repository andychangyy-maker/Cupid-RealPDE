"""Isolate distilled UQ; restore every original point and energy tensor."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,'/dev/shm/dual_gain_20260923')
import joint_deployed as j
j.ROOT=Path(__file__).resolve().parent


def main(angle,blend=False):
    torch.set_num_threads(4)
    model,stats,teacher,cal,names,identity=j.setup(angle,'080')
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    source=j.ROOT/f'reflection_distill_held{angle}/model.pth'
    trained=torch.load(str(source),map_location='cpu',mmap=True)
    assert trained['normalizer']==identity['normalizer'] and trained['identity']['held']==angle
    prefixes=('uq_projections.','uq_fusion.','quantile_head.','utility_refinement_head.')
    updates={k:v for k,v in trained['model_state_dict'].items() if k.startswith(prefixes)}
    if blend:
        other_source=j.ROOT/f'joint_uq_adaptive080_held{angle}/joint_uq.pth'
        other=torch.load(str(other_source),map_location='cpu',mmap=True)
        assert other['normalizer']==trained['normalizer'] and other['identity']['held']==angle
        assert set(other['model_state_dict'])==set(updates)
        updates={k:torch.lerp(other['model_state_dict'][k].float(),v.float(),.5) for k,v in updates.items()}
    endpoint={**original,**updates};assert all(torch.equal(endpoint[k],v) for k,v in original.items() if not k.startswith(prefixes))
    out=j.ROOT/f'reflection_uq_{"blend" if blend else "only"}_held{angle}';out.mkdir(exist_ok=False)
    identity.update(method='Original 0.80 point with reflection-distilled UQ only',source_sha256=j.sha(source),steps=4000,point_frozen=True)
    if blend:identity.update(method='Equal UQ parameter average: supervised refinement and reflection student',other_source_sha256=j.sha(other_source),steps=None,component_steps=[4000,8000],blend_weights=[.5,.5])
    torch.save(dict(model_state_dict=updates,normalizer=identity['normalizer'],model_config=identity['model_config'],identity=identity),out/'joint_uq.pth')
    (out/'identity.json').write_text(json.dumps(identity,indent=2))
    j.evaluate(angle,model,stats,teacher,cal,names,original,endpoint,out)
    d=json.loads((out/'evaluation.json').read_text())['delta']
    assert max(abs(d[k]) for k in ('tke_rel_l2_score','mvpe_rel_l2_score'))<.003


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--angles',type=int,nargs='*',default=[0,20]);p.add_argument('--full',action='store_true');p.add_argument('--blend',action='store_true');a=p.parse_args()
    for angle in a.angles:main(angle,a.blend)
    if a.full:main(None,a.blend)
