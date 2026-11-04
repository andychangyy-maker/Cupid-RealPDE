"""Actual B192D3 Sim checkpoint + full-fit normalizer CPU setup check."""
import json,hashlib
from pathlib import Path
import torch
import run_architecture_real_d3_fix1 as fix
from run_architecture_real import NEW

def main():
 torch.set_num_threads(2)
 root=NEW/'architecture_preparation';candidate=root/'candidates/b192d3'
 binding=json.loads((candidate/'architecture_sim_binding.json').read_text())
 sim=Path(binding['checkpoint_path']);normpath=NEW/'point/full/point_normalizer.pt'
 b,t,d,uq,native,info=fix.install_real(192,3,NEW/'point_recipe')
 stats=torch.load(normpath,map_location='cpu',weights_only=False)
 normalizer=d.smu1.FoldNormalizer(stats,torch.device('cpu'))
 model,start=d.ft12.validate_start_checkpoint(sim,binding['checkpoint_sha256'],normalizer,torch.device('cpu'),expected_iteration=binding['iteration'],expected_status=binding['status'])
 rng=torch.get_rng_state().clone()
 supervisor=t.fresh_supervisor(model,torch.device('cpu'),b)
 assert torch.equal(rng,torch.get_rng_state())
 assert len(supervisor.state_dict())==0 and len(tuple(supervisor.parameters()))==0
 emptysha=b.state_sha256(supervisor.state_dict())
 aux=t.auxiliary_arguments('C',supervisor,b)
 assert aux['full_backbone_auxiliary'] is None and aux['full_backbone_auxiliary_weight']==0 and aux['full_backbone_gradient_audit_update']==0
 try:supervisor(torch.zeros(1))
 except RuntimeError as e:assert 'must never' in str(e)
 else:raise AssertionError('Disabled supervisor unexpectedly forwarded')
 params=d.ft12.configure_trainable(model)
 model.train();out=model.forward_uq_training(torch.randn(1,20,8,16,3))
 loss=sum(v.square().mean() for v in out if isinstance(v,torch.Tensor));loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
 result=dict(passed=True,cpu_only=True,actual_sim=start,normalizer_path=str(normpath),normalizer_sha256=fix.original.sha(normpath),depth=3,base=192,empty_supervisor_state_sha256=emptysha,cpu_rng_unchanged=True,auxiliary_arguments={k:v for k,v in aux.items() if k!='full_backbone_auxiliary'},auxiliary_is_none=True,forward_explicitly_forbidden=True,finite_gradients=len(params),nonzero_gradients=sum(bool(p.grad.abs().max()>0) for p in params),no_training_launched=True,fix_sha256=fix.original.sha(fix.__file__))
 (root/'parametric_recipe/cpu_real_d3_fix1_actual.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()