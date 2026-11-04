"""Real runtime import/setup only, no checkpoint, data, optimizer, CUDA or training."""
import argparse,json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import run_architecture_real_foreach_audit as entry

p=argparse.ArgumentParser();p.add_argument('--base',type=int,required=True);p.add_argument('--depth',type=int,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2)
original=entry.configure(a.depth)
state=torch.get_rng_state().clone()
result=original.install_real(a.base,a.depth,original.NEW/'point_recipe')
b,transfer,dense,uq,native,info=result
assert torch.equal(state,torch.get_rng_state()),'Installation consumed CPU RNG'
audit,gate=entry.load_verified_audit()
loop=dense.small4k.train_candidate
assert loop.__globals__['npt1']._gradient_audit is audit.foreach_gradient_audit
# Small real CPU tensors exercise installed function through the loop's actual global binding.
class Tiny:
    def __init__(self):
        self.rows=[]
        for name,value in [('stem.weight',1.),('energy_controller.body.0.weight',0.),
                           ('energy_controller.body.5.weight',-2.)]:
            x=torch.nn.Parameter(torch.zeros(3));x.grad=torch.full_like(x,value);self.rows.append((name,x))
    def named_parameters(self):return iter(self.rows)
tiny=Tiny();before=[(p.grad.clone(),p.grad._version) for _,p in tiny.rows]
expected=audit._gradient_audit(tiny)
observed=loop.__globals__['npt1']._gradient_audit(tiny)
assert expected==observed
assert all(torch.equal(p.grad,g) and p.grad._version==v for (_,p),(g,v) in zip(tiny.rows,before))
assert torch.equal(state,torch.get_rng_state())
if a.depth==3:
    with torch.device('meta'):model=b.EnsembleUNet2d(b.b210_config('candidate'))
    assert transfer.fresh_supervisor(model,torch.device('cpu'),b).state_dict()=={}
report=dict(status='PASS',base=a.base,depth=a.depth,actual_original_runtime_import=True,
            point_loop_binding_verified=True,ft12_alias_verified=True,
            tiny_CPU_result_equal=True,grad_bytes_versions_rng_unchanged=True,
            d3_disabled_auxiliary_preserved=a.depth==3,
            gpu_used=False,training_started=False,metadata=info['read_only_gradient_audit_engineering'])
a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))
