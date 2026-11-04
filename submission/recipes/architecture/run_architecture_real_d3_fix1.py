"""D3-only engineering fix: skip a disabled three-scale Real auxiliary head."""
import torch
import run_architecture_real as original

class DisabledSupervisor(torch.nn.Module):
 def forward(self,*args,**kwargs):
  raise RuntimeError('Disabled auxiliary supervisor must never be forwarded')

_original_install=original.install_real
def install_real(base,depth,runtime_root):
 if depth!=3:raise ValueError('This isolated repair is only for depth3; use original entry for D4')
 b,transfer,dense,uq,native,info=_original_install(base,depth,runtime_root)
 if transfer.ARMS['C'][1] is not False:raise RuntimeError('Repair requires disabled auxiliary C arm')
 def disabled_supervisor(model,device,sim):
  if model.config.depth!=3 or transfer.ARMS['C'][1] is not False:
   raise RuntimeError('Disabled-supervisor repair scope mismatch')
  # Empty construction consumes no CPU/CUDA RNG. Original head construction was
  # isolated with fork_rng; this unused module enters no optimizer, EMA or graph.
  return DisabledSupervisor()
 transfer.fresh_supervisor=disabled_supervisor
 info['engineering_fix']={'id':'d3_disabled_auxiliary_setup_v1',
  'reason':'Original unconditionally instantiated a fixed-three-scale auxiliary head despite C auxiliary being disabled.',
  'auxiliary_enabled':False,'empty_state_audit':True,'scientific_training_unchanged':True,
  'repair_entry_sha256':original.sha(__file__),
  'original_real_adapter_sha256':original.sha(original.__file__)}
 return b,transfer,dense,uq,native,info
original.install_real=install_real
if __name__=='__main__':original.main()