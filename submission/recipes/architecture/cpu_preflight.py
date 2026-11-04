"""CPU-only compatibility checks; not a GPU/quality/throughput claim."""
import argparse,dataclasses,gc,json,sys,hashlib
from pathlib import Path
import torch
import parametric_original as adapter

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=int,required=True);p.add_argument('--depth',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2)
 b=adapter.import_original();original_config=b.b210_config('candidate')
 argv=sys.argv
 sys.argv=['probe','--teacher-cache','dummy.pt','--teacher-cache-sha256','8f73ecbfb792101ce7b7f900947326f8946263be4ce30b5aab0fb8c1fd063301','--output-dir','dummy','--real-closure-bank','closure.pt']
 args=b.parse_args();sys.argv=argv
 before=vars(b.training_args(args))
 funcs={n:getattr(b,n) for n in ['training_args','build_arm_model','configure_trainable']}
 info=adapter.install_architecture(b,a.base,a.depth)
 assert vars(b.training_args(args))==before
 assert all(getattr(b,n) is f for n,f in funcs.items())
 changed=[k for k,v in dataclasses.asdict(original_config).items() if dataclasses.asdict(b.b210_config('candidate'))[k]!=v]
 assert set(changed)<= {'base','depth'}
 model=b.build_arm_model('candidate',torch.device('cpu'))
 assert sum(p.numel() for p in model.parameters())==info['total_parameters']
 native_trainable=b.configure_trainable(model,'candidate')
 assert sum(p.numel() for p in native_trainable)==info['total_parameters']
 model.eval();x=torch.randn(1,20,8,16,3)
 with torch.no_grad():native=model.forward_uq_training(x)
 native_shapes=[list(v.shape) if isinstance(v,torch.Tensor) else None for v in native]
 assert native[0].shape==x.shape
 assert all(torch.isfinite(v).all() for v in native if isinstance(v,torch.Tensor))
 del native
 pre=adapter.configure_stage('pretext',a.base,a.depth);masked=pre.masked
 selected=pre.configure_named_parameters(model,masked.shared_parameter_names(model))
 channels=[model.config.base*2**level for level in range(model.config.depth-2,-1,-1)]
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(pre.SEED+51000);head=masked.ReconstructionReadout(channels,model.config.out_steps,model.config.out_channels)
 model.train();features=model._decode(x);assert len(features)==len(channels)
 y=head(features,(8,16));mask=torch.ones(1,3,dtype=torch.bool)
 loss=pre.balanced_relative_l2(y,torch.randn_like(y),mask);loss.backward()
 finite=sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in selected)
 nonzero=sum(p.grad is not None and bool(p.grad.abs().max()>0) for p in selected)
 assert finite==len(selected) and nonzero>0
 frozen=[p for p in model.parameters() if not p.requires_grad];assert all(p.grad is None for p in frozen)
 # Check the original stage validators with synthetic metadata only. No synthetic checkpoint is saved.
 stage1=adapter.configure_stage('standard1',a.base,a.depth)
 stage1.validate_pretrain_payload(dict(model_state_dict={},phase='sim_real_prefix_causal_pretrain',iteration=4000,training_config={'method':adapter.methods(a.base,a.depth)[0]},pretraining_audit={'changed_untouched_state_tensors':[],'initial_non_backbone_sha256':'same','final_non_backbone_sha256':'same'}))
 stage2=adapter.configure_stage('standard2',a.base,a.depth)
 stage2.validate_stage1(dict(model_state_dict={},iteration=10000,training_config={'method':adapter.methods(a.base,a.depth)[1]}))
 with torch.device('meta'):clone=b.EnsembleUNet2d(b.b210_config('candidate'))
 clone.load_state_dict(model.state_dict(),strict=True,assign=True)
 info.update(cpu_only=True,passed=True,changed_config_fields=changed,original_scientific_training_args_unchanged=True,original_training_functions_unchanged=True,native_output_shapes=native_shapes,temporary_readout_channels=channels,temporary_readout_parameters=sum(p.numel() for p in head.parameters()),shared_parameter_tensors=len(selected),finite_shared_gradients=finite,nonzero_shared_gradients=nonzero,nonshared_gradients_none=True,native_total_parameter_tensors=len(native_trainable),strict_same_structure_reload_pass=True,synthetic_stage_metadata_validator_checks_pass=True,source_sha256=hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest(),scientific_args=before)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(info,indent=2,default=str));print(json.dumps({k:v for k,v in info.items() if k not in ['original_config','actual_config','scientific_args']}))
if __name__=='__main__':main()
