"""Insert nonlinear member adapters at the C-to-Real boundary; teacher only."""
from dataclasses import replace
import json

import train_b210_transfer_real4k as transfer


INTERVENTION = dict(name='fresh_real_member_teacher_v1', members=4, scales=3,
                    hidden=64, added_parameters=1650168, initialization_seed=32011,
                    initialization='zero_final_projection', deployable=False)


def install(model, torch):
    from models.ensemble_unet2d import MemberFeatureAdapter
    assert model.config.ensemble_members == 4 and model.member_feature_adapters is None
    assert model.config.member_adapter_channels == 0 and not model.config.mean_preserving_member_adapters
    assert model.point_feature_refiner is None and model.member_point_head_adapters is None
    before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    old = {name:value.detach().clone() for name,value in model.state_dict().items()}
    widths = [head.in_channels for head in model.member_point_heads[0]]
    assert len(widths) == 3
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(INTERVENTION['initialization_seed'])
        adapters = torch.nn.ModuleList(torch.nn.ModuleList(
            MemberFeatureAdapter(width, 64, model.config.groups) for width in widths) for _ in range(4))
    p = next(model.parameters())
    model.member_feature_adapters = adapters.to(device=p.device,dtype=p.dtype)
    model.config = replace(model.config, member_adapter_channels=64)
    assert sum(p.numel() for p in adapters.parameters()) == 1650168
    assert sum(p.numel() for p in model.parameters()) == 180355743
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in old.items())
    assert all(torch.count_nonzero(adapter.body[-1].weight)==0 and torch.count_nonzero(adapter.body[-1].bias)==0
               for member in adapters for adapter in member)
    assert torch.equal(before,torch.get_rng_state())
    if cuda_before:
        assert all(torch.equal(a,b) for a,b in zip(cuda_before,torch.cuda.get_rng_state_all()))


def main():
    args = transfer.parse_args()
    assert args.arm=='C' and args.full_real_trajectories and args.held_aoa==15
    assert args.real_seed==4 and args.inner_held_aoa is None
    assert not any(getattr(args,name) for name in (
        'vertical_reflection','real_time_jitter','no_input_corruption','sam_real',
        'constant_controller_tke','batch_mean_score','raw_grid_phase','full_real_final'))
    args.real_member_teacher = True
    old_fields = transfer.intervention_fields
    transfer.intervention_fields = lambda value:{**old_fields(value),'real_member_teacher':INTERVENTION}
    if args.execute:
        dense = transfer.load_runtime()
        old_load,old_train = dense.ft12.validate_start_checkpoint,dense.small4k.train_candidate
        def load(*positional,**keywords):
            model,report = old_load(*positional,**keywords)
            install(model,dense.torch)
            report['real_member_teacher'] = INTERVENTION
            return model,report
        def train(model,*positional,**keywords):
            keywords['expected_parameters'] = 180355743
            return old_train(model,*positional,**keywords)
        dense.ft12.validate_start_checkpoint = load
        dense.small4k.train_candidate = train
    print(json.dumps(transfer.run(args),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
