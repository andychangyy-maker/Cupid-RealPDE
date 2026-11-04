"""Fixed width32 shared point refiner; reuse the unchanged C Real4000 recipe."""
from dataclasses import replace
import json

import train_b210_transfer_real4k as transfer


INTERVENTION = {
    'name': 'shared_point_feature_refiner', 'hidden': 32,
    'added_parameters': 13682, 'initialization_seed': 31011,
    'initialization': 'identity_zero_output', 'scope': 'finest_point_features',
}


def install(model, torch):
    from models.ensemble_unet2d import SharedPointFeatureRefiner
    assert model.config.base == 210 and model.point_feature_refiner is None
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(INTERVENTION['initialization_seed'])
        refiner = SharedPointFeatureRefiner(model.config.base)
    parameter = next(model.parameters())
    model.point_feature_refiner = refiner.to(device=parameter.device, dtype=parameter.dtype)
    model.config = replace(model.config, point_feature_refiner_channels=32)
    assert sum(p.numel() for p in refiner.parameters()) == INTERVENTION['added_parameters']
    assert torch.count_nonzero(refiner.body[-1].weight) == 0
    assert torch.count_nonzero(refiner.body[-1].bias) == 0


def main():
    args = transfer.parse_args()
    assert args.arm == 'C' and args.full_real_trajectories and args.held_aoa in (15, 20)
    assert args.real_seed == 4 and args.inner_held_aoa is None
    assert not any(getattr(args, name) for name in (
        'vertical_reflection', 'real_time_jitter', 'no_input_corruption', 'sam_real',
        'constant_controller_tke', 'batch_mean_score', 'raw_grid_phase', 'full_real_final'))
    old_fields = transfer.intervention_fields
    transfer.intervention_fields = lambda value: {**old_fields(value), 'point_structure': INTERVENTION}
    if args.execute:
        dense = transfer.load_runtime()
        old_load = dense.ft12.validate_start_checkpoint
        old_train = dense.small4k.train_candidate

        def load(*positional, **keywords):
            model, report = old_load(*positional, **keywords)
            install(model, dense.torch)
            report['point_structure'] = INTERVENTION
            return model, report

        def train(model, *positional, **keywords):
            keywords['expected_parameters'] = sum(p.numel() for p in model.parameters())
            return old_train(model, *positional, **keywords)

        dense.ft12.validate_start_checkpoint = load
        dense.small4k.train_candidate = train
    print(json.dumps(transfer.run(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
