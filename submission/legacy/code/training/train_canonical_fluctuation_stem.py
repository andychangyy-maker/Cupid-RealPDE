"""One fixed normalized-fluctuation input branch, with full C Real4000 training."""
import json

import train_b210_transfer_real4k as transfer


INTERVENTION = dict(name='canonical_fluctuation_stem_v1', added_parameters=9030,
                    input_steps=20, shape_channels=40, amplitude_channels=2,
                    normalized_rms_floor_squared=1e-6, projection_kernel=1,
                    initialization='zero_projection', full_model_updates=4000)


def main():
    args = transfer.parse_args()
    assert args.arm == 'C' and args.full_real_trajectories and args.held_aoa == 15
    assert args.real_seed == 4 and args.inner_held_aoa is None
    assert not any(getattr(args, name) for name in (
        'vertical_reflection', 'real_time_jitter', 'no_input_corruption', 'sam_real',
        'constant_controller_tke', 'batch_mean_score', 'raw_grid_phase', 'full_real_final'))
    old_fields = transfer.intervention_fields
    transfer.intervention_fields = lambda value: {**old_fields(value), 'canonical_fluctuation_stem': INTERVENTION}
    if args.execute:
        dense = transfer.load_runtime()
        from models.canonical_fluctuation_stem import install
        old_load, old_train = dense.ft12.validate_start_checkpoint, dense.small4k.train_candidate
        def load(*positional, **keywords):
            model, report = old_load(*positional, **keywords)
            install(model)
            report['canonical_fluctuation_stem'] = INTERVENTION
            return model, report
        def train(model, *positional, **keywords):
            keywords['expected_parameters'] = 178714605
            return old_train(model, *positional, **keywords)
        dense.ft12.validate_start_checkpoint = load
        dense.small4k.train_candidate = train
    print(json.dumps(transfer.run(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
