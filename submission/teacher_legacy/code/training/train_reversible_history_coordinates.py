"""Full Real4000 with a fixed reversible change of input/output coordinates."""
import json

import train_b210_transfer_real4k as transfer

INTERVENTION = dict(name='reversible_history_coordinates_v1', added_parameters=0,
                    fixed_buffer_values=4, input_steps=20, normalized_variance_floor=1e-6,
                    statistics_axes='time_height_width_per_sample_per_uv', full_model_updates=4000,
                    output_units='original_target_normalizer', learned_affine=False)


def main():
    args = transfer.parse_args()
    assert args.arm == 'C' and args.full_real_trajectories and args.held_aoa == 15
    assert args.real_seed == 4 and args.inner_held_aoa is None
    assert not any(getattr(args, name) for name in (
        'vertical_reflection', 'real_time_jitter', 'no_input_corruption', 'sam_real',
        'constant_controller_tke', 'batch_mean_score', 'raw_grid_phase', 'full_real_final'))
    old_fields = transfer.intervention_fields
    transfer.intervention_fields = lambda value: {**old_fields(value), 'reversible_history_coordinates': INTERVENTION}
    if args.execute:
        dense = transfer.load_runtime()
        from models.reversible_history_coordinates import install
        old_train = dense.small4k.train_candidate
        def train(model, dataset, schedule, normalizer, device, deadline, output_dir, start_report, access, **kwargs):
            install(model, normalizer)
            report = {**start_report, 'reversible_history_coordinates': INTERVENTION}
            return old_train(model, dataset, schedule, normalizer, device, deadline, output_dir, report, access, **kwargs)
        dense.small4k.train_candidate = train
    print(json.dumps(transfer.run(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
