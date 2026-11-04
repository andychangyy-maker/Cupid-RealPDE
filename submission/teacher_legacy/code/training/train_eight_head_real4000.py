"""Train eight bootstrap heads, then evaluate a separately sealed four-head fold."""
import copy
from dataclasses import replace
import json

import train_b210_transfer_real4k as transfer


CONTROL_STREAM = 'e00aeee3c15d51a48a5401f4c226c336649d0778234ae0fee71953d15dcc2319'
INTERVENTION = dict(name='eight_train_four_deploy_v1', training_members=8,
                    deployed_members=4, extra_bootstrap_seed=41011,
                    bootstrap_probability=.75, added_parameters=371196,
                    initialization='duplicate_0_1_2_3', folding_pairs=[[i, i+4] for i in range(4)])


def install(model, torch):
    assert model.config.ensemble_members == 4 and len(model.member_point_heads) == 4
    assert model.member_feature_adapters is None and model.member_point_head_adapters is None
    assert model.point_feature_refiner is None
    before = torch.get_rng_state().clone()
    device_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.member_point_heads.extend(copy.deepcopy(list(model.member_point_heads)))
    model.config = replace(model.config, ensemble_members=8)
    state = model.state_dict()
    assert all(torch.equal(state[name], value) for name, value in original.items())
    for i in range(4):
        assert all(torch.equal(value, model.member_point_heads[i+4].state_dict()[name])
                   for name, value in model.member_point_heads[i].state_dict().items())
    added = sum(p.numel() for heads in model.member_point_heads[4:] for p in heads.parameters())
    assert added == INTERVENTION['added_parameters']
    assert sum(p.numel() for p in model.parameters()) == 178705575 + added
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(device_states, torch.cuda.get_rng_state_all())) if device_states else True


def fold_model(model, torch):
    from models.ensemble_unet2d import EnsembleUNet2d
    assert model.config.ensemble_members == 8 and len(model.member_point_heads) == 8
    state = dict(model.state_dict())
    for i in range(4):
        for suffix in model.member_point_heads[i].state_dict():
            first, second = f'member_point_heads.{i}.{suffix}', f'member_point_heads.{i+4}.{suffix}'
            state[first] = (state[first] + state.pop(second)) * .5
    folded = EnsembleUNet2d(replace(model.config, ensemble_members=4)).float()
    folded.load_state_dict(state, strict=True)
    assert sum(p.numel() for p in folded.parameters()) == 178705575
    return folded


def extended_schedule(old, dense):
    assert old['sha256'] == CONTROL_STREAM
    before = dense.torch.get_rng_state().clone()
    rows, masks = [], []
    for row in old['rows']:
        original = dense.torch.tensor(row['bootstrap_mask'], dtype=dense.torch.bool)
        extra = dense.ft12.npt1.ensemble_train.trajectory_member_mask(
            dense.torch.tensor(row['trajectory_ids']), .75, INTERVENTION['extra_bootstrap_seed'], 4)
        mask = dense.torch.cat((original, extra), dim=1)
        masks.append(mask)
        rows.append({**row, 'bootstrap_mask': mask.to(dense.torch.uint8).tolist()})
    assert dense.torch.equal(before, dense.torch.get_rng_state())
    assert any(row['bootstrap_mask'][0][:4] != row['bootstrap_mask'][0][4:] for row in rows)
    projected = [{**row, 'bootstrap_mask': [mask[:4] for mask in row['bootstrap_mask']]} for row in rows]
    assert projected == old['rows']
    return dict(batches=old['batches'], rows=rows, masks=masks, sha256=dense.dr1.canonical_sha256(rows))


def main():
    args = transfer.parse_args()
    assert args.arm == 'C' and args.full_real_trajectories and args.held_aoa == 15
    assert args.real_seed == 4 and args.inner_held_aoa is None
    assert not any(getattr(args, name) for name in (
        'vertical_reflection', 'real_time_jitter', 'no_input_corruption', 'sam_real',
        'constant_controller_tke', 'batch_mean_score', 'raw_grid_phase', 'full_real_final'))
    actual = {}
    old_fields = transfer.intervention_fields
    transfer.intervention_fields = lambda value: {**old_fields(value), 'eight_head_training': {**INTERVENTION, **actual}}
    if args.execute:
        dense = transfer.load_runtime()
        old_load, old_train = dense.ft12.validate_start_checkpoint, dense.small4k.train_candidate
        old_schedule = dense.smu1.paired_batch_schedule
        def load(*positional, **keywords):
            model, report = old_load(*positional, **keywords)
            install(model, dense.torch)
            report['eight_head_training'] = INTERVENTION
            return model, report
        def train(model, *positional, **keywords):
            keywords.update(expected_parameters=179076771, expected_member_count=8)
            return old_train(model, *positional, **keywords)
        def schedule(*positional, **keywords):
            value = extended_schedule(old_schedule(*positional, **keywords), dense)
            actual.update(sample_stream_sha256=value['sha256'], original_four_projection_sha256=CONTROL_STREAM)
            return value
        dense.ft12.validate_start_checkpoint = load
        dense.small4k.train_candidate = train
        dense.smu1.paired_batch_schedule = schedule
    print(json.dumps(transfer.run(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
