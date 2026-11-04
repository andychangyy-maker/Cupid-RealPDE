"""Conditional backup: same C/Real4000 recipe with an equal-Re sample stream."""
import hashlib
import json

import train_b210_transfer_real4k as transfer
from re_balanced_schedule import balanced_batches


CONTROL_STREAM = 'e00aeee3c15d51a48a5401f4c226c336649d0778234ae0fee71953d15dcc2319'
INDEX_STREAM = 'ea8582d3ae597c9aee89eb23637f440e49ad626cf2409354c879fc3c92c699fc'
INTERVENTION = dict(name='equal_re_v1', seed=4, groups=18,
                    support='same_full_other_four_trajectories',
                    within_group='shuffled_cycles_without_replacement')


def replace_schedule(original, windows, trajectory_re, dense):
    assert original['sha256'] == CONTROL_STREAM
    before = dense.torch.get_rng_state().clone()
    window_re = [trajectory_re[window.trajectory] for window in windows]
    batches, report = balanced_batches([len(batch) for batch in original['batches']], window_re)
    assert hashlib.sha256(json.dumps(batches, separators=(',', ':')).encode()).hexdigest() == INDEX_STREAM
    masks = {}
    for row in original['rows']:
        for trajectory, mask in zip(row['trajectory_ids'], row['bootstrap_mask']):
            assert trajectory not in masks or masks[trajectory] == mask
            masks[trajectory] = mask
    assert set(masks) == set(range(64))
    rows = []
    for batch in batches:
        trajectories = [windows[index].trajectory for index in batch]
        rows.append(dict(sample_indices=batch, trajectory_ids=trajectories,
                         bootstrap_mask=[masks[trajectory] for trajectory in trajectories]))
    schedule = dict(batches=batches, rows=rows,
                    masks=[dense.torch.tensor(row['bootstrap_mask'], dtype=dense.torch.bool) for row in rows],
                    sha256=dense.dr1.canonical_sha256(rows))
    assert dense.torch.equal(before, dense.torch.get_rng_state())
    report.update(control_stream_sha256=CONTROL_STREAM, index_stream_sha256=INDEX_STREAM,
                  sample_stream_sha256=schedule['sha256'], torch_cpu_rng_unchanged=True,
                  bootstrap_by_trajectory_unchanged=True)
    return schedule, report


def main():
    args = transfer.parse_args()
    assert args.arm == 'C' and args.full_real_trajectories and args.held_aoa == 15
    assert args.real_seed == 4 and args.inner_held_aoa is None
    assert not any(getattr(args, name) for name in (
        'vertical_reflection', 'real_time_jitter', 'no_input_corruption', 'sam_real',
        'constant_controller_tke', 'batch_mean_score', 'raw_grid_phase', 'full_real_final'))
    actual = {}
    old_fields = transfer.intervention_fields
    transfer.intervention_fields = lambda value: {
        **old_fields(value), 're_balanced_sampling': {**INTERVENTION, **actual}}
    if args.execute:
        dense = transfer.load_runtime()
        assert dense.dr1.sha256(args.clean_manifest) == dense.ft12.FORMAL_MANIFEST_FILE_SHA256
        rows = json.loads(args.clean_manifest.read_text())
        fit = [row for row in rows if row['aoa'] != 15]
        assert len(fit) == 64
        original_schedule = dense.smu1.paired_batch_schedule
        def schedule(fit_indices, windows, updates, seed):
            assert list(fit_indices) == list(range(10380)) and updates == 4000 and seed == 4
            old = original_schedule(fit_indices, windows, updates=updates, seed=seed)
            new, report = replace_schedule(old, windows, [row['re'] for row in fit], dense)
            actual.update(report)
            return new
        dense.smu1.paired_batch_schedule = schedule
    print(json.dumps(transfer.run(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
