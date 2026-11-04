"""Change only Real-stage coupled SPS weight from 0.05 to 0.80."""
import argparse
import json
import sys
from pathlib import Path


def main(args):
    import torch
    torch.set_num_threads(4)
    from coupling import install
    root = args.plan.parent
    plan = json.loads(args.plan.read_text())
    src = Path(plan['source_root'])
    sys.path[:0] = [str(root), str(src / 'code/training'), str(src / 'code')]
    import point_runtime as transfer
    import train_hybrid_fno2d as common
    dense = transfer.load_runtime()
    small = dense.small4k
    transfer.UPDATES = small.UPDATES = 5334
    small.PRIMARY_ENDPOINT = 'ema_u5334'
    original_train = small.train_candidate
    original_fields = transfer.intervention_fields
    term = {'coefficient': 1.0, 'weights': [5, 1, 1], 'batch_mean_errors': True}
    calls = [0]
    schedule = dict(original_weight=.05, weight=.80, warmup_updates=1500, tau_fraction=.05)
    assert small.npt1.joint_real.REAL_SOFT_SPS_WEIGHT == schedule['original_weight']
    assert small.npt1.joint_real.REAL_SOFT_SPS_WARMUP_UPDATES == schedule['warmup_updates']
    assert small.npt1.joint_real.REAL_SOFT_SPS_TAU_FRACTION == schedule['tau_fraction']
    small.npt1.joint_real.REAL_SOFT_SPS_WEIGHT = schedule['weight']
    print(json.dumps({'sps_weight_intervention': schedule}), flush=True)

    def fields(options):
        assert options.vertical_reflection and transfer.UPDATES == 5334
        return {**original_fields(options),
                'full_sps_coupling': 'native SPS differentiates point center and factor; all parameters; Real weight 0.80',
                'additional_final_score_term': term,
                'real_sps_weight_intervention': schedule}

    def final_objective(history, prediction, physical_target, target_norm, normalizer):
        loss, _ = common.accuracy_objective(prediction, target_norm, normalizer,
                                           (5., 1., 1.), batch_mean_errors=True)
        assert loss.requires_grad
        calls[0] += 1
        if calls[0] in [1, 2] or calls[0] % 1000 == 0:
            print(json.dumps({'final_score_term_call': calls[0],
                              'loss': float(loss.detach()), 'requires_grad': True}), flush=True)
        return loss, {'final_score_term_loss': float(loss.detach())}

    def train(*values, **kwargs):
        install(small.npt1.ensemble_train, root / key)
        kwargs.update(endpoint_filename='ema999_u5334_primary.pth',
                      point_accuracy_weights=(5., 3., 2.), member_accuracy_weights=(5., 3., 2.),
                      region_auxiliary=final_objective, region_auxiliary_weight=1.0)
        result = original_train(*values, **kwargs)
        return result

    transfer.intervention_fields = fields
    small.train_candidate = train
    key = 'full' if args.full_fit else f'held{args.angle}'
    command = ['--execute', '--arm', 'C', '--full-real-trajectories', '--vertical-reflection',
               '--source-commit', plan['source_commit_label'], '--start-sim', plan['sim'],
               '--start-sim-sha256', plan['sim_sha256'], '--start-metadata', plan['sim_metadata'],
               '--start-metadata-sha256', plan['sim_metadata_sha256'], '--clean-manifest', plan['manifest'],
               '--real-dir', plan['real_dir'], '--output-dir', str(root / key),
               '--wall-budget-seconds', str(plan['point_wall_budget_seconds']), '--device', 'cuda:0']
    command += ['--full-real-final'] if args.full_fit else ['--held-aoa', str(args.angle)]
    result = transfer.run(transfer.parse_args(command))
    assert calls[0] == 5334, ('objective not applied at every update', calls[0])
    assert small.npt1.joint_real.REAL_SOFT_SPS_WEIGHT == .80
    (root / key / 'additional_objective.json').write_text(json.dumps({
        'calls': calls[0], **term, 'original_point_member_weights': [5, 3, 2]}, indent=2))
    print(json.dumps({'key': key, 'elapsed_seconds': result['elapsed_seconds'],
                      'checkpoint_sha256': result['seal']['file_sha256']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--angle', type=int, choices=[0, 5, 10, 15, 20])
    mode.add_argument('--full-fit', action='store_true')
    main(parser.parse_args())
