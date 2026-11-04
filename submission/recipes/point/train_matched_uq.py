"""Fit native UQ to a sealed coupled point model, without a mean-loss shift."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import torch


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main(root, angle, preflight, full_fit=False):
    assert (angle is None) == full_fit
    torch.set_num_threads(4)
    plan = json.loads((root/'plan.json').read_text())
    source = Path(plan['source_root'])/'code'
    sys.path[:0] = [str(source/'training'), str(source)]
    import train_b210_transfer_uq as uq
    import models.ensemble_unet2d as ensemble
    output = root/('full' if full_fit else f'held{angle}')
    control_key = 'full' if full_fit else str(angle)
    endpoint = output/'ema999_u5334_primary.pth'
    binding_path = output/'point_endpoint.json'
    receipt = json.loads(binding_path.read_text())
    assert sha(endpoint) == receipt['checkpoint_sha256']
    point = torch.load(endpoint, map_location='cpu', weights_only=False, mmap=True)

    def validate(record, label, full_real_point=False, full_real_final=False, expected_updates=4000):
        assert record == receipt and label == 'C' and full_real_point and full_real_final == full_fit
        assert record['schema'] == uq.POINT_RECEIPT_SCHEMA and record['held_aoa'] == angle
        assert record['updates'] == 5334 and record['ema_decay'] == .999 and record['seed'] == 4
        assert record['start_sim_sha256'] == plan['sim_sha256']
        assert record['start_metadata_sha256'] == plan['sim_metadata_sha256']
        assert record['full_sps_coupling'] == 'native SPS differentiates point center and factor; all parameters; Real weight 0.80'
        assert record['real_sps_weight_intervention'] == dict(original_weight=.05, weight=.80, warmup_updates=1500, tau_fraction=.05)
        assert record['vertical_reflection'] == dict(probability=.25, isolated_cpu_seed=29009,
                                                     space='physical_uvp_before_normalization')
        assert record['tail_read'] is True
        if full_fit:
            assert record['held_data_read'] is None
            assert record['optimizer_support'] == 'full_clean81_trajectories'
            assert record['normalizer_support'] == 'clean81_prefix2544'
            assert record['final_fit'] and record['clean716_overlaps_point_training']
            assert len(record['fit_files']) == 81 and '7575_0.h5' not in record['fit_files']
        else:
            assert record['held_data_read'] is False
            assert record['optimizer_support'] == 'full_other_four_trajectories'
            assert record['normalizer_support'] == 'other_four_prefixes'
            assert all(int(Path(n).stem.split('_')[-1]) != angle for n in record['fit_files'])
        assert point['start']['full_sps_coupling'] == record['full_sps_coupling']
        assert point['start']['additional_final_score_term'] == record['additional_final_score_term']

    uq.validate_point_receipt = validate
    validate(receipt, 'C', True, full_fit)
    uq.validate_point_payload(point, receipt)
    config = dict(point['model_config']); assert config.pop('architecture') == 'ensemble_unet'
    assert set(config) <= {field.name for field in fields(ensemble.EnsembleUNet2dConfig)}
    ensemble.EnsembleUNet2dConfig(**config)
    controls = json.loads((root/'control_uq_receipts.json').read_text())
    expected = controls.get(control_key, dict(updates=3200, seed=18, batch_size=8))
    command = ['--execute', '--label', 'full', '--point-label', 'C', '--initialization', 'point',
               '--full-real-point', '--uq-full-time', '--uq-phase-jitter', '--uq-physical-reflection', '--compact-uq',
               '--source-commit', plan['source_commit_label'], '--point-receipt', str(binding_path),
               '--point-receipt-sha256', sha(binding_path), '--clean-manifest', plan['manifest'],
               '--real-dir', plan['real_dir'], '--output-dir', str(output/'matched_uq'),
               '--wall-budget-seconds', '1800', '--device', 'cuda:0']
    command += ['--full-real-final'] if full_fit else ['--held-aoa', str(angle)]
    args = uq.parse_args(command)
    proof = dict(angle=angle, checkpoint_sha256=sha(endpoint), receipt_sha256=sha(binding_path),
                 training_script_sha256=sha(__file__), reused_uq_script_sha256=sha(uq.__file__),
                 expected_sampling=expected, original_sampling_receipt_available=control_key in controls,
                 full_fit=full_fit, loss_prediction_shifted=False, preflight_passed=True)
    (output/'matched_uq_preflight.json').write_text(json.dumps(proof, indent=2))
    if preflight:
        print(json.dumps(proof), flush=True)
        return
    assert torch.cuda.device_count() == 1
    result = uq.execute(args)
    for key, value in expected.items():
        assert result[key] == value, (key, result[key], value)
    assert result['point_state_bitwise'] and result['training']['objective']['cvar_weight'] == .35
    result['coupling_binding'] = proof
    (output/'matched_uq_done.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(angle=angle, checkpoint_sha256=result['checkpoint_sha256'],
                         point_frozen=True, wall_seconds=result['wall_seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument('--angle', type=int, choices=[0, 5, 10, 15, 20])
    scope.add_argument('--full-fit', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(); main(args.root, args.angle, args.preflight, args.full_fit)
