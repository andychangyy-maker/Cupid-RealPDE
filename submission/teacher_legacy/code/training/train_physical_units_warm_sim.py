"""Adapt the sealed winning C Sim initialization to physical units, then refit Real."""
import argparse
import json
from pathlib import Path
import sys

import train_b210_pure_sim_start as pure


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--initial-checkpoint', type=Path, required=True)
    initial, remaining = parser.parse_known_args()
    args = pure.parse_args(remaining)
    assert args.execute and args.teacher_cache is None
    initial_sha = '628860c7d82239d942895f08a6fd72ede81feb6a7cad4de4e32e110ed2742b9b'
    assert pure.sha256(initial.initial_checkpoint) == initial_sha
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import train_b210_np_functional_sim as sim
    from models.physical_units_b210 import install
    payload = torch.load(str(initial.initial_checkpoint), map_location='cpu', weights_only=False)
    assert payload['phase'] == 'sim' and payload['iteration'] == 10000
    assert payload['training_config']['method'] == 'b210_causal_denoising_pretrain_then_standard_stride20x2_v1'
    common = sim.rehearsal.ensemble_train.common
    original_stats, original_plan = common.set_model_stats, pure.plan
    loaded = False

    def stats(model, normalizer):
        nonlocal loaded, payload
        assert not loaded, 'warm initialization must happen exactly once'
        current_config = model.architecture_config()
        old_config = payload['model_config']
        missing = set(current_config) - set(old_config)
        assert missing == {'point_feature_refiner_channels', 'uq_full_history_context', 'uq_sparse_history_context'}
        assert all(current_config[key] == 0 for key in missing)
        assert all(current_config[key] == value for key, value in old_config.items())
        model.load_state_dict(payload['model_state_dict'], strict=True)
        original_stats(model, normalizer)
        install(model, normalizer, uq_global=True)
        loaded = True
        payload = None

    def plan(options):
        value = original_plan(options)
        value.update(start='sealed_C_causal4k_standard20',
                     training_visibility='current_stage_sim_only_inherited_C_real_prefix_ancestry',
                     initial_checkpoint=str(initial.initial_checkpoint), initial_checkpoint_sha256=initial_sha,
                     inherited_pretraining='C causal4k plus two standard Sim10k stages; includes historical Real-prefix, teacher and closure ancestry',
                     joint_uq_physical_reference=True)
        return value

    common.set_model_stats, pure.plan = stats, plan
    pure.SCHEMA = 'physical_units_csim_warmstart_v1'
    pure.STATUS = 'WARM_C_PHYSICAL_UNITS_SIM10000_SEALED'
    result = pure.execute(args)
    assert loaded
    result['representation'] = 'physical_units_b210_v1'
    result['warm_wrapper_sha256'] = pure.sha256(__file__)
    result['units_source_sha256'] = pure.sha256(Path(__file__).resolve().parents[1]/'models/physical_units_b210.py')
    (args.output_dir/'sim_endpoint.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ['decision', 'checkpoint_path', 'checkpoint_sha256', 'initial_checkpoint_sha256']}))


if __name__ == '__main__':
    main()
