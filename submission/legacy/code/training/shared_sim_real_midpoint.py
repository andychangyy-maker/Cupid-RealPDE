"""One fixed parameter midpoint of Real seed4/5 sharing the exact Sim start."""
import torch
import train_b210_held_residual_uq as held


def materialize(root, replica, replica_receipt, control_receipt):
    reference_receipt = control_receipt['point_receipt']
    assert reference_receipt['seed'] == 4 and replica_receipt['seed'] == 5
    assert reference_receipt['held_aoa'] == replica_receipt['held_aoa'] == 15
    for key in ('start_sim_sha256', 'start_metadata_sha256', 'fit_files', 'optimizer_windows', 'ema_decay', 'updates'):
        assert reference_receipt[key] == replica_receipt[key], key
    assert reference_receipt['updates'] == 4000
    assert reference_receipt['sample_stream_sha256'] != replica_receipt['sample_stream_sha256']
    assert held.sha256(reference_receipt['checkpoint_path']) == reference_receipt['checkpoint_sha256']
    reference = torch.load(reference_receipt['checkpoint_path'], map_location='cpu', weights_only=False)
    assert reference['model_config'] == replica['model_config']
    assert reference['normalizer'] == replica['normalizer']
    state, other = reference['model_state_dict'], replica['model_state_dict']
    assert set(state) == set(other)
    averaged = {}
    for key, a in state.items():
        b = other[key]
        assert a.shape == b.shape and a.dtype == b.dtype, key
        if a.is_floating_point():
            averaged[key] = (a + b) * .5
            assert bool(torch.isfinite(averaged[key]).all()), key
        else:
            assert torch.equal(a, b), key
            averaged[key] = a.clone()
    parents = [{'seed': r['seed'], 'checkpoint_sha256': r['checkpoint_sha256'],
                'sample_stream_sha256': r['sample_stream_sha256']} for r in (reference_receipt, replica_receipt)]
    payload = {'schema': 'b210_shared_sim_real_midpoint_v1', 'model_state_dict': averaged,
               'model_config': reference['model_config'], 'normalizer': reference['normalizer'],
               'normalizer_sha256': reference['normalizer_sha256'],
               'parents': parents, 'weights': [.5, .5], 'updates_per_parent': 4000,
               'shared_start_sim_sha256': reference_receipt['start_sim_sha256'],
               'midpoint_source_sha256': held.sha256(__file__)}
    path = root / 'midpoint_point.pth'
    assert not path.exists()
    torch.save(payload, path)
    receipt = {'schema': 'b210_shared_sim_real_midpoint_v1_receipt', 'parents': parents,
               'checkpoint_path': str(path), 'checkpoint_sha256': held.sha256(path),
               'normalizer_sha256': reference['normalizer_sha256'],
               'midpoint_source_sha256': held.sha256(__file__),
               'weights': [.5, .5], 'held_aoa': 15, 'fit_files': reference_receipt['fit_files']}
    receipt_path = root / 'midpoint_point_receipt.json'
    held.write_json(receipt_path, receipt)
    return payload, receipt, receipt_path
