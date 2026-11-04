"""Fixed held15 PC3D Real adaptation for the heterogeneous teacher experiment."""
import argparse
import gc
import json
from pathlib import Path

import pc3d_scaled_primary_contract as scaled
import train_pyramid_causal3d_sim_long as stage_s

scaled.install_sim_contract(stage_s)
import train_pyramid_causal3d_real_dense_aug as stage_r

scaled.install_real_contract(stage_r, stage_s)

BASELINE_SHA = 'ce84999755b685bfc84061cec3367ee324f64d71661ee5fcfd33320182c5b5fa'
MANIFEST_SHA = 'efe4c2fe3f2266a61407e90761b23a818f48ce21f0527a9c214bef42768de67e'
REFERENCE_NORMALIZER_SHA = '641e7fdfa05af5fefc195a8d7906e0cd44796e66b7d3cc4fac42ca99eabf7b1d'
REFERENCE_STREAM_SHA = 'e00aeee3c15d51a48a5401f4c226c336649d0778234ae0fee71953d15dcc2319'


def main(args):
    import train_b210_transfer_real4k as transfer
    from b210_stage_fold import select_rows
    from train_b210_full_released_real_support import load_full_rows

    assert args.execute and len(args.source_commit) == 40
    angle = getattr(args, 'held_aoa', 15)
    assert angle in (0, 15, 20)
    dense = transfer.load_runtime()
    torch, dr1, smu1 = dense.torch, dense.dr1, dense.smu1
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0')
    output = args.root / 'candidate'
    assert not output.exists()
    assert dr1.sha256(args.manifest) == MANIFEST_SHA
    control = json.loads(args.control_receipt.read_text())
    baseline_sha = control['checkpoint_sha256']
    assert baseline_sha == dr1.sha256(Path(control['checkpoint_path']))
    assert control['held_aoa'] == angle and control['updates'] == 4000
    assert control['arm'] == 'C' and control['seed'] == 4 and not control['held_data_read']
    assert control['start_sim_sha256'] == '628860c7d82239d942895f08a6fd72ede81feb6a7cad4de4e32e110ed2742b9b'
    if angle == 15:
        assert baseline_sha == BASELINE_SHA
    else:
        assert control['source_commit'] == args.source_commit
        assert Path(control['checkpoint_path']).parent.resolve() == (args.root / 'control').resolve()
    sim_result = json.loads((args.root / 'sim' / 'result.json').read_text())
    sim_path = args.root / 'sim' / 'checkpoints' / 'checkpoint_u00010000.pt'
    assert Path(sim_result['final_checkpoint']['path']).resolve() == sim_path.resolve()
    assert dr1.sha256(sim_path) == sim_result['final_checkpoint']['sha256']
    if angle == 15:
        assert sim_result['git_seal'] == stage_s.base.git_report()
    else:
        assert sim_result['final_checkpoint']['sha256'] == 'd80b44c2d182b14431e740a050480c8dab382d9e68ee4477a88983c43da9d454'

    rows, _ = dr1.load_manifest(args.manifest, dense.ft12.FORMAL_MANIFEST_SEMANTIC_SHA256)
    rows = select_rows(rows, angle)
    files = [row['file'] for row in rows]
    assert files == control['fit_files']
    if angle == 15: assert len(files) == 64
    audit = dense.PrefixAudit()
    trajectories, _, sparse = dr1.load_prefix_rows(args.real_dir, rows, 'fit', audit)
    assert len(sparse) == control['fit_windows'] and set(audit.opened) == set(files)
    if angle == 15: assert len(sparse) == 2000
    statistics = smu1.fold_statistics(smu1.window_sufficient_statistics(trajectories, sparse), range(len(sparse)))
    reference_path = Path(control['checkpoint_path']).parent / 'point_normalizer.pt'
    reference = torch.load(reference_path, map_location='cpu', weights_only=False)
    assert len(reference) == len(statistics) == 4
    assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(statistics, reference))
    normalizer = smu1.FoldNormalizer(statistics, device)
    normalizer_sha = torch.load(control['checkpoint_path'], map_location='cpu', weights_only=False)['normalizer_sha256']
    if angle == 15: assert normalizer_sha == REFERENCE_NORMALIZER_SHA
    del trajectories
    gc.collect()
    full_audit = dense.PrefixAudit()
    trajectories, _ = load_full_rows(args.real_dir, rows, 'fit', full_audit)
    assert set(full_audit.opened) == set(files)
    windows = dense.dense_windows(trajectories, 5)
    assert len(windows) == control['optimizer_windows'] == sum((r['frames'] - 40) // 5 + 1 for r in rows)
    if angle == 15: assert len(windows) == 10380
    dataset = dense.ft12.npt1.ensemble_train.common.repro.WindowDataset(trajectories, windows)
    schedule = smu1.paired_batch_schedule(range(len(windows)), windows, updates=4000, seed=4)
    assert schedule['sha256'] == control['sample_stream_sha256']
    if angle == 15: assert schedule['sha256'] == REFERENCE_STREAM_SHA
    indices = [row['sample_indices'] for row in schedule['rows']]

    def same_sample_schedule(samples, updates, batch_size):
        assert (samples, updates, batch_size) == (len(windows), 4000, 8)
        return indices

    stage_r.make_sample_schedule = same_sample_schedule
    model, sim_source = stage_r.load_pc3d_sim_checkpoint(sim_path, device)
    stage_r.install_normalizer(model, normalizer)
    assert sum(p.numel() for p in model.parameters()) == 25638640
    assert all(p.requires_grad for p in model.parameters())
    source = dict(experiment=('heterogeneous_pc3d_teacher_v1' if angle == 15 else 'heterogeneous_pc3d_confirm_v1'), source_commit=args.source_commit,
                  sim=sim_source, baseline_point_sha256=baseline_sha, held_aoa=angle,
                  fit_files=files, optimizer_windows=len(windows), normalizer_windows=len(sparse),
                  manifest_sha256=MANIFEST_SHA, baseline_normalizer_file_sha256=dr1.sha256(reference_path),
                  normalizer_reference_semantic_sha256=normalizer_sha,
                  normalizer_raw_statistics_sha256=smu1.statistics_digest(statistics),
                  original_c_sample_and_bootstrap_sha256=schedule['sha256'],
                  point_sample_indices_sha256=stage_r.schedule_sha(indices),
                  bootstrap_used=False, held_data_read=False)
    output.mkdir()
    stage_s.atomic_json_save(source, output / 'execution_contract.json')
    training, endpoint = stage_r.train(model, dataset, normalizer, source, output,
                                      updates=4000, batch_size=8, checkpoint_every=500, device=device)
    assert training['updates'] == 4000 and training['sample_cursor'] == 4000
    assert training['sample_schedule_sha256'] == source['point_sample_indices_sha256']
    result = dict(source=source, training=training, endpoint=endpoint,
                  decision='PC3D_REAL4000_SEALED_NOT_SCORED')
    stage_s.atomic_json_save(result, output / 'point_endpoint.json')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'manifest', 'real-dir', 'control-receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--held-aoa', type=int, choices=(0, 15, 20), default=15)
    parser.add_argument('--execute', action='store_true')
    main(parser.parse_args())
