"""One preregistered random-initialization B210 Real12000 endpoint and screen."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import train_b210_held_residual_uq as held


BASELINE_SHA = 'ce84999755b685bfc84061cec3367ee324f64d71661ee5fcfd33320182c5b5fa'
NORMALIZER_SHA = '641e7fdfa05af5fefc195a8d7906e0cd44796e66b7d3cc4fac42ca99eabf7b1d'
STATISTICS_FILE_SHA = 'a1d67b2bb8712bea4f59d8c57001fcbd367f5faf2943a0b94b360de60eeef628'


def main(args):
    started = time.perf_counter()
    deadline = started + 4200
    assert not args.root.exists(), 'Refusing to overwrite an experiment'
    torch, dense, _, _ = held.runtime()
    import evaluate_b210_transfer as ev
    from models.ensemble_unet2d import EnsembleUNet2d
    from train_b210_full_released_real_support import load_full_rows
    ft12, smu1, trainer = dense.ft12, dense.smu1, dense.small4k
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0')
    assert torch.cuda.is_available()
    assert shutil.disk_usage(args.root.parent).free > 2 * 1024**3
    assert len(args.source_commit) == 40 and all(c in '0123456789abcdef' for c in args.source_commit)
    baseline = json.loads((args.control_dir / 'point_endpoint.json').read_text())
    assert baseline['checkpoint_sha256'] == BASELINE_SHA
    assert baseline['held_aoa'] == 15 and baseline['updates'] == 4000
    rows = ev.clean_rows(args.manifest)
    fit_rows = [row for row in rows if row['aoa'] != 15]
    fit_files = [row['file'] for row in fit_rows]
    assert fit_files == baseline['fit_files'] and len(fit_files) == 64
    assert sum(len(row['train_starts']) for row in fit_rows) == 2000
    statistics_path = args.control_dir / 'point_normalizer.pt'
    assert held.sha256(statistics_path) == STATISTICS_FILE_SHA
    # This is the only torch.load before training: four normalization tensors,
    # containing no learned model weights and no held15 statistics.
    statistics = torch.load(statistics_path, map_location='cpu', weights_only=False)
    normalizer = smu1.FoldNormalizer(statistics, device)
    assert ft12.normalizer_payload(normalizer)['sha256'] == NORMALIZER_SHA
    audit = dense.PrefixAudit()
    trajectories, metadata = load_full_rows(args.real_dir, fit_rows, 'fit', audit)
    assert audit.opened == fit_files
    windows = dense.dense_windows(trajectories, 5)
    assert len(windows) == 10380
    dataset = ft12.npt1.ensemble_train.common.repro.WindowDataset(trajectories, windows)
    schedule = smu1.paired_batch_schedule(range(len(windows)), windows, updates=12000, seed=4)
    schedule['updates'] = 12000
    ft12.npt1.ensemble_train.seed_everything(4)
    model = EnsembleUNet2d(dense.sim_tke.b210_config('candidate')).float()
    assert model.config.point_feature_refiner_channels == 0
    assert sum(p.numel() for p in model.parameters()) == 178705575
    ft12.npt1.ensemble_train.common.set_model_stats(model, normalizer)
    start = dict(initialization='random', seed=4, source_commit=args.source_commit,
                 pretrained_weight_files_read=0, sim_data_files_read=0,
                 initial_model_state_sha256=dense.dr1.model_state_sha256(model.state_dict()),
                 parameter_count=178705575, held_aoa=15,
                 normalizer_sha256=NORMALIZER_SHA)
    model.to(device)
    args.root.mkdir()
    shutil.copyfile(statistics_path, args.root / 'point_normalizer.pt')
    write = dense.dr1.write_json_atomic
    write(args.root / 'execution_contract.json', dict(
        decision='RANDOM_REAL12000_STARTED', start=start, updates=12000,
        manifest_sha256=held.sha256(args.manifest), normalizer_file_sha256=STATISTICS_FILE_SHA,
        baseline_checkpoint_sha256=BASELINE_SHA, optimizer_windows=10380,
        fit_files=fit_files, full_training_files_opened=audit.opened,
        sample_stream_sha256=schedule['sha256'], input_augmentation=dense.AUGMENTATION,
        wall_budget_seconds=4200, intermediate_checkpoints=False))
    trainer.UPDATES, trainer.HELD_AOA, trainer.SEED = 12000, 15, 4
    trainer.EXPERIMENT = 'b210_random_real12000_v1'
    trainer.ENDPOINT_SCHEMA = 'b210_random_real12000_endpoint_v1'
    trainer.PRIMARY_ENDPOINT = 'ema999_u12000_primary'
    access = dense.ArmSeal('random_real')
    training, seal = trainer.train_candidate(
        model, dataset, schedule, normalizer, device, deadline, args.root, start, access,
        input_augmentation=dense.AUGMENTATION, gradient_ratio_diagnostics_only=True,
        scheduler_t_max=12000, endpoint_filename='ema999_u12000_primary.pth')
    assert access.value == seal and training['updates'] == 12000
    assert time.perf_counter() < deadline
    write(args.root / 'training_result.json', dict(start=start, training=training, seal=seal,
          full_training_files_opened=audit.opened, elapsed_seconds=time.perf_counter()-started))
    write(args.root / 'point_endpoint.json', dict(
        schema='b210_random_real12000_point_receipt_v1', initialization='random',
        source_commit=args.source_commit, held_aoa=15, updates=12000, seed=4,
        fit_files=fit_files, fit_windows=2000, optimizer_windows=10380,
        normalizer_support='other_four_prefixes', optimizer_support='full_other_four_trajectories',
        checkpoint_path=seal['path'], checkpoint_sha256=seal['file_sha256'],
        normalizer_sha256=NORMALIZER_SHA, sample_stream_sha256=schedule['sha256'],
        ema_decay=.999, held_data_read=False, uq_pending=True,
        holdout_scope='Candidate training excludes held15 entirely; recipe selection has reused held15.'))
    del model, dataset, trajectories, normalizer, schedule
    gc.collect()
    torch.cuda.empty_cache()
    from screen_random_real12000 import main as screen
    screen(args)
    assert time.perf_counter() < deadline
    write(args.root / 'COMPLETE.json', dict(elapsed_seconds=time.perf_counter()-started,
          checkpoint_sha256=seal['file_sha256']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'control-dir', 'manifest', 'real-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    main(parser.parse_args())
