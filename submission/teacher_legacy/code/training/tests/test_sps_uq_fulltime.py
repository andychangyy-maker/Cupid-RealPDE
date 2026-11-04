"""Check the changed data boundary and the official SPS decomposition."""
from pathlib import Path
import sys
import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
for folder in ('code', 'code/training', 'code/evaluation'):
    sys.path.insert(0, str(ROOT / folder))
import train_b210_transfer_uq as training
import run_sps_uq_fulltime as study


def test_full_support_reads_only_selected_files_and_includes_fit_tail(tmp_path):
    row = {'file': '100_0.h5', 'frames': 100, 'val_boundary': 60,
           'train_starts': [0, 20], 're': 100, 'aoa': 0}
    with h5py.File(tmp_path / row['file'], 'w') as handle:
        values = np.arange(100, dtype=np.float32)[:, None, None] * np.ones((1, 4, 4))
        handle['u'] = values; handle['v'] = values + 1
        handle['re'] = 100; handle['aoa'] = 0
    (tmp_path / '100_5.h5').write_bytes(b'held file must never be opened')
    prefix, _, pw, pa = training.load_uq_rows(tmp_path, [row], False)
    full, _, fw, fa = training.load_uq_rows(tmp_path, [row], True)
    assert prefix[0].shape[0] == 60 and full[0].shape[0] == 100
    assert [w.start for w in pw] == [0, 20]
    assert [w.start for w in fw] == [0, 20, 40, 60]
    assert pa['tail_arrays_read'] == 0 and fa['tail_arrays_read'] == 1
    assert pa['prefix_only'] and not fa['prefix_only']
    assert fa['opened'] == pa['opened'] == ['100_0.h5']
    assert fa['held_arrays_read'] == 0


def test_full_time_rejects_a_non_bound_point_protocol():
    with pytest.raises(SystemExit):
        training.parse_args(['--label', 'C', '--uq-full-time'])
    args = training.parse_args(['--label', 'full', '--point-label', 'C', '--held-aoa', '5',
                                '--full-real-point', '--initialization', 'point', '--uq-full-time'])
    assert training.plan(args)['uq_full_time']


def test_sps_loss_parts_match_scorer_with_mask_asymmetry_and_misses():
    official = study.replay.import_file(study.replay.SCORER, 'fulltime_test_scoring')
    rng = np.random.default_rng(23)
    truth = rng.normal(.1, .05, (2, 20, 32, 64, 3)).astype(np.float32)
    truth[..., 2] = 0
    truth[:, :, :3, :4, :2] = 0
    pred = truth + rng.normal(0, .008, truth.shape).astype(np.float32)
    pred[..., 2] = 0
    lower, upper = pred-.006, pred+.009
    lower[..., 2] = upper[..., 2] = 0
    parts = study.loss_parts(official, pred, truth, lower, upper)
    score, _ = official.aggregate_sps(pred, truth, 2, lower, upper)
    loss = sum(parts[k] for k in ('accuracy_loss', 'miss_loss', 'width_loss')) / parts['count']
    assert abs(score + loss - 1) < 2e-7
    assert all(parts[k] > 0 for k in ('accuracy_loss', 'miss_loss', 'width_loss'))
