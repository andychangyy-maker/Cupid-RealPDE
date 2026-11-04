"""Reject cross-fold training and preserve exact complementary populations."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b210_stage_fold import select_rows
import train_b210_transfer_uq as uq


def test_stage_fold_partition_does_not_mutate_or_overlap():
    rows = [{'file': f'case_{a}.h5', 'aoa': a, 'train_starts': [0, 20]} for a in (0, 5, 10, 15, 20)]
    original = copy.deepcopy(rows)
    evaluated = []
    for held in (0, 5, 10, 15, 20):
        fit = select_rows(rows, held)
        test = select_rows(rows, held, evaluation=True)
        assert {r['file'] for r in fit}.isdisjoint(r['file'] for r in test)
        assert len(fit) == 4 and len(test) == 1
        evaluated.extend(r['file'] for r in test)
    assert rows == original and sorted(evaluated) == sorted(r['file'] for r in rows)
    with pytest.raises(ValueError):
        select_rows(rows, 7)
    with pytest.raises(ValueError):
        select_rows(rows[:-1], 0)


def test_point_payload_cannot_claim_a_different_stage_fold():
    receipt = {'sample_stream_sha256': 'stream', 'arm': 'C', 'start_sim_sha256': 'sim', 'held_aoa': 0}
    payload = {'schema': uq.POINT_SCHEMA, 'phase': 'real', 'iteration': 4000, 'endpoint': 'ema_u4000',
               'primary': True, 'ema': {'updates': 4000, 'decay': .999}, 'sample_stream_sha256': 'stream',
               'start': {'transfer_arm': 'C', 'checkpoint_sha256': 'sim', 'stage_held_aoa': 5}}
    with pytest.raises(ValueError, match='stage fold'):
        uq.validate_point_payload(payload, receipt)
