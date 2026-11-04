"""Reject the actual provenance mistakes that invalidate a Real holdout."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from train_b210_held_residual_uq import validate_pure_start


def pure_receipt():
    return dict(schema="b210_pure_sim_start_v1", training_visibility="pure_sim_only",
                updates=10000, seed=0, start="random_initialization", sealed=True,
                real_closure=False, pretext_checkpoint=None, checkpoint_selection=False,
                real_arrays_read=0, real_statistics_read=0, teacher_reads=0,
                p1_distill_weight=0, held_arrays_read=0, tail_arrays_read=0)


def test_phase_sim_does_not_make_a_causal_checkpoint_pure():
    receipt = pure_receipt()
    receipt.update(phase="sim", pretext_checkpoint="causal4k.pth")
    with pytest.raises(ValueError, match="without Real or teacher"):
        validate_pure_start(receipt)


@pytest.mark.parametrize("key,value", [("real_closure", True), ("teacher_reads", 1),
                                      ("real_statistics_read", 1), ("real_arrays_read", 1)])
def test_zero_distillation_weight_does_not_erase_other_real_access(key, value):
    receipt = pure_receipt()
    receipt[key] = value
    with pytest.raises(ValueError, match="without Real or teacher"):
        validate_pure_start(receipt)


def test_missing_visibility_evidence_is_not_zero_access():
    receipt = pure_receipt()
    del receipt["teacher_reads"]
    with pytest.raises(ValueError):
        validate_pure_start(receipt)
    validate_pure_start(pure_receipt())
