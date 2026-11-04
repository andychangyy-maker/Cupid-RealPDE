from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = CODE_ROOT / "training"
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_msclosure_dct_planned_forcing_scaled_long as subject  # noqa: E402


def test_exact_coefficient_match_has_finite_zero_gradient():
    truth = torch.randn(2, 2, 4, 5, 7)
    prediction = truth.detach().clone().requires_grad_()
    free = torch.ones(2, 5, 7, 1, dtype=torch.bool)
    error = subject.coefficient_relative_error(prediction, truth, free)
    assert torch.equal(error, torch.zeros_like(error))
    error.mean().backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) == 0


def test_recipe_reuses_scaled_control_split_stream_and_schedule():
    assert subject.FIT_AOAS == subject.control.FIT_AOAS
    assert subject.HELD_AOA == subject.control.HELD_AOA
    assert subject.SEED == subject.control.SEED
    assert subject.SCHEDULE_SEED == subject.control.SCHEDULE_SEED
    assert subject.UPDATES == 12_000
    assert subject.LOSS_WEIGHTS == (1.0, 4.0, 1.0)
    assert subject.MSE_WEIGHT == 0.25
    assert subject.COEFFICIENT_AUX_WEIGHT == 1.0 / 6.0
    formal_schedule = subject.control.make_schedule(
        3840, subject.UPDATES, subject.BATCH_SIZE)
    assert hashlib.sha256(formal_schedule.tobytes()).hexdigest() == (
        "8fd25c5b184b3764c29bc85a7b3e18c"
        "ccedcf742e067277938a505fea5afe277")


def test_truth_coefficient_projection_is_detached_from_label():
    model = subject.construct_model(torch.device("cpu"), tiny=True)
    target = torch.randn(1, 20, 8, 8, 3, requires_grad=True)
    predicted = torch.randn(1, 2, 4, 8, 8, requires_grad=True)
    free = torch.ones(1, 8, 8, 1, dtype=torch.bool)
    loss, _relative, truth = subject.coefficient_auxiliary_objective(
        predicted, target, free, model)
    assert truth.requires_grad is False
    loss.backward()
    assert predicted.grad is not None
    assert target.grad is None
