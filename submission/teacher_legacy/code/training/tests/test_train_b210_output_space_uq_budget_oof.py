from __future__ import annotations

from pathlib import Path
import sys

import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.ensemble_unet2d import (  # noqa: E402
    EnsembleUNet2d,
    EnsembleUNet2dConfig,
)
import train_b210_output_space_uq_allocator_oof as spatial  # noqa: E402
import train_b210_output_space_uq_budget_oof as subject  # noqa: E402


def model(*, allocator: bool) -> EnsembleUNet2d:
    value = EnsembleUNet2d(EnsembleUNet2dConfig(
        base=8, depth=3, groups=4, difference_steps=2,
        ensemble_members=3, uq_hidden_channels=8,
        uq_flow_context=True,
        output_space_uq_channels=8 if allocator else 0,
        output_space_uq_blocks=1,
        output_space_uq_budget_channels=8,
        output_space_uq_budget_blocks=1))
    value.set_normalizer_statistics(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3))
    return value


def test_budget_only_and_joint_trainability_are_disjoint() -> None:
    budget = model(allocator=False)
    spatial.configure_output_space_only(
        budget, train_allocator=False, train_budgeter=True)
    names = {
        name for name, parameter in budget.named_parameters()
        if parameter.requires_grad}
    assert names
    assert all(name.startswith("output_space_uq_budgeter.") for name in names)

    joint = model(allocator=True)
    spatial.configure_output_space_only(
        joint, train_allocator=True, train_budgeter=True)
    names = {
        name for name, parameter in joint.named_parameters()
        if parameter.requires_grad}
    assert any(name.startswith("output_space_uq_allocator.") for name in names)
    assert any(name.startswith("output_space_uq_budgeter.") for name in names)
    assert all(name.startswith((
        "output_space_uq_allocator.",
        "output_space_uq_budgeter.")) for name in names)


def test_selection_reports_spatial_increment_and_best_passing_arm(
        monkeypatch) -> None:
    rows = {
        ("budget_only", "source_u800"): {
            "native_sps_delta": 0.15, "positive_folds": 5,
            "worst_fold": 0.01},
        ("budget_only", "native_continue_u1600"): {
            "native_sps_delta": 0.06, "positive_folds": 4,
            "worst_fold": -0.01},
        ("budget_plus_spatial", "source_u800"): {
            "native_sps_delta": 0.18, "positive_folds": 5,
            "worst_fold": 0.02},
        ("budget_plus_spatial", "native_continue_u1600"): {
            "native_sps_delta": 0.09, "positive_folds": 4,
            "worst_fold": 0.00},
        ("budget_plus_spatial", "budget_only"): {
            "native_sps_delta": 0.03, "positive_folds": 4,
            "worst_fold": -0.01},
    }

    def fake_compare(_rows, candidate, control):
        return rows[(candidate, control)]

    monkeypatch.setattr(subject, "compare", fake_compare)
    result = subject.select([{}])
    assert result["pass"] is True
    assert result["selected"] == "budget_plus_spatial"
    assert result["spatial_increment"]["native_sps_delta"] == 0.03
