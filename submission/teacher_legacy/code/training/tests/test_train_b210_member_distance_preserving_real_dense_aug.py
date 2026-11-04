from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys

import pytest
import torch
from torch import nn


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_member_distance_preserving_real_dense_aug as method


class TinyMembers(nn.Module):
    def __init__(self):
        super().__init__()
        self.member_point_heads = nn.ModuleList([
            nn.ModuleList([nn.Conv2d(2 + scale, 3, 1, bias=True)
                           for scale in range(method.EXPECTED_SCALES)])
            for _member in range(method.EXPECTED_MEMBERS)
        ])
        with torch.no_grad():
            for member, heads in enumerate(self.member_point_heads):
                for scale, head in enumerate(heads):
                    head.weight.fill_(0.01 * (1 + member + 2 * scale))
                    head.bias.fill_(10.0 + member)


def test_pairwise_readout_rms_is_scale_major_and_ignores_bias():
    model = TinyMembers()
    value = method.pairwise_readout_rms(model)
    assert value.shape == (method.EXPECTED_PAIRS,)
    assert torch.all(value > 0.0)
    before = value.clone()
    with torch.no_grad():
        for member in model.member_point_heads:
            for head in member:
                head.bias.add_(1000.0)
    assert torch.equal(before, method.pairwise_readout_rms(model))


def test_distance_loss_is_one_sided_and_bounded_at_collapse():
    reference = torch.ones(method.EXPECTED_PAIRS)
    loss_same, ratio_same = method.distance_preservation_loss(
        reference.clone(), reference)
    assert float(loss_same) == 0.0
    assert torch.equal(ratio_same, torch.ones_like(ratio_same))
    loss_larger, _ = method.distance_preservation_loss(
        1.2 * reference, reference)
    assert float(loss_larger) == 0.0
    loss_collapsed, ratio_collapsed = method.distance_preservation_loss(
        torch.zeros_like(reference), reference)
    assert float(loss_collapsed) == pytest.approx(1.0)
    assert float(ratio_collapsed.max()) == 0.0


def test_distance_preserver_adds_weight_only_gradient_after_shrinkage():
    model = TinyMembers()
    preserver = method.DistancePreserver(model)
    with torch.no_grad():
        for member in model.member_point_heads:
            for head in member:
                head.weight.mul_(0.5)
    values = preserver(7, model)
    assert values["member_distance_preservation_loss"] > 0.0
    assert values["member_distance_gradient_norm"] > 0.0
    assert preserver.gradient_seen
    assert all(head.weight.grad is not None
               for member in model.member_point_heads for head in member)
    assert all(head.bias.grad is None
               for member in model.member_point_heads for head in member)


def test_endpoint_distance_ratios_recover_uniform_scaling():
    model = TinyMembers()
    start = OrderedDict((name, value.detach().clone())
                        for name, value in model.state_dict().items())
    endpoint = OrderedDict((name, value.detach().clone())
                           for name, value in start.items())
    for name in endpoint:
        if name.endswith(".weight"):
            endpoint[name].mul_(0.75)
    ratios = method.endpoint_distance_ratios(start, endpoint)
    assert ratios == pytest.approx({
        "minimum": 0.75, "mean": 0.75, "maximum": 0.75})


def test_identical_arm_contracts_are_value_equal_but_independent():
    trajectories = [torch.zeros(60, 4, 8, 2)]
    windows = method.identical_dense_windows(trajectories, [])
    assert [(row.trajectory, row.start) for row in windows["control"]] == [
        (row.trajectory, row.start) for row in windows["candidate"]]
    assert windows["control"] is not windows["candidate"]
    augmentation = method.identical_dense_augmentation()
    assert augmentation["control"] == augmentation["candidate"]
    assert augmentation["control"] is not augmentation["candidate"]
