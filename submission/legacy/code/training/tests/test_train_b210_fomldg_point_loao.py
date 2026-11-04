from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = TRAINING_ROOT.parent
for path in (CODE_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_b210_fomldg_point_loao as subject  # noqa: E402


def _fit_rows() -> list[dict[str, object]]:
    """Two matched trajectories per fitting AoA; no held-AoA row is supplied."""
    return [
        {
            "file": f"aoa{aoa}_re{re}.h5", "aoa": aoa, "re": re,
            "train_starts": [0, 8, 16], "val_boundary": 80,
        }
        for aoa in subject.FIT_AOAS
        for re in (100_000, 200_000)
    ]


def test_1200_by_4_schedule_is_balanced_and_exactly_paired(monkeypatch):
    rows = _fit_rows()
    # The production digest protects the fixed manifest.  This unit fixture has
    # the same required pairing shape but intentionally synthetic identities.
    monkeypatch.setattr(subject, "FIT_ROWS_SHA256", subject.canonical_sha256(rows))
    schedule = subject.build_cross_condition_schedule(
        rows, updates=1_200, batch_size=4, seed=17)

    assert len(schedule["rows"]) == 1_200
    assert set(schedule["pair_counts"].values()) == {100}
    assert all(value["A"] == value["B"] == 300
               for value in schedule["role_counts"].values())
    assert len(schedule["ordered_aoa_pairs"]) == 12
    for row in schedule["rows"]:
        assert row["source_aoa"] != row["query_aoa"]
        assert len(row["A"]) == len(row["B"]) == 4
        assert set(row["rng"]) == {"A", "B"}
        for role in ("A", "B"):
            seeds = row["rng"][role]
            assert set(seeds) == {"gain", "blur", "noise", "dropout"}
            assert len(set(seeds.values())) == 4
            assert seeds == {
                kind: subject.derive_seed(17, row["update"], role, kind)
                for kind in ("gain", "blur", "noise", "dropout")
            }
        for support, query in zip(row["A"], row["B"]):
            assert support["re"] == query["re"]
            assert support["start"] == query["start"]
            assert support["aoa"] != subject.HELD_AOA
            assert query["aoa"] != subject.HELD_AOA


def test_bootstrap_masks_are_nonempty_and_deterministic():
    samples = [
        {"file": "one.h5", "start": 0},
        {"file": "two.h5", "start": 8},
    ]
    first = subject.bootstrap_mask(5, 9, "A", samples)
    assert first == subject.bootstrap_mask(5, 9, "A", samples)
    assert len(first) == len(samples)
    assert all(len(mask) == 4 and any(mask) for mask in first)


def test_pair_access_audit_requires_dual_seal_before_held_open(tmp_path):
    audit = subject.PairAccessAudit(["fit.h5"], ["held.h5"])
    held = {"file": "held.h5", "aoa": 10, "val_boundary": 80}
    with pytest.raises(RuntimeError, match="before dual seal"):
        audit.before_open(held, "held")

    for arm in ("candidate", "control"):
        endpoint = tmp_path / f"{arm}.pth"
        endpoint.write_bytes(arm.encode())
        audit.seal(arm, {"path": str(endpoint),
                         "file_sha256": subject.sha256(endpoint)})
    audit.before_open(held, "held")
    audit.record_held(held, [0, 40])
    report = audit.report()
    assert report["all_endpoints_sealed"] is True
    assert report["held_first_open_after_both_seals"] is True
    assert report["held_windows_opened"] == 2


class _TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.frozen = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)
        self.register_buffer("scale", torch.tensor(2.0))

    def forward(self, values):
        return self.linear(values) * self.scale


def _gradients(module: torch.nn.Module):
    loss = module(torch.tensor([[1.0, -2.0]])).square().sum()
    return torch.autograd.grad(loss, tuple(
        parameter for parameter in module.parameters() if parameter.requires_grad),
        create_graph=True)


def test_fast_parameter_mapping_is_detached_for_alpha_zero_and_positive():
    module = _TinyModule()
    grads = _gradients(module)
    original = dict(module.named_parameters())

    zero, buffers, zero_trainable = subject.build_fast_parameter_mapping(
        module, grads, alpha=0.0)
    assert tuple(zero_trainable) == tuple(
        zero[name] for name, parameter in module.named_parameters()
        if parameter.requires_grad)
    assert torch.equal(zero["linear.weight"], original["linear.weight"])
    assert zero["linear.weight"].data_ptr() != original["linear.weight"].data_ptr()
    assert all(value.is_leaf and value.grad_fn is None for value in zero_trainable)
    assert buffers["scale"].grad_fn is None
    assert zero["frozen"].requires_grad is False
    query_input = torch.tensor([[0.5, 1.5]])
    zero_output = torch.func.functional_call(module, (zero, buffers),
                                              (query_input,))
    direct_output = module(query_input)
    assert torch.allclose(zero_output, direct_output)
    zero_gradient = torch.autograd.grad(zero_output.sum(), zero_trainable)
    direct_gradient = torch.autograd.grad(
        direct_output.sum(), tuple(
            parameter for parameter in module.parameters()
            if parameter.requires_grad))
    assert all(torch.allclose(left, right)
               for left, right in zip(zero_gradient, direct_gradient))

    alpha = 0.25
    fast, _buffers, fast_trainable = subject.build_fast_parameter_mapping(
        module, grads, alpha=alpha)
    assert torch.allclose(fast["linear.weight"],
                          original["linear.weight"] - alpha * grads[0])
    assert all(value.is_leaf and value.grad_fn is None for value in fast_trainable)
    query_loss = torch.func.functional_call(module, (fast, _buffers),
                                             (torch.ones(1, 2),)).sum()
    query_grads = torch.autograd.grad(query_loss, fast_trainable)
    assert all(gradient.grad_fn is None for gradient in query_grads)


def test_install_average_gradient_handles_values_and_missing_terms():
    parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in range(3)]
    support = [torch.tensor([2.0]), None, torch.tensor([6.0])]
    query = [torch.tensor([4.0]), torch.tensor([8.0]), None]
    subject.install_average_gradient(parameters, support, query)
    assert parameters[0].grad.tolist() == pytest.approx([3.0])
    assert parameters[1].grad.tolist() == pytest.approx([4.0])
    assert parameters[2].grad.tolist() == pytest.approx([3.0])
    with pytest.raises(ValueError, match="lengths"):
        subject.install_average_gradient(parameters, support[:2], query)
