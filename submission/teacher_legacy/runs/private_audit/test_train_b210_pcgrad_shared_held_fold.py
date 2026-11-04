from __future__ import annotations

import pytest
import torch

import train_b210_pcgrad_shared_held_fold as subject


def test_pcgrad_projects_a_conflict_deterministically():
    corrected, info = subject.deterministic_pcgrad((torch.tensor([1., 0.]), torch.tensor([-1., 1.])))
    assert torch.allclose(corrected, torch.tensor([0.25, 0.75]))
    assert info["conflict_pairs"] == 2.0
    assert info["correction_norm"] == pytest.approx(0.125 ** 0.5)


def test_pcgrad_leaves_aligned_gradients_unchanged():
    gradients = (torch.tensor([1., 2.]), torch.tensor([2., 4.]))
    corrected, info = subject.deterministic_pcgrad(gradients)
    assert torch.equal(corrected, torch.tensor([1.5, 3.]))
    assert info["conflict_pairs"] == 0.0


def test_gram_implementation_matches_direct_cyclic_projection():
    gradients = torch.tensor([
        [1.0, -0.2, 0.3],
        [-0.8, 0.4, 0.1],
        [0.1, -0.9, 0.7],
        [0.5, 0.2, -0.6],
    ])
    projected = []
    for source in range(len(gradients)):
        value = gradients[source].clone()
        for offset in range(1, len(gradients)):
            target = (source + offset) % len(gradients)
            reference = gradients[target]
            dot = torch.dot(value, reference)
            if dot < 0:
                value -= dot / torch.dot(reference, reference) * reference
        projected.append(value)
    corrected, _info = subject.deterministic_pcgrad(gradients)
    assert torch.allclose(corrected, torch.stack(projected).mean(0), atol=1e-6)


def test_pcgrad_rejects_a_zero_group_gradient():
    with pytest.raises(FloatingPointError, match="singular"):
        subject.deterministic_pcgrad((torch.tensor([1., 0.]), torch.zeros(2)))


def test_shared_scope_matches_only_authorized_prefixes():
    model = torch.nn.Module()
    model.up = torch.nn.Linear(1, 1); model.decoder = torch.nn.Linear(1, 1)
    model.encoder = torch.nn.Linear(1, 1)
    assert len(subject.shared_decoder_parameters(model, require_audited_size=False)) == 4


def test_held_fold_is_fixed_to_aoa20():
    with pytest.raises(ValueError, match="AoA20"):
        subject.run(type("Args", (), {"held_aoa": 0})())
