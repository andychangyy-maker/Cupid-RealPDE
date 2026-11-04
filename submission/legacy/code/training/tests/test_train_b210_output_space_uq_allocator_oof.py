from __future__ import annotations

import json
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
import train_b210_output_space_uq_allocator_oof as subject  # noqa: E402
import train_b210_output_space_uq_allocator_refit as refit  # noqa: E402


def model() -> EnsembleUNet2d:
    value = EnsembleUNet2d(EnsembleUNet2dConfig(
        base=8, depth=3, groups=4, difference_steps=2,
        ensemble_members=3, uq_hidden_channels=8,
        uq_flow_context=True, output_space_uq_channels=8,
        output_space_uq_blocks=1))
    value.set_normalizer_statistics(
        torch.zeros(3), torch.zeros(3), torch.ones(3), torch.ones(3))
    return value


def test_configure_allocator_only_freezes_every_other_responsibility() -> None:
    value = model()
    parameters = subject.configure_allocator_only(value)
    names = {
        name for name, parameter in value.named_parameters()
        if parameter.requires_grad}
    assert parameters
    assert names
    assert all(name.startswith("output_space_uq_allocator.") for name in names)
    assert all(
        parameter.requires_grad is False
        for name, parameter in value.named_parameters()
        if not name.startswith("output_space_uq_allocator."))


def test_allocator_endpoint_roundtrip_and_frozen_hash(tmp_path: Path) -> None:
    value = model()
    before = subject.full.dr1.model_state_sha256(
        subject.frozen_without_allocator_state(value))
    with torch.no_grad():
        value.output_space_uq_allocator[-1].bias.add_(0.2)
    after = subject.full.dr1.model_state_sha256(
        subject.frozen_without_allocator_state(value))
    assert after == before
    seal = subject.save_endpoint(
        tmp_path / "endpoint.pth", value, held_aoa=10,
        arm="output_allocator", training={"updates": 800})
    payload = subject.load_endpoint(
        Path(seal["path"]), seal["file_sha256"])
    assert payload["uq_state_dict"]
    assert any(
        key.startswith("output_space_uq_allocator.")
        for key in payload["uq_state_dict"])


def test_promotion_requires_architecture_to_beat_both_controls(monkeypatch) -> None:
    comparisons = {
        "source_u800": {
            "native_sps_delta": 0.12, "positive_folds": 5,
            "worst_fold": 0.01},
        "native_continue_u1600": {
            "native_sps_delta": 0.02, "positive_folds": 4,
            "worst_fold": -0.01},
    }

    def fake_compare(_rows, _candidate, control):
        return comparisons[control]

    monkeypatch.setattr(subject, "compare", fake_compare)
    result = subject.promotion([{}])
    assert result["conditions"]["pooled_vs_source"] is True
    assert result["conditions"]["pooled_vs_native_continue"] is False
    assert result["pass"] is False


def test_refit_requires_exact_oof_gate_and_architecture(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({
        "schema": subject.SCHEMA,
        "decision": "PASS_OUTPUT_SPACE_UQ_ALLOCATOR_OOF",
        "gate": {"pass": True},
        "architecture": {
            "channels": subject.CHANNELS,
            "blocks": subject.BLOCKS,
            "log_limit": subject.LOG_LIMIT,
        },
    }))
    sha = subject.full.dr1.sha256(path)
    row = refit.validate_oof_summary(path, sha)
    assert row["gate"]["pass"] is True

    path.write_text(path.read_text().replace('"channels": 48', '"channels": 32'))
    sha = subject.full.dr1.sha256(path)
    try:
        refit.validate_oof_summary(path, sha)
    except RuntimeError as error:
        assert "architecture changed" in str(error)
    else:
        raise AssertionError("changed allocator architecture was accepted")
